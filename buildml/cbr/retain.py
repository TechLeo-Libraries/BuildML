"""Leakage-safe retain: add newly labeled cases into the case memory."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.cbr.cases import Case, CaseBase, encode_categoricals
from buildml.cbr.features import (
    matrix_from_frame,
    standardize_apply,
)
from buildml.cbr.results import CbrPlan, CbrRetainResult
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan


def retain_cbr(
    dataset: Dataset,
    plan: CbrPlan,
    split_plan: SplitPlan | None,
    *,
    labeled_frame: pd.DataFrame,
    solution_column: str | None = None,
    source_disclosure: str,
    allow_overlap_with_train: bool = True,
) -> tuple[CbrPlan, CbrRetainResult]:
    """Append labeled cases to the case base with leakage checks.

    Leakage discipline
    ------------------
    - Refuses any row whose index appears in Session **validation** or **test**.
    - Requires a non-empty ``source_disclosure`` explaining where labels came from
      (e.g. human labeling of a production stream — never Session holdout).
    - Train-overlap rows may be skipped (default) or kept when
      ``allow_overlap_with_train=True`` (still skipped if already present by index).

    Honesty: this is a lite retain hook, not a full revise/retain cognitive cycle.
    """
    if not str(source_disclosure).strip():
        raise ValidationError(
            "retain_cbr requires a non-empty source_disclosure describing "
            "where the new labels came from (must not be Session holdout)."
        )
    if labeled_frame is None or len(labeled_frame) == 0:
        raise ValidationError("retain_cbr requires a non-empty labeled_frame.")

    sol_col = solution_column or plan.target_column
    if sol_col not in labeled_frame.columns:
        raise ValidationError(
            f"labeled_frame missing solution column {sol_col!r}."
        )
    if labeled_frame[sol_col].isna().any():
        raise ValidationError("retain_cbr refuses null solutions.")

    forbidden = _holdout_label_index_set(dataset, split_plan)
    existing = {c.row_index for c in plan.case_base.cases}
    train_idx = _partition_label_index_set(dataset, split_plan, "train")

    cols = list(plan.columns)
    cat_cols = list(plan.categorical_columns)
    for c in cols + cat_cols:
        if c not in labeled_frame.columns:
            raise ValidationError(
                f"labeled_frame missing feature column {c!r} required by CbrPlan."
            )

    # Transform with train-fit params (never refit on retained rows).
    if cols:
        x = matrix_from_frame(labeled_frame, cols)
        mem = plan.case_base
        if (
            plan.standardize
            and plan.metric != "mixed"
            and mem.numeric_mean_ is not None
            and mem.numeric_scale_ is not None
        ):
            x = standardize_apply(x, mem.numeric_mean_, mem.numeric_scale_)
    else:
        x = np.zeros((len(labeled_frame), 0), dtype=float)

    if cat_cols:
        codes = np.column_stack(
            [
                encode_categoricals(
                    labeled_frame[c].tolist(),
                    vocab,
                )
                for c, vocab in zip(
                    cat_cols, plan.case_base.cat_vocabularies_, strict=True
                )
            ]
        )
    else:
        codes = np.zeros((len(labeled_frame), 0), dtype=int)

    new_cases: list[Case] = []
    new_x: list[np.ndarray] = []
    new_cat: list[np.ndarray] = []
    n_skipped = 0
    base_id = plan.case_base.n_cases

    for i, idx in enumerate(labeled_frame.index):
        if idx in forbidden:
            raise ValidationError(
                f"retain_cbr refused row index {idx!r}: it belongs to Session "
                "validation/test. Case memory must never absorb holdout labels."
            )
        if idx in existing:
            n_skipped += 1
            continue
        if (not allow_overlap_with_train) and idx in train_idx:
            n_skipped += 1
            continue
        sol = labeled_frame[sol_col].iloc[i]
        if plan.task == "regression":
            sol = float(sol)
        new_cases.append(
            Case(
                case_id=f"retained-{base_id + len(new_cases)}",
                row_index=idx,
                solution=sol,
                numeric_features=tuple(float(v) for v in x[i]) if cols else (),
                categorical_features=tuple(
                    labeled_frame[c].iloc[i] for c in cat_cols
                ),
                source="retained",
                disclosures=(str(source_disclosure),),
            )
        )
        new_x.append(x[i])
        new_cat.append(codes[i] if codes.shape[1] else np.zeros(0, dtype=int))

    disclosures = [
        f"Retained {len(new_cases)} case(s); skipped {n_skipped}.",
        f"source_disclosure: {source_disclosure}",
        "Holdout (validation/test) indices are refused. Train-fit distance "
        "transforms were reused (not refit on retained rows).",
        "Honesty: lite retain hook — not a full CBR revise/retain research cycle.",
    ]

    if not new_cases:
        result = CbrRetainResult(
            n_added=0,
            n_cases_after=plan.case_base.n_cases,
            n_skipped=n_skipped,
            disclosures=tuple(disclosures),
            warnings=("No new cases retained.",),
        )
        return plan, result

    mem = plan.case_base
    x_all = np.vstack([mem.numeric_matrix, np.asarray(new_x, dtype=float)])
    if mem.categorical_matrix.shape[1] > 0:
        cat_all = np.vstack(
            [mem.categorical_matrix, np.asarray(new_cat)]
        )
    else:
        cat_all = np.zeros((x_all.shape[0], 0), dtype=int)

    new_base = CaseBase(
        cases=tuple([*mem.cases, *new_cases]),
        numeric_matrix=x_all,
        categorical_matrix=cat_all,
        numeric_columns=mem.numeric_columns,
        categorical_columns=mem.categorical_columns,
        metric=mem.metric,
        numeric_mean_=mem.numeric_mean_,
        numeric_scale_=mem.numeric_scale_,
        numeric_ranges_=mem.numeric_ranges_,
        cat_vocabularies_=mem.cat_vocabularies_,
        disclosures=tuple([*mem.disclosures, *disclosures]),
        n_retained=mem.n_retained + len(new_cases),
    )
    new_plan = CbrPlan(
        task=plan.task,
        metric=plan.metric,
        reuse=plan.reuse,
        adapt=plan.adapt,
        k=plan.k,
        columns=plan.columns,
        categorical_columns=plan.categorical_columns,
        target_column=plan.target_column,
        n_train_rows=plan.n_train_rows,
        case_base=new_base,
        classes_=plan.classes_,
        label_encoder_=plan.label_encoder_,
        distance_eps=plan.distance_eps,
        standardize=plan.standardize,
        disclosures=tuple([*plan.disclosures, *disclosures]),
        warnings=plan.warnings,
        used_reduce_components=plan.used_reduce_components,
        config=dict(plan.config),
    )
    result = CbrRetainResult(
        n_added=len(new_cases),
        n_cases_after=new_base.n_cases,
        n_skipped=n_skipped,
        disclosures=tuple(disclosures),
        warnings=(),
    )
    return new_plan, result


def _partition_label_index_set(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: str,
) -> set[Any]:
    """Map SplitPlan positional indices → dataset frame label indices."""
    if split_plan is None:
        return set()
    pos = getattr(split_plan, f"{partition}_indices", ()) or ()
    if not pos:
        return set()
    frame = dataset._ensure_pandas()
    return set(frame.index[list(pos)].tolist())


def _holdout_label_index_set(
    dataset: Dataset, split_plan: SplitPlan | None
) -> set[Any]:
    """Label indices belonging to validation or test (never retain these)."""
    out: set[Any] = set()
    out |= _partition_label_index_set(dataset, split_plan, "validation")
    out |= _partition_label_index_set(dataset, split_plan, "test")
    return out


def retain_from_indices(
    dataset: Dataset,
    plan: CbrPlan,
    split_plan: SplitPlan | None,
    *,
    row_indices: Sequence[Any],
    source_disclosure: str,
) -> tuple[CbrPlan, CbrRetainResult]:
    """Retain rows from the live Dataset by index (holdout indices refused)."""
    if not row_indices:
        raise ValidationError("row_indices must be non-empty.")
    frame = dataset._ensure_pandas()
    missing = [i for i in row_indices if i not in frame.index]
    if missing:
        raise ValidationError(
            f"retain_cbr row_indices not in dataset: {missing[:5]!r}..."
        )
    labeled = frame.loc[list(row_indices)]
    return retain_cbr(
        dataset,
        plan,
        split_plan,
        labeled_frame=labeled,
        solution_column=plan.target_column,
        source_disclosure=source_disclosure,
    )
