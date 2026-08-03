"""Add newly resolved cases to memory, with the guards that keep it honest.

Retention is what distinguishes case-based reasoning from a static model. Solve
a case, find out how it actually turned out, keep it — and the reasoner improves
without any retraining. In production this is the loop that makes the method
attractive.

It is also the easiest way to destroy the evaluation. Retaining a validation or
test row puts it in memory, where it becomes its own nearest neighbour at
distance zero, and the holdout score silently becomes a measurement of storage.
The damage is permanent and invisible: nothing looks wrong, the number just goes
up.

So the guards here are strict rather than convenient. Holdout indices are
refused outright, not warned about. A ``source_disclosure`` is mandatory,
forcing the caller to state in writing where the labels came from — the point
being that having to write "human review of production traffic" makes it hard to
retain a holdout partition without noticing.

This is a retain hook, not a full revise-and-retain cognitive cycle. Cases are
added as given; nothing repairs a failed solution before storing it.

See Also
--------
buildml.cbr.results.CbrRetainResult : What comes back.
buildml.cbr.evaluate.evaluate_cbr : Re-scoring after memory has grown.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.cbr.adapters.industry_ann import add_vectors_to_ann_index
from buildml.cbr.adapters.text_embed import embed_text_cases
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
    """Add labelled cases to memory, refusing anything that would corrupt evaluation.

    Encodes the new rows with the plan's train-fitted transforms, appends them
    as retained cases, and updates any search index in place. Returns an updated
    plan and a report of what was admitted and what was refused.

    Parameters
    ----------
    dataset:
        The source data, used to resolve which indices belong to holdout
        partitions.
    plan:
        The fitted reasoner whose memory is being extended.
    split_plan:
        Partition membership, which is what makes the holdout check possible.
    labeled_frame:
        The new cases. Must carry the plan's feature columns and a solution
        column, with no nulls in either.
    solution_column:
        Where the outcome lives, defaulting to the plan's target column.
    source_disclosure:
        Mandatory, non-empty. Where these labels came from — human review, a
        resolved support ticket, a settled transaction. Never a Session holdout
        partition.
    allow_overlap_with_train:
        Whether rows already represented in train may be retained again. Rows
        already present by index are skipped either way.

    Returns
    -------
    tuple
        ``(plan, result)`` — the plan with extended memory, and the counts of
        what was added and skipped.

    Raises
    ------
    ValidationError
        If ``source_disclosure`` is empty, ``labeled_frame`` is empty, the
        solution column is missing or contains nulls, a feature column is
        absent, or any row's index belongs to validation or test.

    Notes
    -----
    **Holdout rows are refused, not warned about.** Retaining one makes it its
    own nearest neighbour and inflates every later holdout score, with no
    symptom to notice. There is no override.

    **``source_disclosure`` exists to make the mistake hard to make silently.**
    Being required to write down the provenance is a small friction that
    surfaces "these came from the test set" before the data does.

    **Distance transforms are not refitted.** Standardisation and vocabularies
    stay as fitted on train, which is what keeps evaluation comparable — and
    means a sustained distribution shift in retained cases is scaled by
    increasingly stale statistics. Refit periodically.

    **Identity is the frame index, not the feature values.** A new frame built
    with a default ``RangeIndex`` collides with the row indices already in
    memory, and every row is skipped as a duplicate — ``n_added`` comes back
    zero for data that is genuinely new. Carry the original indices, or assign
    fresh ones outside the dataset's range.

    **Re-evaluate after retaining.** The previous holdout score described the
    memory as it was.

    Examples
    --------
    Retain cases resolved by human review::

        plan, result = retain_cbr(
            dataset, plan, split_plan,
            labeled_frame=reviewed,
            source_disclosure="Human review of production traffic, Q3.",
        )
        print(result.n_added, result.n_skipped, result.n_cases_after)

    See Also
    --------
    retain_from_indices : Retaining rows already in the dataset.
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

    search_all, ann_index, ann_library = _extend_search_artifacts(
        plan,
        labeled_frame=labeled_frame.loc[[c.row_index for c in new_cases]],
        new_numeric=np.asarray(new_x, dtype=float),
        start_id=mem.n_cases,
    )

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
        search_matrix_=search_all,
        ann_index_=ann_index,
        ann_library_=ann_library,
        text_embedder_id_=mem.text_embedder_id_,
        torch_encoder_=mem.torch_encoder_,
        disclosures=tuple([*mem.disclosures, *disclosures]),
        n_retained=mem.n_retained + len(new_cases),
    )
    new_plan = CbrPlan(
        task=plan.task,
        backend=plan.backend,
        metric=plan.metric,
        reuse=plan.reuse,
        adapt=plan.adapt,
        k=plan.k,
        columns=plan.columns,
        categorical_columns=plan.categorical_columns,
        text_columns=plan.text_columns,
        text_model_name=plan.text_model_name,
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


def _extend_search_artifacts(
    plan: CbrPlan,
    *,
    labeled_frame: pd.DataFrame,
    new_numeric: np.ndarray,
    start_id: int,
) -> tuple[np.ndarray, Any, str | None]:
    mem = plan.case_base
    base_search = mem.search_matrix_
    if base_search is None:
        base_search = mem.numeric_matrix
    backend = str(plan.backend)
    if backend == "embedding" and plan.text_columns:
        new_search, _ = embed_text_cases(
            labeled_frame,
            plan.text_columns,
            model_name=plan.text_model_name,
            numeric_matrix=new_numeric if new_numeric.shape[1] else None,
        )
    elif backend == "torch" and mem.torch_encoder_ is not None:
        from buildml.cbr.adapters.torch_metric import encode_with_torch

        new_search = encode_with_torch(
            mem.torch_encoder_,
            new_numeric,
            device=str(plan.config.get("device", "cpu")),
        )
    else:
        new_search = np.asarray(new_numeric, dtype=float)
    search_all = np.vstack([base_search, new_search])
    ann_index = mem.ann_index_
    ann_library = mem.ann_library_
    if ann_index is not None and ann_library:
        ann_index = add_vectors_to_ann_index(
            ann_index,
            ann_library,
            new_search,
            start_id=start_id,
        )
    return search_all, ann_index, ann_library


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
    """Retain rows that are already in the dataset, named by index.

    A convenience over :func:`retain_cbr` for the common case where the newly
    resolved cases are rows you already have — labelled after the fact, or
    corrected. Pulls them from the dataset and hands them over with the same
    guards applied.

    Parameters
    ----------
    dataset:
        The source data.
    plan:
        The fitted reasoner.
    split_plan:
        Partition membership, used for the holdout check.
    row_indices:
        Which rows to retain. Must all exist in the dataset.
    source_disclosure:
        Mandatory, non-empty. Where the labels came from.

    Returns
    -------
    tuple
        ``(plan, result)`` — the extended plan and the retention report.

    Raises
    ------
    ValidationError
        If ``row_indices`` is empty, any index is absent from the dataset, or
        any names a validation or test row.

    Notes
    -----
    **Naming a holdout index still fails.** Convenience does not relax the
    guard; the check happens in :func:`retain_cbr` either way.

    **The solution comes from the plan's target column**, so these rows must
    already carry a resolved outcome.

    See Also
    --------
    retain_cbr : The general form, for cases from outside the dataset.
    """
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
