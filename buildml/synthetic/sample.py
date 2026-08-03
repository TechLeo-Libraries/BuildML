"""Sample from a frozen synthesizer; optional train-extend merge with provenance."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.ingest.detect import schema_from_dataframe
from buildml.synthetic.adapters.sdv import SdvTabularGenerator
from buildml.synthetic.models import (
    BootstrapGenerator,
    GaussianCopulaGenerator,
    SmoteGenerator,
)
from buildml.synthetic.results import SyntheticSampleResult, SynthesizerPlan
from buildml.synthetic.types import MergeMode


def sample_synthetic(
    plan: SynthesizerPlan,
    *,
    n: int | None = None,
    random_state: int | None = None,
    condition: dict[str, Any] | None = None,
) -> SyntheticSampleResult:
    """Draw ``n`` rows from a fitted synthesizer (does not mutate Session).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
plan:
    Fitted plan object carrying model state and feature contract.
n:
    n (int | None).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
condition:
    condition (dict[str, Any] | None).

Returns
-------
SyntheticSampleResult
    Serializable result summary (SyntheticSampleResult) for history recording.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if plan is None or plan.generator_ is None:
        raise ValidationError("No fitted synthesizer. Call fit_synthesizer(...) first.")
    n_rows = int(plan.n_rows_fitted if n is None else n)
    if n_rows < 1:
        raise ValidationError("n must be >= 1.")
    if condition and plan.method != "gaussian_copula":
        raise ValidationError(
            "condition= is currently supported for method='gaussian_copula' only."
        )

    generator = plan.generator_
    if isinstance(generator, BootstrapGenerator):
        frame = generator.sample(n_rows, random_state=random_state)
    elif isinstance(generator, GaussianCopulaGenerator):
        frame = generator.sample(
            n_rows, random_state=random_state, condition=condition
        )
    elif isinstance(generator, SmoteGenerator):
        frame = generator.sample(n_rows, random_state=random_state)
    elif isinstance(generator, SdvTabularGenerator):
        frame = generator.sample(
            n_rows, random_state=random_state, condition=condition
        )
    else:
        raise ValidationError(
            f"Unsupported generator type on SynthesizerPlan: {type(generator)!r}."
        )

    disclosures = [
        f"Sampled n={n_rows} from method={plan.method!r} "
        f"(fitted on train n={plan.n_rows_fitted}).",
        "Frame returned without mutating Session roles/splits "
        "(merge_mode='none'). Pass merge_mode='extend_train' to append.",
        "Synthetic rows are not real observations — provenance should be tracked.",
    ]
    return SyntheticSampleResult(
        method=plan.method,
        n_rows=int(len(frame)),
        columns=tuple(frame.columns),
        frame=frame,
        merged=False,
        merge_mode="none",
        provenance_column=None,
        disclosures=tuple(disclosures),
        warnings=tuple(plan.warnings),
    )


def merge_synthetic_into_train(
    dataset: Dataset,
    split_plan: SplitPlan,
    sample_frame: pd.DataFrame,
    *,
    provenance_column: str = "_synthetic",
    mark_value: Any = True,
) -> tuple[Dataset, SplitPlan, str]:
    """Append synthetic rows to **train only**; holdouts unchanged.

The provenance column is assigned role ``ignore`` so it cannot silently
become a modeling feature. Existing roles for real columns are preserved;
no roles are invented for missing columns.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
sample_frame:
    sample frame (pd.DataFrame).
provenance_column:
    provenance column (str).
mark_value:
    mark value (Any).

Returns
-------
tuple[Dataset, SplitPlan, str]
    Tuple of results (tuple[Dataset, SplitPlan, str]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if sample_frame is None or sample_frame.empty:
        raise ValidationError("Cannot merge an empty synthetic frame.")
    if provenance_column in dataset.frame.columns:
        raise ValidationError(
            f"provenance_column {provenance_column!r} already exists on the dataset. "
            "Choose another name to avoid silently overwriting real data."
        )

    ordered_cols = list(dataset.columns)
    syn = sample_frame.copy()
    # Align columns: require modeled columns present; fill extras with NA
    for col in ordered_cols:
        if col not in syn.columns:
            syn[col] = pd.NA
    syn = syn[ordered_cols]
    syn[provenance_column] = mark_value

    train = dataset.frame.iloc[list(split_plan.train_indices)].reset_index(drop=True)
    valid = (
        dataset.frame.iloc[list(split_plan.validation_indices)].reset_index(drop=True)
        if split_plan.validation_indices
        else None
    )
    test = dataset.frame.iloc[list(split_plan.test_indices)].reset_index(drop=True)

    train = train.copy()
    train[provenance_column] = False if mark_value is not False else 0
    if valid is not None:
        valid = valid.copy()
        valid[provenance_column] = False if mark_value is not False else 0
    test = test.copy()
    test[provenance_column] = False if mark_value is not False else 0

    # Fingerprints for leakage guard (holdouts byte-stable aside from provenance col)
    valid_fp = None if valid is None else valid.drop(columns=[provenance_column])
    test_fp = test.drop(columns=[provenance_column])

    parts = [train.reset_index(drop=True), syn.reset_index(drop=True)]
    if valid is not None and len(valid):
        parts.append(valid.reset_index(drop=True))
    parts.append(test.reset_index(drop=True))
    combined = pd.concat(parts, ignore_index=True)

    n_train = len(train) + len(syn)
    n_valid = 0 if valid is None else len(valid)
    train_idx = tuple(range(0, n_train))
    valid_idx = tuple(range(n_train, n_train + n_valid)) if n_valid else ()
    test_idx = tuple(range(n_train + n_valid, len(combined)))

    new_plan = SplitPlan(
        kind=f"synthetic_extend_{split_plan.kind}",
        test_size=split_plan.test_size,
        validation_size=split_plan.validation_size,
        random_state=split_plan.random_state,
        stratify_column=split_plan.stratify_column,
        train_indices=train_idx,
        validation_indices=valid_idx,
        test_indices=test_idx,
    )
    new_plan.assert_disjoint()

    # Holdout guard: original columns unchanged
    if valid_fp is not None and n_valid:
        rebuilt = combined.iloc[list(valid_idx)].drop(columns=[provenance_column])
        if not rebuilt.reset_index(drop=True).equals(valid_fp.reset_index(drop=True)):
            raise ValidationError(
                "Internal leakage guard failed: validation changed during synthetic merge."
            )
    rebuilt_test = combined.iloc[list(test_idx)].drop(columns=[provenance_column])
    if not rebuilt_test.reset_index(drop=True).equals(test_fp.reset_index(drop=True)):
        raise ValidationError(
            "Internal leakage guard failed: test changed during synthetic merge."
        )

    roles = dict(dataset.roles)
    roles[provenance_column] = ColumnRole.IGNORE
    new_dataset = Dataset.from_transformed(
        dataset,
        combined,
        schema=schema_from_dataframe(combined),
        roles=roles,
    )
    return new_dataset, new_plan, provenance_column


def sample_and_maybe_merge(
    dataset: Dataset,
    split_plan: SplitPlan,
    plan: SynthesizerPlan,
    *,
    n: int | None = None,
    random_state: int | None = None,
    condition: dict[str, Any] | None = None,
    merge_mode: MergeMode = "none",
    provenance_column: str = "_synthetic",
) -> tuple[SyntheticSampleResult, Dataset | None, SplitPlan | None]:
    """Sample, optionally merge into train; return updated dataset/split when merged.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
plan:
    Fitted plan object carrying model state and feature contract.
n:
    n (int | None).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
condition:
    condition (dict[str, Any] | None).
merge_mode:
    merge mode (MergeMode).
provenance_column:
    provenance column (str).

Returns
-------
tuple[SyntheticSampleResult, Dataset | None, SplitPlan | None]
    Tuple of results (tuple[SyntheticSampleResult, Dataset | None, SplitPlan | None]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    result = sample_synthetic(
        plan, n=n, random_state=random_state, condition=condition
    )
    if merge_mode == "none":
        return result, None, None
    if merge_mode != "extend_train":
        raise ValidationError(
            f"Unknown merge_mode={merge_mode!r}. Expected 'none' | 'extend_train'."
        )
    assert result.frame is not None
    new_ds, new_split, prov = merge_synthetic_into_train(
        dataset,
        split_plan,
        result.frame,
        provenance_column=provenance_column,
    )
    result.merged = True
    result.merge_mode = "extend_train"
    result.provenance_column = prov
    result.disclosures = tuple(
        list(result.disclosures)
        + [
            f"Merged {result.n_rows} synthetic rows into train "
            f"(provenance column {prov!r} role=ignore).",
            "Validation/test row values unchanged; split indices rebuilt.",
            "Synthetic train rows can bias estimators — disclose in model cards.",
        ]
    )
    return result, new_ds, new_split
