"""Leakage gates and column selection for synthetic-data systems."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from buildml.core.errors import LeakageError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition


def require_split(split_plan: SplitPlan | None) -> SplitPlan:
    """Require an existing split before synthesizer fitting.

Returns the supplied split unchanged; missing splits raise ValidationError.

Parameters
----------
split_plan:
    Train/validation/test split; fit uses train partition only.

Returns
-------
SplitPlan
    The supplied split plan.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if split_plan is None:
        raise ValidationError(
            "A split is required before fitting a synthesizer. "
            "Call Session.split(...) first so generators fit on train only."
        )
    return split_plan


def assert_train_only_fit(partition: str) -> None:
    """Reject synthesizer fitting outside the training partition.

Validation and test rows must remain separate from generator fitting to avoid leaking holdout structure.

Parameters
----------
partition:
    Holdout partition name or ``all`` for the full frame.

Raises
------
LeakageError
    If the requested partition is not train.
    """
    if partition != "train":
        raise LeakageError(
            "Synthesizer fitting is restricted to partition='train'. "
            f"Got partition={partition!r}. Fitting a generator on validation "
            "or test leaks holdout structure into synthetic samples. "
            "Use evaluate_synthetic on holdout for utility/fidelity checks."
        )


def require_train_frame(
    dataset: Dataset,
    split_plan: SplitPlan,
) -> pd.DataFrame:
    """Return a copy of the training rows for generator fitting.

Checks that the split permits fitting on train before selecting its row indices.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.

Returns
-------
pd.DataFrame
    Independent copy of the selected rows.
    """
    assert_fit_partition(split_plan, "train")
    return frame_for_partition(dataset, split_plan, "train").copy()


def resolve_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    *,
    columns: Sequence[str] | None,
    target_column: str | None = None,
    method: str = "gaussian_copula",
) -> list[str]:
    """Select columns for a tabular synthesizer.

Explicit columns are checked against the training frame. Otherwise use feature roles and the target, excluding ID, ignore, and weight roles; SMOTE requires a target.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
train:
    train (pd.DataFrame).
columns:
    Explicit modeled columns; ``None`` selects columns from dataset roles.
target_column:
    Name of the supervised target column.
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
list[str]
    Ordered column names to model.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if columns is not None:
        cols = [str(c) for c in columns]
        missing = [c for c in cols if c not in train.columns]
        if missing:
            raise ValidationError(f"Unknown synthesizer columns: {missing[:12]}")
        return cols

    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    target_name = None
    for name, role in dataset.roles.items():
        if role == ColumnRole.TARGET:
            target_name = name
            break

    ignore = {
        name
        for name, role in dataset.roles.items()
        if role in {ColumnRole.ID, ColumnRole.IGNORE, ColumnRole.WEIGHT}
    }
    if method == "smote":
        tgt = target_column or target_name
        if tgt is None:
            raise ValidationError(
                "method='smote' requires a target role or target_column."
            )
        feats = feature_cols or [
            c for c in train.columns if c != tgt and c not in ignore
        ]
        cols = list(feats) + ([tgt] if tgt not in feats else [])
        return cols

    # bootstrap / gaussian_copula / sdv: features + target (if present), skip id/ignore
    cols = []
    if feature_cols:
        cols.extend(feature_cols)
        if target_name is not None and target_name not in cols:
            cols.append(target_name)
    else:
        cols = [c for c in train.columns if c not in ignore]
    if not cols:
        raise ValidationError("No columns available for synthesizer fit.")
    return cols


def partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: str,
) -> pd.DataFrame:
    """Return a copy of the requested dataset partition.

The special partition name ``all`` selects the full frame; other names are resolved through the supplied split.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
partition:
    Holdout partition name or ``all`` for the full frame.

Returns
-------
pd.DataFrame
    Independent copy of the selected rows.
    """
    if partition == "all":
        return dataset.frame.copy()
    return frame_for_partition(dataset, split_plan, partition).copy()  # type: ignore[arg-type]
