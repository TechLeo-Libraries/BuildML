"""Leakage gates and column selection for synthetic-data systems."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from buildml.core.errors import LeakageError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition


def require_split(split_plan: SplitPlan | None) -> SplitPlan:
    if split_plan is None:
        raise ValidationError(
            "A split is required before fitting a synthesizer. "
            "Call Session.split(...) first so generators fit on train only."
        )
    return split_plan


def assert_train_only_fit(partition: str) -> None:
    """Synthesizers always fit on Session train — never validation/test."""
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
    """Choose columns to model; never silently drop the target for SMOTE."""
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
    if partition == "all":
        return dataset.frame.copy()
    return frame_for_partition(dataset, split_plan, partition).copy()  # type: ignore[arg-type]
