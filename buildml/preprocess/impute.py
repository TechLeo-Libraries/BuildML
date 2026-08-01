"""Simple imputation with explicit train-fit semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from buildml.core.errors import ValidationError
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe

Strategy = Literal["mean", "median", "most_frequent", "constant"]


@dataclass(slots=True)
class SimpleImputePlan:
    """Fitted imputation plan learned from the train partition only."""

    columns: tuple[str, ...]
    strategy: Strategy
    fill_value: Any | None
    statistics_: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "strategy": self.strategy,
            "fill_value": self.fill_value,
            "statistics_": dict(self.statistics_),
        }


def fit_simple_imputer(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    strategy: Strategy = "median",
    fill_value: Any | None = None,
) -> SimpleImputePlan:
    """Fit a simple imputer on the **train** partition only.

    Parameters
    ----------
    dataset:
        Full dataset (split membership selects train rows).
    split_plan:
        Required split plan. Fitting without a split is rejected.
    columns:
        Columns to impute. Defaults to numeric non-target columns.
    strategy:
        Sklearn ``SimpleImputer`` strategy.
    fill_value:
        Used when ``strategy='constant'``.

    Raises
    ------
    LeakageError
        If no split exists.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_columns(dataset, train, columns)
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    imputer.fit(train[list(cols)])
    stats = {
        col: _jsonable_stat(value) for col, value in zip(cols, imputer.statistics_, strict=True)
    }
    return SimpleImputePlan(
        columns=tuple(cols),
        strategy=strategy,
        fill_value=fill_value,
        statistics_=stats,
    )


def transform_simple_imputer(dataset: Dataset, plan: SimpleImputePlan) -> Dataset:
    """Apply a fitted impute plan to the full dataset frame.

    Parameters
    ----------
    dataset:
        Dataset to transform.
    plan:
        Plan from :func:`fit_simple_imputer`.

    Notes
    -----
    **Leakage:** Statistics must come from train-only fit. Do not construct a
    plan by fitting on the full frame.
    """
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Impute plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    for column in plan.columns:
        fill = plan.statistics_[column]
        frame[column] = frame[column].fillna(fill)

    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
    )


def _resolve_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is not None:
        return validate_column_names(columns, dataset.columns)

    target_cols = set(dataset.role_columns("target"))
    numeric = [
        c for c in train.columns if c not in target_cols and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not numeric:
        raise ValidationError("No numeric columns available for simple imputation")
    return [str(c) for c in numeric]


def _jsonable_stat(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
