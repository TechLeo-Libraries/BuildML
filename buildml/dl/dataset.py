"""Partition frames → Torch tensor datasets."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.dl.extras import require_torch
from buildml.dl.labels import encode_class_targets, fit_class_labels
from buildml.dl.transforms import apply_standardize, fit_standardize, frame_to_numeric_matrix
from buildml.dl.types import FeatureContract, TaskSpec


def resolve_feature_target(
    dataset: Dataset,
) -> tuple[list[str], str]:
    """Return feature column names and the target column name."""
    target = dataset.require_target()
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    if not feature_cols:
        skip = {
            *dataset.role_columns(ColumnRole.TARGET),
            *dataset.role_columns(ColumnRole.ID),
            *dataset.role_columns(ColumnRole.IGNORE),
            *dataset.role_columns(ColumnRole.GROUP),
            *dataset.role_columns(ColumnRole.TIME),
            *dataset.role_columns(ColumnRole.WEIGHT),
        }
        feature_cols = [c for c in dataset.columns if c not in skip and c != target]
    if not feature_cols:
        raise ValidationError("No feature columns available for Torch loaders")
    return feature_cols, target


def infer_task(y: pd.Series, task: TaskSpec) -> Literal["classification", "regression"]:
    """Infer classification vs regression from labels when task is ``auto``."""
    if task != "auto":
        return task
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(10, int(0.2 * len(y))):
        return "regression"
    return "classification"


def partition_arrays(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: Literal["train", "validation", "test"],
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize numeric feature/target arrays for one partition."""
    indices = split_plan.indices_for(partition)
    if not indices:
        return (
            np.empty((0, len(feature_columns)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    # Index membership only — avoid frame_for_partition's empty-validation raise.
    frame = dataset._ensure_pandas().iloc[list(indices)].copy()
    if frame.empty:
        return (
            np.empty((0, len(feature_columns)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    x = frame_to_numeric_matrix(frame, feature_columns)
    y_series = frame[target_column]
    if pd.api.types.is_numeric_dtype(y_series):
        y = y_series.to_numpy(dtype=np.float64, copy=True)
    else:
        # Classification with non-numeric labels handled by caller via label map.
        raise ValidationError(
            f"Target '{target_column}' must be numeric for the tabular Torch slice "
            "(encode class labels to integers first)"
        )
    if np.isnan(y).any():
        raise ValidationError("Target contains NaN; clean labels before make_torch_loaders")
    return x, y


def build_feature_contract(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    task: TaskSpec = "auto",
    normalize: bool = True,
) -> tuple[FeatureContract, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Fit optional train normalize stats and return arrays per partition."""
    feature_cols, target = resolve_feature_target(dataset)
    x_train, y_train = partition_arrays(dataset, split_plan, "train", feature_cols, target)
    if len(x_train) == 0:
        raise ValidationError("Train partition is empty; cannot build Torch loaders")
    resolved = infer_task(pd.Series(y_train), task)

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    if normalize:
        mean, std = fit_standardize(x_train)

    class_labels: tuple[Any, ...] = ()
    if resolved == "classification":
        class_labels = fit_class_labels(y_train)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("train", "validation", "test"):
        if name == "train":
            x, y = x_train, y_train
        else:
            x, y = partition_arrays(dataset, split_plan, name, feature_cols, target)  # type: ignore[arg-type]
        if normalize and mean is not None and std is not None and len(x):
            x = apply_standardize(x, mean, std)
        if resolved == "classification" and len(y):
            y = encode_class_targets(y, class_labels).astype(np.float64, copy=False)
        arrays[name] = (x, y)

    contract = FeatureContract(
        feature_columns=tuple(feature_cols),
        target_column=target,
        task=resolved,
        class_labels=class_labels,
        normalize_mean=None if mean is None else tuple(float(v) for v in mean),
        normalize_std=None if std is None else tuple(float(v) for v in std),
    )
    return contract, arrays


def arrays_to_tensor_dataset(x: np.ndarray, y: np.ndarray, *, task: str) -> Any:
    """Wrap NumPy arrays as a ``TensorDataset`` (lazy Torch import)."""
    torch = require_torch(feature="Tabular Torch datasets")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    if task == "classification":
        y_t = torch.as_tensor(y, dtype=torch.long)
    else:
        y_t = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    return torch.utils.data.TensorDataset(x_t, y_t)
