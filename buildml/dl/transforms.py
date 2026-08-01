"""Train-fit normalize helpers for tabular Torch loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError


def fit_standardize(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-column mean/std on train features only."""
    if x_train.ndim != 2:
        raise ValidationError("Standardize expects a 2-D feature matrix")
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_standardize(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply frozen train mean/std to a feature matrix."""
    if x.shape[1] != mean.shape[0]:
        raise ValidationError(
            f"Feature width {x.shape[1]} does not match normalize width {mean.shape[0]}"
        )
    return (x - mean) / std


def frame_to_numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Cast selected columns to float64, rejecting non-numeric dtypes."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing columns for Torch tensors: {missing}")
    subset = frame.loc[:, columns]
    for name in columns:
        if not pd.api.types.is_numeric_dtype(subset[name]):
            raise ValidationError(
                f"Column '{name}' is not numeric; encode or drop before make_torch_loaders"
            )
    values = subset.to_numpy(dtype=np.float64, copy=True)
    if np.isnan(values).any():
        raise ValidationError(
            "Feature matrix contains NaN; impute on train before make_torch_loaders"
        )
    return values
