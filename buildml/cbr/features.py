"""Feature / train helpers for case-based reasoning (train-only fit)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_cbr_columns",
    "resolve_categorical_columns",
    "encode_classification_targets",
    "decode_predictions",
    "regression_targets",
    "train_partition_frame",
    "classification_accuracy",
    "regression_metrics",
    "standardize_fit",
    "standardize_apply",
    "numeric_ranges",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Case-based reasoning")
        raise ValidationError(msg) from exc


def resolve_cbr_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns (same contract as semi-supervised)."""
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    out = [note.replace("semi-supervised", "case-based reasoning") for note in disclosures]
    return cols, used_reduce, out


def resolve_categorical_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    categorical_columns: list[str] | None,
    *,
    target_column: str,
    numeric_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve optional categorical columns for mixed-metric CBR.

    When ``categorical_columns`` is None and metric will be mixed, returns
    empty — callers must pass columns explicitly for categoricals (honesty:
    no silent auto-detection of high-cardinality strings).
    """
    disclosures: list[str] = []
    if not categorical_columns:
        return [], disclosures
    protected = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    names = validate_column_names(categorical_columns, dataset.columns)
    out: list[str] = []
    for name in names:
        if name == target_column or dataset.roles.get(name) in protected:
            continue
        if name in numeric_columns:
            raise ValidationError(
                f"Column {name!r} cannot be both numeric and categorical for CBR."
            )
        if name not in frame.columns:
            raise ValidationError(f"Categorical column {name!r} missing from frame.")
        if frame[name].isna().any():
            raise ValidationError(
                f"Categorical column {name!r} has nulls; impute or drop before CBR."
            )
        out.append(name)
    if not out:
        raise ValidationError(
            "No usable categorical columns after excluding protected roles / target."
        )
    disclosures.append(
        f"Using {len(out)} explicit categorical column(s) for mixed-metric CBR."
    )
    return out, disclosures


def encode_classification_targets(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode classification targets; refuse missing labels."""
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "CBR classification requires non-null train targets (case solutions)."
        )
    values = y.astype(str)
    encoder = LabelEncoder()
    if classes is not None:
        encoder.fit([str(c) for c in classes])
        codes = encoder.transform(values)
    else:
        codes = encoder.fit_transform(values)
    return np.asarray(codes), encoder, tuple(encoder.classes_)


def decode_predictions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Map integer class codes back toward original label values."""
    codes = np.asarray(pred_codes).astype(int)
    decoded = label_encoder.inverse_transform(codes)
    out: list[Any] = []
    for value in decoded:
        text = str(value)
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            out.append(int(text))
        else:
            try:
                out.append(float(text) if "." in text else text)
            except ValueError:
                out.append(text)
    return out


def regression_targets(y: pd.Series) -> np.ndarray:
    """Numeric regression targets; refuse nulls."""
    if y.isna().any():
        raise ValidationError(
            "CBR regression requires non-null numeric targets (case solutions)."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "CBR regression requires a numeric target column."
        )
    return y.to_numpy(dtype=float)


def train_partition_frame(
    dataset: Dataset, split_plan: SplitPlan
) -> pd.DataFrame:
    """Train frame helper."""
    return frame_for_partition(dataset, split_plan, "train")


def classification_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Simple accuracy (string-normalized)."""
    if len(y_true) == 0:
        return float("nan")
    match = sum(str(a) == str(b) for a, b in zip(y_true, y_pred, strict=True))
    return float(match) / float(len(y_true))


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """RMSE / MAE / R2 for regression holdout."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit mean/scale on train numeric matrix; return transformed + params."""
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def standardize_apply(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply train-fit standardization."""
    return (x - mean) / scale


def numeric_ranges(x: np.ndarray) -> np.ndarray:
    """Per-column ranges for Gower-style mixed distance (train-fit)."""
    lo = np.min(x, axis=0)
    hi = np.max(x, axis=0)
    ranges = hi - lo
    return np.where(ranges < 1e-12, 1.0, ranges)
