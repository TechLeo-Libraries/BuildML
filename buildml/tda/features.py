"""Feature / train helpers for TDA (train-only fit contracts)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_tda_columns",
    "train_partition_frame",
    "partition_frame",
    "standardize_fit",
    "standardize_apply",
    "infer_tda_task",
    "encode_classification_targets",
    "decode_predictions",
    "regression_targets",
    "classification_metrics",
    "regression_metrics",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Topological Data Analysis")
        raise ValidationError(msg) from exc


def resolve_tda_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str | None = None,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for point-cloud construction."""
    target = target_column if target_column is not None else dataset.require_target()
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    out = [note.replace("semi-supervised", "TDA") for note in disclosures]
    if len(cols) < 2:
        raise ValidationError(
            "TDA requires at least 2 numeric feature columns to form point clouds."
        )
    return cols, used_reduce, out


def train_partition_frame(dataset: Dataset, split_plan: SplitPlan) -> pd.DataFrame:
    """Train frame helper."""
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset, split_plan: SplitPlan | None, partition: str
) -> pd.DataFrame:
    """Frame for a named partition (or full frame when partition='all')."""
    if partition == "all":
        return dataset.frame.copy()
    if split_plan is None:
        raise ValidationError("A SplitPlan is required for partitioned TDA transforms.")
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit mean/scale on train numeric matrix."""
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply train-fit standardization."""
    return (x - mean) / scale


def infer_tda_task(y: pd.Series) -> str:
    """Infer classification vs regression from target dtype / cardinality."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(10, int(0.2 * len(y))):
        return "regression"
    return "classification"


def encode_classification_targets(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode classification targets; refuse missing labels."""
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError("TDA classification head requires non-null train targets.")
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
        raise ValidationError("TDA regression head requires non-null numeric targets.")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError("TDA regression requires a numeric target column.")
    return y.to_numpy(dtype=float)


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """Accuracy + macro F1 for classification holdout."""
    from sklearn.metrics import accuracy_score, f1_score

    yt = [str(v) for v in y_true]
    yp = [str(v) for v in y_pred]
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / R2 for regression holdout."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
