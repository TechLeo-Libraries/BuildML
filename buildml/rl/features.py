"""Feature / column helpers for imitation + RL (train-only fit contracts)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_rl_columns",
    "infer_imitation_task",
    "encode_discrete_actions",
    "decode_discrete_actions",
    "continuous_actions",
    "classification_metrics",
    "regression_metrics",
    "softmax",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features."""
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Imitation / RL")
        raise ValidationError(msg) from exc


def resolve_rl_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
    exclude_columns: Sequence[str] = (),
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature/context columns (same contract as semi-supervised)."""
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    exclude = {str(c) for c in exclude_columns}
    filtered = [c for c in cols if c not in exclude]
    if not filtered:
        raise ValidationError(
            "No usable feature/context columns remain after excluding "
            f"{sorted(exclude)}."
        )
    out = [
        note.replace("semi-supervised", "imitation / reinforcement learning")
        for note in disclosures
    ]
    if exclude:
        out.append(
            f"Excluded non-state columns from the design matrix: {sorted(exclude)}."
        )
    return filtered, used_reduce, out


def infer_imitation_task(action: pd.Series) -> str:
    """Infer classification vs regression from the action column dtype."""
    if pd.api.types.is_numeric_dtype(action) and not pd.api.types.is_bool_dtype(action):
        nunique = int(action.nunique(dropna=True))
        # Small integer cardinalities → discrete actions (classification BC).
        if pd.api.types.is_integer_dtype(action) and nunique <= 20:
            return "classification"
        if nunique <= 8 and set(np.unique(action.dropna().to_numpy())).issubset(
            {0, 1, 2, 3, 4, 5, 6, 7}
        ):
            return "classification"
        return "regression"
    return "classification"


def encode_discrete_actions(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode discrete actions; refuse missing labels."""
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "Imitation / bandit discrete actions require non-null train values."
        )
    values = y.astype(str)
    encoder = LabelEncoder()
    if classes is not None:
        encoder.fit([str(c) for c in classes])
        codes = encoder.transform(values)
    else:
        codes = encoder.fit_transform(values)
    return np.asarray(codes, dtype=int), encoder, tuple(encoder.classes_)


def decode_discrete_actions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Map integer action codes back toward original label values."""
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


def continuous_actions(y: pd.Series) -> np.ndarray:
    """Numeric continuous actions; refuse nulls."""
    if y.isna().any():
        raise ValidationError(
            "Imitation regression requires non-null numeric action values."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "Imitation regression requires a numeric action column."
        )
    return y.to_numpy(dtype=float)


def classification_metrics(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> dict[str, float]:
    """Accuracy + macro-F1 for discrete imitation."""
    from sklearn.metrics import accuracy_score, f1_score

    yt = [str(v) for v in y_true]
    yp = [str(v) for v in y_pred]
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / R2 for continuous actions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def softmax(logits: np.ndarray, *, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    t = max(float(temperature), 1e-8)
    z = np.asarray(logits, dtype=float) / t
    z = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=axis, keepdims=True)
