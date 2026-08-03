"""Feature / train helpers for symbolic / neuro-symbolic ML (train-only fit)."""

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
    "resolve_symbolic_columns",
    "encode_classification_targets",
    "decode_predictions",
    "regression_targets",
    "train_partition_frame",
    "classification_accuracy",
    "regression_metrics",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix for rule evaluation and sklearn bases.

    Refuses null feature values because rule predicates and estimators cannot
    score incomplete rows silently.

    Parameters
    ----------
    frame:
        Partition frame containing the requested columns.
    columns:
        Numeric feature columns referenced by induced or declared rules.

    Returns
    -------
    numpy.ndarray
        Design matrix shaped ``(n_rows, n_features)``.

    Raises
    ------
    ValidationError
        When any selected column contains nulls or invalid values.
    """
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Symbolic learning")
        raise ValidationError(msg) from exc


def resolve_symbolic_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for rule induction and neuro-symbolic bases.

    Reuses semi-supervised column resolution (including optional PCA components)
    with symbolic-specific disclosure wording.

    Parameters
    ----------
    dataset:
        Session dataset carrying roles and target metadata.
    frame:
        Train partition frame used for fit.
    columns:
        Explicit feature list. When ``None``, resolved from roles / reduce plan.
    reduce_plan:
        Optional dimensionality-reduction plan whose components may be preferred.
    prefer_reduce_components:
        When True and a reduce plan exists, use its component columns.
    target_column:
        Target name excluded from feature columns.

    Returns
    -------
    tuple[list[str], bool, list[str]]
        Column names, whether reduce components were used, and disclosure strings.
    """
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    out = [note.replace("semi-supervised", "symbolic") for note in disclosures]
    return cols, used_reduce, out


def encode_classification_targets(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode classification targets with a sklearn LabelEncoder.

    Fit happens on train labels only; the encoder is stored on the plan for
    holdout decode.

    Parameters
    ----------
    y:
        Train target column. Must not contain nulls.
    classes:
        Optional fixed class list for consistent encoding across bundles.

    Returns
    -------
    tuple[numpy.ndarray, LabelEncoder, tuple]
        Integer codes, fitted encoder, and class tuple.

    Raises
    ------
    ValidationError
        When ``y`` contains null labels.
    """
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "Symbolic classification requires non-null train targets."
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
    """Map integer class codes back toward original label dtypes.

    Reverses the train-fitted encoder so Session predict results use familiar
    label values rather than internal codes.

    Parameters
    ----------
    pred_codes:
        Raw classifier output codes.
    label_encoder:
        Train-fitted encoder stored on the symbolic plan.

    Returns
    -------
    list
        Decoded predictions (int, float, or str as appropriate).
    """
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
    """Extract numeric regression targets from a train column.

    Refuses nulls and non-numeric dtypes before fitting regression rules or bases.

    Parameters
    ----------
    y:
        Train target column.

    Returns
    -------
    numpy.ndarray
        Float target vector.

    Raises
    ------
    ValidationError
        When ``y`` contains nulls or is not numeric.
    """
    if y.isna().any():
        raise ValidationError(
            "Symbolic regression requires non-null numeric targets."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "Symbolic regression requires a numeric target column."
        )
    return y.to_numpy(dtype=float)


def train_partition_frame(
    dataset: Dataset, split_plan: SplitPlan
) -> pd.DataFrame:
    """Return the train partition frame for symbolic fit.

    Thin wrapper over :func:`buildml.data.splits.frame_for_partition`.

    Parameters
    ----------
    dataset:
        Session dataset.
    split_plan:
        Split plan with train indices.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by ``split_plan.train_indices``.
    """
    return frame_for_partition(dataset, split_plan, "train")


def classification_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute holdout accuracy with string-normalized label comparison.

    Shared by evaluate and fit reporting so classification scores stay
    consistent when labels mix strings and numeric encodings.

    Parameters
    ----------
    y_true, y_pred:
        Parallel label sequences.

    Returns
    -------
    float
        Fraction of matching labels, or ``nan`` when empty.
    """
    if len(y_true) == 0:
        return float("nan")
    match = sum(str(a) == str(b) for a, b in zip(y_true, y_pred, strict=True))
    return float(match) / float(len(y_true))


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a regression holdout.

    Used by evaluate and fit paths for symbolic and neuro-symbolic regression
    tasks on validation or test partitions.

    Parameters
    ----------
    y_true, y_pred:
        Parallel numeric arrays.

    Returns
    -------
    dict[str, float]
        Keys ``rmse``, ``mae``, and ``r2``.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
