"""Feature / train helpers for TDA (train-only fit contracts)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
)
from buildml.semisupervised.features import (
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
    """Build a float design matrix for point-cloud construction.

    Refuses null feature values because ripser/giotto cannot handle NaNs in the
    local clouds.

    Parameters
    ----------
    frame:
        Partition frame containing the requested columns.
    columns:
        Numeric feature column names (at least two required upstream).

    Returns
    -------
    numpy.ndarray
        Design matrix shaped ``(n_rows, n_features)``.

    Raises
    ------
    ValidationError
        When any selected column contains nulls or non-numeric values.
    """
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
    """Resolve numeric feature columns for local point-cloud construction.

    Reuses semi-supervised column resolution (including optional PCA components)
    but enforces at least two numeric columns because TDA needs a genuine cloud
    per row.

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
        Target name excluded from features. Defaults to dataset target.

    Returns
    -------
    tuple[list[str], bool, list[str]]
        Column names, whether reduce components were used, and disclosure strings.

    Raises
    ------
    ValidationError
        When fewer than two numeric feature columns remain.
    """
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
    """Return the train partition frame for TDA fit.

    Thin wrapper over :func:`buildml.data.splits.frame_for_partition` used by
    fit paths before subsampling.

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


def partition_frame(
    dataset: Dataset, split_plan: SplitPlan | None, partition: str
) -> pd.DataFrame:
    """Return a frame for transform, predict, or evaluate on one partition.

    ``partition='all'`` copies the full dataset frame; named partitions require
    a split plan.

    Parameters
    ----------
    dataset:
        Session dataset.
    split_plan:
        Split plan. Required unless ``partition='all'``.
    partition:
        ``train``, ``validation``, ``test``, or ``all``.

    Returns
    -------
    pandas.DataFrame
        Rows for the requested partition.

    Raises
    ------
    ValidationError
        When a named partition is requested but ``split_plan`` is ``None``.
    """
    if partition == "all":
        return dataset.frame.copy()
    if split_plan is None:
        raise ValidationError("A SplitPlan is required for partitioned TDA transforms.")
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-column mean and scale on the train design matrix.

    Zero-variance columns are left unscaled (scale set to 1) so PH still runs.

    Parameters
    ----------
    x:
        Train design matrix shaped ``(n_train, n_features)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Standardized matrix, mean vector, and scale vector for
        :func:`standardize_apply` on holdout rows.
    """
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply train-fitted standardization to a holdout design matrix.

    Uses mean and scale vectors from :func:`standardize_fit`: never recomputes
    statistics on holdout rows.

    Parameters
    ----------
    x:
        Holdout design matrix.
    mean, scale:
        Vectors from :func:`standardize_fit` on train: never refit on holdout.

    Returns
    -------
    numpy.ndarray
        Standardized holdout matrix using frozen train statistics.
    """
    return (x - mean) / scale


def infer_tda_task(y: pd.Series) -> str:
    """Infer classification versus regression from target dtype and cardinality.

    Numeric targets with many unique values are treated as regression; otherwise
    classification (including low-cardinality integer codes).

    Parameters
    ----------
    y:
        Train target column.

    Returns
    -------
    str
        ``classification`` or ``regression``.
    """
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(10, int(0.2 * len(y))):
        return "regression"
    return "classification"


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
        Integer codes, fitted encoder, and class tuple for the plan.

    Raises
    ------
    ValidationError
        When ``y`` contains null labels.
    """
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
    """Map integer class codes back toward original label dtypes.

    Reverses the train-fitted encoder so Session predict results use familiar
    label values rather than internal codes.

    Parameters
    ----------
    pred_codes:
        Raw classifier output codes.
    label_encoder:
        Train-fitted encoder stored on :class:`~buildml.tda.results.TdaPlan`.

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

    Refuses nulls and non-numeric dtypes before fitting a regression head on
    topological features.

    Parameters
    ----------
    y:
        Train target column. Must be numeric and non-null.

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
        raise ValidationError("TDA regression head requires non-null numeric targets.")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError("TDA regression requires a numeric target column.")
    return y.to_numpy(dtype=float)


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """Compute accuracy and macro F1 for a classification holdout.

    Labels are string-normalized before sklearn metrics so mixed dtypes still
    score consistently.

    Parameters
    ----------
    y_true, y_pred:
        Parallel label sequences (string-normalized internally).

    Returns
    -------
    dict[str, float]
        Keys ``accuracy`` and ``macro_f1``.
    """
    from sklearn.metrics import accuracy_score, f1_score

    yt = [str(v) for v in y_true]
    yp = [str(v) for v in y_pred]
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a regression holdout.

    Standard sklearn regression metrics on the frozen head's holdout predictions.

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
