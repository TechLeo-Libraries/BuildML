"""Feature / client-column helpers for federated learning simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
)
from buildml.semisupervised.features import (
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_client_column",
    "resolve_target_column",
    "resolve_federated_columns",
    "encode_labels",
    "decode_predictions",
    "client_ids_in_frame",
    "frame_for_client",
    "extract_linear_params",
    "set_linear_params",
    "average_linear_params",
    "clone_estimator_with_params",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix from selected columns.

    Delegates to semi-supervised matrix building with federated error wording
    when null features are detected.

    Parameters
    ----------
    frame:
        Source DataFrame.
    columns:
        Feature column names to extract.

    Returns
    -------
    numpy.ndarray
        2-D float feature matrix.

    Raises
    ------
    ValidationError
        When columns are missing or contain null values.
    """
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Federated learning")
        raise ValidationError(msg) from exc


def resolve_client_column(
    dataset: Dataset,
    client_column: str | None,
) -> tuple[str, list[str]]:
    """Resolve the client / group column from roles or an explicit name.

    Federated learning requires a stable client identifier separate from
    feature columns and the prediction target.

    Parameters
    ----------
    dataset:
        BuildML dataset with column roles.
    client_column:
        Optional explicit client column; when ``None``, requires exactly one
        ``role='group'`` column.

    Returns
    -------
    tuple[str, list[str]]
        Resolved client column name and disclosure notes.

    Raises
    ------
    ValidationError
        When no client column is defined or multiple group columns exist.
    """
    disclosures: list[str] = []
    if client_column is not None:
        name = validate_column_names([client_column], dataset.columns)[0]
        disclosures.append(
            f"Federated client column taken from explicit client_column={name!r}."
        )
        return name, disclosures

    groups = list(dataset.role_columns(ColumnRole.GROUP))
    if len(groups) == 1:
        disclosures.append(
            f"Federated client column taken from role='group' column: {groups[0]!r}."
        )
        return groups[0], disclosures
    if len(groups) > 1:
        raise ValidationError(
            "Multiple role='group' columns found "
            f"({groups}). Pass client_column= explicitly to select the "
            "federated client identifier."
        )
    raise ValidationError(
        "Federated learning needs a client/group column. Assign role='group' "
        "to the client identifier column, or pass client_column=."
    )


def resolve_target_column(dataset: Dataset) -> tuple[str, list[str]]:
    """Resolve exactly one target column for federated fit.

    Multi-target joint fitting belongs on ``fit_multitask``; federated
    simulation partitions by a client/group column with a single target.

    Parameters
    ----------
    dataset:
        BuildML dataset with column roles.

    Returns
    -------
    tuple[str, list[str]]
        Resolved target column name and disclosure notes.

    Raises
    ------
    ValidationError
        When zero or multiple ``role='target'`` columns are present.
    """
    targets = list(dataset.role_columns(ColumnRole.TARGET))
    if len(targets) != 1:
        raise ValidationError(
            "Federated learning requires exactly one role='target' column "
            f"(found {targets!r}). Multi-target joint fitting belongs on "
            "fit_multitask; federated simulation partitions by a client/group "
            "column."
        )
    return targets[0], [
        f"Federated target taken from role='target' column: {targets[0]!r}."
    ]


def resolve_federated_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
    client_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for federated local updates.

    Excludes target and client columns from the feature set and reuses
    semi-supervised column resolution for reduce-component preferences.

    Parameters
    ----------
    dataset:
        BuildML dataset with column roles and optional reduce plan.
    frame:
        Train partition frame used for column discovery.
    columns:
        Optional explicit feature column list.
    reduce_plan:
        Optional preprocess reduce plan from Session.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    target_column:
        Resolved target column to exclude from features.
    client_column:
        Resolved client column to exclude from features.

    Returns
    -------
    tuple[list[str], bool, list[str]]
        Feature column names, whether reduce components were used, and
        disclosure notes.

    Raises
    ------
    ValidationError
        When no usable feature columns remain after exclusions.
    """
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    # Rephrase disclosures for federated context.
    disclosures = [
        d.replace("Semi-supervised learning", "Federated learning").replace(
            "semi-supervised", "federated"
        )
        for d in disclosures
    ]
    if client_column in cols:
        cols = [c for c in cols if c != client_column]
        disclosures.append(
            f"Excluded client column {client_column!r} from federated features "
            "(client id must not be a model input)."
        )
    if not cols:
        raise ValidationError(
            "Federated learning needs at least one numeric feature column "
            f"after excluding target={target_column!r} and "
            f"client={client_column!r}."
        )
    return cols, used_reduce, disclosures


def encode_labels(
    series: pd.Series,
    *,
    label_encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, LabelEncoder, tuple[Any, ...]]:
    """Encode classification targets for federated local updates.

    Reuses a fitted encoder when provided so evaluation and prediction share
    the train-time class vocabulary.

    Parameters
    ----------
    series:
        Target column values to encode.
    label_encoder:
        Optional pre-fitted :class:`~sklearn.preprocessing.LabelEncoder`.

    Returns
    -------
    tuple[numpy.ndarray, LabelEncoder, tuple[Any, ...]]
        Integer-encoded targets, encoder instance, and class tuple.

    Raises
    ------
    ValidationError
        When targets contain nulls or unseen labels at transform time.
    """
    if series.isna().any():
        raise ValidationError(
            "Federated learning classification targets contain nulls; "
            "impute or drop before fit_federated."
        )
    values = series.to_numpy()
    if label_encoder is None:
        enc = LabelEncoder()
        encoded = enc.fit_transform(values)
    else:
        enc = label_encoder
        known = set(enc.classes_)
        unseen = sorted({v for v in values if v not in known}, key=str)
        if unseen:
            raise ValidationError(
                "Federated learning encountered unseen class labels "
                f"{unseen} not present during fit class discovery "
                f"(known={list(enc.classes_)})."
            )
        encoded = enc.transform(values)
    classes = tuple(enc.classes_.tolist())
    return np.asarray(encoded, dtype=int), enc, classes


def decode_predictions(
    encoded: np.ndarray,
    label_encoder: LabelEncoder | None,
) -> tuple[Any, ...]:
    """Map encoded class indices back to original labels.

    Inverse-transforms sklearn integer predictions so Session-facing outputs use
    the same label vocabulary discovered during federated fit.

    Parameters
    ----------
    encoded:
        Integer class codes from a sklearn classifier.
    label_encoder:
        Optional fitted label encoder from fit time.

    Returns
    -------
    tuple
        Original label values in prediction order.
    """
    if label_encoder is None:
        return tuple(encoded.tolist())
    return tuple(label_encoder.inverse_transform(np.asarray(encoded)).tolist())


def client_ids_in_frame(frame: pd.DataFrame, client_column: str) -> list[Any]:
    """Return stable unique client ids present in a frame.

    Preserves first-seen order from ``pandas.unique`` while dropping null
    identifiers unsuitable for federated partitioning.

    Parameters
    ----------
    frame:
        Partition or client slice DataFrame.
    client_column:
        Column holding federated client identifiers.

    Returns
    -------
    list
        Unique non-null client id values in first-seen order.

    Raises
    ------
    ValidationError
        When ``client_column`` is missing from ``frame``.
    """
    if client_column not in frame.columns:
        raise ValidationError(
            f"Client column {client_column!r} missing from frame."
        )
    ids = pd.unique(frame[client_column])
    return [x for x in ids.tolist() if pd.notna(x)]


def frame_for_client(
    frame: pd.DataFrame,
    client_column: str,
    client_id: Any,
) -> pd.DataFrame:
    """Return rows belonging to one client id.

    Returns a defensive copy so local client updates cannot mutate shared
    partition frames used across rounds.

    Parameters
    ----------
    frame:
        Source partition DataFrame.
    client_column:
        Column holding federated client identifiers.
    client_id:
        Client identifier value to select.

    Returns
    -------
    pandas.DataFrame
        Copy of rows where ``client_column == client_id``.
    """
    out = frame.loc[frame[client_column] == client_id].copy()
    if not isinstance(out, pd.DataFrame):
        raise ValidationError(
            "frame_for_client expected a DataFrame slice after client filtering"
        )
    return out


def extract_linear_params(estimator: Any) -> dict[str, np.ndarray]:
    """Extract ``coef_`` and ``intercept_`` for FedAvg aggregation.

    Copies coefficient arrays so client updates can be averaged without
    mutating the global template estimator in place.

    Parameters
    ----------
    estimator:
        Sklearn linear or SGD estimator with fitted coefficients.

    Returns
    -------
    dict[str, numpy.ndarray]
        Copied ``coef_`` and ``intercept_`` arrays.

    Raises
    ------
    ValidationError
        When the estimator lacks ``coef_`` or ``intercept_`` attributes.
    """
    if not hasattr(estimator, "coef_") or not hasattr(estimator, "intercept_"):
        raise ValidationError(
            "Federated aggregation requires estimators with coef_ and "
            f"intercept_ (got {type(estimator).__name__})."
        )
    return {
        "coef_": np.asarray(estimator.coef_, dtype=float).copy(),
        "intercept_": np.asarray(estimator.intercept_, dtype=float).copy(),
    }


def set_linear_params(estimator: Any, params: dict[str, np.ndarray]) -> None:
    """Write aggregated linear parameters onto an estimator.

    Installs FedAvg-aggregated weights on the global model or a cloned local
    estimator before the next round or prediction call.

    Parameters
    ----------
    estimator:
        Sklearn linear or SGD estimator to update in place.
    params:
        Mapping with ``coef_`` and ``intercept_`` arrays from aggregation.
    """
    estimator.coef_ = np.asarray(params["coef_"], dtype=float).copy()
    estimator.intercept_ = np.asarray(params["intercept_"], dtype=float).copy()


def average_linear_params(
    param_list: list[dict[str, np.ndarray]],
    weights: list[float],
) -> dict[str, np.ndarray]:
    """Compute a weighted average of linear parameters (FedAvg).

    Weights are typically client sample counts so larger clients influence the
    global model proportionally.

    Parameters
    ----------
    param_list:
        Per-client parameter dicts from :func:`extract_linear_params`.
    weights:
        Non-negative weights aligned with ``param_list`` (e.g. row counts).

    Returns
    -------
    dict[str, numpy.ndarray]
        Weighted average ``coef_`` and ``intercept_`` arrays.

    Raises
    ------
    ValidationError
        When lists are empty, lengths mismatch, or weights sum to zero.
    """
    if not param_list:
        raise ValidationError("No client parameters to aggregate.")
    if len(param_list) != len(weights):
        raise ValidationError("param_list and weights length mismatch.")
    total = float(sum(weights))
    if total <= 0:
        raise ValidationError("Aggregation weights must sum to a positive value.")
    coef = np.zeros_like(param_list[0]["coef_"], dtype=float)
    intercept = np.zeros_like(param_list[0]["intercept_"], dtype=float)
    for params, w in zip(param_list, weights, strict=True):
        coef = coef + (float(w) / total) * params["coef_"]
        intercept = intercept + (float(w) / total) * params["intercept_"]
    return {"coef_": coef, "intercept_": intercept}


def clone_estimator_with_params(
    template: Any,
    params: dict[str, np.ndarray],
    *,
    classes: np.ndarray | None = None,
) -> Any:
    """Clone a template estimator and install linear parameters.

    Copies sklearn bookkeeping attributes when present so ``predict`` works
    after parameter assignment.

    Parameters
    ----------
    template:
        Source estimator defining attribute shapes and metadata.
    params:
        Aggregated ``coef_`` and ``intercept_`` arrays.
    classes:
        Optional class array for classifiers.

    Returns
    -------
    Any
        Cloned estimator with parameters and optional ``classes_`` set.
    """
    from sklearn.base import clone

    est = clone(template)
    # Ensure attribute slots exist before assignment for partial_fit models.
    if hasattr(template, "coef_"):
        set_linear_params(est, params)
    else:
        set_linear_params(est, params)
    if classes is not None and hasattr(est, "classes_"):
        est.classes_ = np.asarray(classes)
    # Copy sklearn bookkeeping when present so predict works after set.
    for attr in ("n_features_in_", "feature_names_in_", "n_iter_", "t_"):
        if hasattr(template, attr):
            setattr(est, attr, getattr(template, attr))
    return est
