"""Feature / label helpers for semi-supervised Session ops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.semisupervised.types import SKLEARN_UNLABELED


def resolve_semisupervised_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns (exclude target and protected roles)."""
    disclosures: list[str] = []
    protected = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    exclude = {target_column}

    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        names = [
            name
            for name in names
            if dataset.roles.get(name) not in protected and name not in exclude
        ]
        if not names:
            raise ValidationError(
                "No usable columns after excluding protected roles and the target."
            )
        _assert_numeric(frame, names)
        return names, False, disclosures

    if prefer_reduce_components and reduce_plan is not None:
        feature_names = getattr(reduce_plan, "feature_names_", None) or ()
        present = [
            str(c)
            for c in feature_names
            if str(c) in frame.columns and str(c) not in exclude
        ]
        if present:
            _assert_numeric(frame, present)
            disclosures.append(
                "Used Session.reduce_dimensions component columns for "
                f"semi-supervised fit ({len(present)} component(s))."
            )
            return present, True, disclosures

    feature_roles = dataset.role_columns(ColumnRole.FEATURE)
    candidates = feature_roles or [
        str(c) for c in frame.columns if dataset.roles.get(str(c)) not in protected
    ]
    names = [
        str(c)
        for c in candidates
        if c in frame.columns
        and c not in exclude
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not names:
        raise ValidationError(
            "No numeric columns available for semi-supervised learning. "
            "Encode/scale first, or call reduce_dimensions."
        )
    return names, False, disclosures


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix; refuse null features."""
    block = frame[list(columns)]
    if block.isna().any().any():
        raise ValidationError(
            "Semi-supervised learning requires non-null features. "
            "Call session.impute(...) first (and typically session.scale(...))."
        )
    return block.to_numpy(dtype=float)


def is_unlabeled_mask(series: pd.Series, unlabeled_marker: Any = None) -> np.ndarray:
    """Return boolean mask where True means unlabeled.

    Convention
    ----------
    * ``unlabeled_marker is None`` (default): pandas missing (NaN / NA / None)
      marks unlabeled rows.
    * otherwise: rows equal to ``unlabeled_marker`` (plus missing) are unlabeled.
    """
    missing = series.isna().to_numpy(dtype=bool)
    if unlabeled_marker is None:
        return missing
    return missing | (series.to_numpy(dtype=object) == unlabeled_marker)


def encode_targets_for_sklearn(
    y: pd.Series,
    *,
    unlabeled_marker: Any = None,
    label_encoder: Any | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...], int, int]:
    """Map BuildML missingness → sklearn ``-1`` unlabeled convention.

    Returns
    -------
    y_sk, label_encoder, classes, n_labeled, n_unlabeled
    """
    from sklearn.preprocessing import LabelEncoder

    mask = is_unlabeled_mask(y, unlabeled_marker)
    n_unlabeled = int(mask.sum())
    n_labeled = int((~mask).sum())
    if n_labeled < 2:
        raise ValidationError(
            "Semi-supervised fit needs at least 2 labeled train rows "
            f"(found n_labeled={n_labeled}, n_unlabeled={n_unlabeled}). "
            "Leave scarce labels as NaN in the target role (or set unlabeled_marker)."
        )

    labeled_values = y.loc[~mask]
    if label_encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labeled_values.astype(str))
    else:
        encoder = label_encoder

    y_sk = np.full(shape=len(y), fill_value=SKLEARN_UNLABELED, dtype=int)
    encoded = encoder.transform(labeled_values.astype(str))
    y_sk[~mask] = encoded
    classes = tuple(encoder.classes_)
    if len(classes) < 2:
        raise ValidationError(
            "Semi-supervised classification requires at least 2 classes among "
            f"labeled train rows (found {classes!r})."
        )
    return y_sk, encoder, classes, n_labeled, n_unlabeled


def decode_predictions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Map sklearn integer codes back to original label strings/values."""
    codes = np.asarray(pred_codes)
    # Graph methods may emit floats; round to nearest class code.
    if np.issubdtype(codes.dtype, np.floating):
        codes = np.rint(codes).astype(int)
    else:
        codes = codes.astype(int)
    decoded = label_encoder.inverse_transform(codes)
    # Prefer original dtype-ish: if all look like ints, cast back
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


def _assert_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValidationError(
            "Semi-supervised learning requires numeric columns; encode/scale first. "
            f"Non-numeric: {non_numeric[:12]}"
        )
