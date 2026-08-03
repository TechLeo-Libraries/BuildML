"""Feature / pool helpers for active learning (reuses semi-supervised NaN convention)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.semisupervised.features import (
    is_unlabeled_mask,
    matrix_from_frame,
    resolve_semisupervised_columns,
)

# Re-export for callers that prefer the AL package surface.
__all__ = [
    "is_unlabeled_mask",
    "matrix_from_frame",
    "resolve_activelearning_columns",
    "encode_labeled_targets",
    "decode_predictions",
]


def resolve_activelearning_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve numeric feature columns for active-learning fit and query.

    Delegates to the semi-supervised column resolver and rephrases disclosures
    for the active-learning domain.

    Parameters
    ----------
    dataset:
        BuildML dataset carrying roles and schema metadata.
    frame:
        Partition frame whose columns are candidates for features.
    columns:
        Optional explicit feature column list; ``None`` auto-selects numerics.
    reduce_plan:
        Optional preprocess reduce plan from Session.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    target_column:
        Target column name excluded from feature selection.

    Returns
    -------
    tuple[list[str], bool, list[str]]
        Selected columns, whether reduce components were used, and disclosures.
    """
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    # Rephrase disclosures for the AL domain.
    out = []
    for note in disclosures:
        out.append(note.replace("semi-supervised", "active-learning"))
    return cols, used_reduce, out


def encode_labeled_targets(
    y: pd.Series,
    *,
    unlabeled_marker: Any = None,
    label_encoder: Any | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...], np.ndarray, int, int]:
    """Encode only labeled rows; return codes aligned to labeled positions.

    Unlabeled rows (NaN or ``unlabeled_marker``) are excluded from encoding.
    The returned ``labeled_mask`` aligns with the input series index.

    Parameters
    ----------
    y:
        Target series containing labeled and unlabeled values.
    unlabeled_marker:
        Optional sentinel marking unlabeled pool rows; ``None`` uses NaN/NA.
    label_encoder:
        Optional fitted :class:`~sklearn.preprocessing.LabelEncoder` to reuse
        or extend when new classes appear after user labeling rounds.

    Returns
    -------
    y_codes:
        Integer class codes for labeled rows only (length ``n_labeled``).
    label_encoder:
        Fitted label encoder covering all labeled classes seen so far.
    classes:
        Tuple of class labels in encoder order.
    labeled_mask:
        Boolean mask aligned to ``y``: ``True`` for labeled rows.
    n_labeled:
        Count of labeled rows.
    n_unlabeled:
        Count of unlabeled pool rows.

    Raises
    ------
    ValidationError
        When fewer than two labeled rows or fewer than two classes are present.
    """
    from sklearn.preprocessing import LabelEncoder

    mask_unlabeled = is_unlabeled_mask(y, unlabeled_marker)
    labeled_mask = ~mask_unlabeled
    n_unlabeled = int(mask_unlabeled.sum())
    n_labeled = int(labeled_mask.sum())
    if n_labeled < 2:
        raise ValidationError(
            "Active learning needs at least 2 labeled train rows to fit a "
            f"classifier (found n_labeled={n_labeled}, n_unlabeled={n_unlabeled}). "
            "Seed labels on the train partition (NaN marks the unlabeled pool)."
        )

    labeled_values = y.loc[labeled_mask]
    if label_encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labeled_values.astype(str))
    else:
        encoder = label_encoder
        # Allow new classes from user-provided labels after seed fit.
        known = set(str(c) for c in encoder.classes_)
        incoming = set(labeled_values.astype(str))
        if not incoming.issubset(known):
            encoder = LabelEncoder()
            encoder.fit(labeled_values.astype(str))

    y_codes = encoder.transform(labeled_values.astype(str))
    classes = tuple(encoder.classes_)
    if len(classes) < 2:
        raise ValidationError(
            "Active learning classification requires at least 2 classes among "
            f"labeled train rows (found {classes!r})."
        )
    return (
        y_codes,
        encoder,
        classes,
        np.asarray(labeled_mask, dtype=bool),
        n_labeled,
        n_unlabeled,
    )


def decode_predictions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Map integer class codes back toward original label values.

    Attempts numeric coercion when the inverse-transformed label looks like a
    number so integer/float targets round-trip cleanly.

    Parameters
    ----------
    pred_codes:
        Integer prediction codes from the fitted label encoder.
    label_encoder:
        Fitted :class:`~sklearn.preprocessing.LabelEncoder` used during fit.

    Returns
    -------
    list[Any]
        Decoded labels in the same order as ``pred_codes``.
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
