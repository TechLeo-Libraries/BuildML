"""Contiguous class-id encoding for Torch classification paths.

CrossEntropyLoss expects targets in ``{0, …, K-1}``. Callers often pass
sparse integer ids (e.g. ``{10, 20, 30}``). Using ``n_classes = len(unique)``
while leaving raw ids in the tensors is a silent correctness footgun.

This mirrors classical BuildML / sklearn ``LabelEncoder`` behavior: fit the
mapping on train labels only, store original ids in ``class_labels`` (index
``i`` ↔ original label), and encode every partition to contiguous indices.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError


def fit_class_labels(y: Any) -> tuple[Any, ...]:
    """Return sorted unique train labels (LabelEncoder ``classes_`` order)."""
    series = pd.Series(y)
    if series.isna().any():
        raise ValidationError("Target contains NaN; clean labels before Torch loaders")
    uniques = pd.unique(series)
    try:
        ordered = sorted(uniques.tolist())
    except TypeError as exc:  # pragma: no cover - mixed-type edge
        raise ValidationError(
            "Class labels must be sortable for contiguous Torch encoding"
        ) from exc
    return tuple(ordered)


def encode_class_targets(y: Any, class_labels: Sequence[Any]) -> np.ndarray:
    """Map original labels to contiguous ``0..K-1`` using train ``class_labels``.

    Unknown labels (not in the train mapping) raise ``ValidationError``.
    """
    if not class_labels:
        raise ValidationError("class_labels is empty; cannot encode classification targets")
    series = pd.Series(y)
    if series.isna().any():
        raise ValidationError("Target contains NaN; clean labels before Torch loaders")
    cat = pd.Categorical(series, categories=list(class_labels), ordered=True)
    codes = np.asarray(cat.codes, dtype=np.int64)
    if (codes < 0).any():
        known = ", ".join(repr(v) for v in class_labels)
        raise ValidationError(
            "Target contains class labels not present in the train partition "
            f"(known labels: {known})."
        )
    return codes


def n_classes_from_labels(class_labels: Sequence[Any], *, minimum: int = 2) -> int:
    """Head width from encoded label cardinality (never ``max(label)+1``)."""
    return max(int(minimum), len(class_labels) or int(minimum))
