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
    """Build the class vocabulary from the training labels.

    Collects the distinct labels and sorts them. Position in the returned tuple
    becomes the class index the network learns, and the tuple itself is what
    turns a predicted index back into a label the caller recognises.

    Parameters
    ----------
    y:
        The training targets. Any pandas-coercible sequence.

    Returns
    -------
    tuple
        The distinct labels in sorted order, matching scikit-learn's
        ``LabelEncoder.classes_``.

    Raises
    ------
    ValidationError
        If the targets contain ``NaN``, or if the labels cannot be sorted —
        typically a column mixing strings and numbers, where no consistent
        ordering exists and any choice would be arbitrary.

    Notes
    -----
    **Sorting is what makes the mapping reproducible.** Encounter order would
    make the class indices depend on how the rows happened to be shuffled, so
    two runs on the same data could disagree about which class is 0 — and a
    saved model would then be misread by a re-fitted vocabulary.

    **Fit this on training labels only.** A class that appears solely in the
    test set is one the model never saw, and it should surface as an error
    during encoding rather than be silently absorbed.

    See Also
    --------
    encode_class_targets : Applying this vocabulary.
    """
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
    """Convert labels into the contiguous indices the loss function needs.

    Replaces each label with its position in the vocabulary. ``CrossEntropyLoss``
    interprets targets as indices into the output layer, so labels such as
    ``{10, 20, 30}`` must become ``{0, 1, 2}`` — leaving them raw would ask the
    network for a 31-wide output and train three of its columns.

    Parameters
    ----------
    y:
        The labels to encode.
    class_labels:
        The vocabulary from :func:`fit_class_labels`.

    Returns
    -------
    numpy.ndarray
        Int64 indices in ``0..K-1``, one per input row.

    Raises
    ------
    ValidationError
        If the vocabulary is empty, if the labels contain ``NaN``, or if any
        label is absent from the vocabulary.

    Notes
    -----
    **An unknown label is an error, not a fallback.** It means the holdout
    contains a class the model was never taught, so no prediction it makes for
    those rows can be right. Mapping them to a catch-all index would bury a data
    problem inside a plausible-looking metric.

    See Also
    --------
    fit_class_labels : Building the vocabulary.
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
    """Work out how wide the output layer should be.

    The width is the number of distinct classes — the vocabulary length, never
    the largest label value. Labels ``{10, 20, 30}`` need three outputs, not
    thirty-one.

    Parameters
    ----------
    class_labels:
        The vocabulary from :func:`fit_class_labels`.
    minimum:
        The smallest width to return. Defaults to 2.

    Returns
    -------
    int
        The number of output units the classification head needs.

    Notes
    -----
    **The floor of two exists for the degenerate case.** A training partition
    with a single observed class would otherwise produce a one-wide head, which
    softmax turns into a constant 1.0 and a gradient of zero — the model cannot
    train and gives no indication why. Two outputs at least keep the machinery
    working, though a single-class training set is a data problem worth fixing
    rather than training around.
    """
    return max(int(minimum), len(class_labels) or int(minimum))
