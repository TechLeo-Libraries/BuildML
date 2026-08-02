"""Split conformal prediction helpers (train-only calibration carve).

Implements MAPIE-style absolute-residual split conformal for regression and
a simple probability-based prediction-set construction for classifiers —
without requiring the MAPIE package. Calibration rows are carved from the
Session **train** partition only; validation/test are never used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile at level ``1 - alpha``.

    Uses the standard ``ceil((n+1)(1-alpha))/n`` order-statistic index
    (clipped), equivalent to the common MAPIE split-conformal recipe for
    absolute residual scores.
    """
    arr = np.asarray(scores, dtype=float).ravel()
    n = int(arr.size)
    if n < 1:
        raise ValidationError("Conformal calibration requires at least one score.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValidationError(f"alpha must be in (0, 1); got {alpha}.")
    level = 1.0 - float(alpha)
    # Finite-sample correction: q-hat = Rank ceil((n+1)(1-α)) / n
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)
    ordered = np.sort(arr)
    return float(ordered[k - 1])


def absolute_residual_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Nonconformity scores for regression: |y − ŷ|."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


def regression_intervals(
    y_pred: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absolute-residual intervals around point predictions."""
    pred = np.asarray(y_pred, dtype=float)
    q = float(quantile)
    return pred - q, pred + q


def classification_nonconformity(
    proba: np.ndarray,
    y_true_codes: np.ndarray,
) -> np.ndarray:
    """Score = 1 − p̂(y_true); higher means less conforming."""
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true_codes).astype(int)
    if p.ndim != 2:
        raise ValidationError("Classification conformal needs a 2-d probability matrix.")
    if y.shape[0] != p.shape[0]:
        raise ValidationError("y_true length must match probability rows.")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValidationError("y_true codes out of range for probability columns.")
    return 1.0 - p[np.arange(len(y)), y]


def classification_prediction_sets(
    proba: np.ndarray,
    quantile: float,
    classes: tuple[Any, ...],
) -> list[tuple[Any, ...]]:
    """Include labels with nonconformity ≤ quantile (1 − p̂(y) ≤ q)."""
    p = np.asarray(proba, dtype=float)
    q = float(quantile)
    sets: list[tuple[Any, ...]] = []
    for row in p:
        members = [classes[j] for j in range(len(classes)) if (1.0 - row[j]) <= q]
        if not members:
            # Guarantee non-empty set: keep the MAP label.
            members = [classes[int(np.argmax(row))]]
        sets.append(tuple(members))
    return sets
