"""Split conformal prediction helpers (train-only calibration carve).

Implements MAPIE-style absolute-residual split conformal for regression and
a simple probability-based prediction-set construction for classifiers :
without requiring the MAPIE package. Calibration rows are carved from the
Session **train** partition only; validation/test are never used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the finite-sample split-conformal quantile at level ``1 - alpha``.

    Uses the standard ``ceil((n+1)(1-alpha))/n`` order-statistic index on sorted
    nonconformity scores from the train-only calibration carve.

    Parameters
    ----------
    scores:
        Nonconformity scores from calibration rows.
    alpha:
        Miscoverage rate in ``(0, 1)``.

    Returns
    -------
    float
        Quantile applied when building intervals or prediction sets.

    Raises
    ------
    ValidationError
        When ``scores`` is empty or ``alpha`` is outside ``(0, 1)``.
    """
    arr = np.asarray(scores, dtype=float).ravel()
    n = int(arr.size)
    if n < 1:
        raise ValidationError("Conformal calibration requires at least one score.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValidationError(f"alpha must be in (0, 1); got {alpha}.")
    level = 1.0 - float(alpha)
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)
    ordered = np.sort(arr)
    return float(ordered[k - 1])


def absolute_residual_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute regression nonconformity scores as absolute residuals.

    Used by native split conformal calibration on the train carve before
    interval construction on holdout partitions.

    Parameters
    ----------
    y_true, y_pred:
        Parallel calibration targets and point predictions.

    Returns
    -------
    numpy.ndarray
        Absolute residual scores ``|y - ŷ|`` per row.
    """
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


def regression_intervals(
    y_pred: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric absolute-residual intervals around point predictions.

    Applies a calibrated conformal quantile as a fixed half-width around each
    point prediction from the fitted probabilistic plan.

    Parameters
    ----------
    y_pred:
        Point predictions for the scoring partition.
    quantile:
        Calibrated conformal half-width from train-only calibration.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper bound arrays aligned with ``y_pred``.
    """
    pred = np.asarray(y_pred, dtype=float)
    q = float(quantile)
    return pred - q, pred + q


def classification_nonconformity(
    proba: np.ndarray,
    y_true_codes: np.ndarray,
) -> np.ndarray:
    """Compute classification nonconformity as ``1 - p̂(y_true)``.

    Higher scores mean the observed label received lower predicted probability
    on calibration rows.

    Parameters
    ----------
    proba:
        Predicted class probabilities shaped ``(n_rows, n_classes)``.
    y_true_codes:
        Integer-encoded true class indices aligned with ``proba`` rows.

    Returns
    -------
    numpy.ndarray
        Nonconformity score per calibration row.

    Raises
    ------
    ValidationError
        When shapes mismatch or class codes are out of range.
    """
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
    """Build prediction sets from calibrated nonconformity thresholds.

    Includes every class whose nonconformity ``1 - p̂(y)`` is at most the
    conformal quantile; falls back to the argmax class when the set is empty.

    Parameters
    ----------
    proba:
        Predicted class probabilities shaped ``(n_rows, n_classes)``.
    quantile:
        Calibrated conformal threshold from train-only calibration.
    classes:
        Class labels corresponding to probability columns.

    Returns
    -------
    list[tuple]
        One prediction-set tuple per row.
    """
    p = np.asarray(proba, dtype=float)
    q = float(quantile)
    sets: list[tuple[Any, ...]] = []
    for row in p:
        members = [classes[j] for j in range(len(classes)) if (1.0 - row[j]) <= q]
        if not members:
            members = [classes[int(np.argmax(row))]]
        sets.append(tuple(members))
    return sets
