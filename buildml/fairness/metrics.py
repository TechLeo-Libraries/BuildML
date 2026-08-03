"""Binary classification group disparity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool_pred(y: np.ndarray, positive_label: Any) -> np.ndarray:
    return np.asarray(y) == positive_label


def group_selection_rates(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    *,
    positive_label: Any = 1,
) -> dict[str, float]:
    """Fraction predicted positive per sensitive group."""
    pred_pos = _as_bool_pred(y_pred, positive_label)
    rates: dict[str, float] = {}
    for group in sorted({str(g) for g in sensitive}):
        mask = np.asarray([str(g) == group for g in sensitive])
        n = int(mask.sum())
        rates[group] = float(pred_pos[mask].mean()) if n else float("nan")
    return rates


def demographic_parity_difference(rates: dict[str, float]) -> float:
    """Max selection rate minus min selection rate across groups."""
    vals = [v for v in rates.values() if v == v]
    if not vals:
        return float("nan")
    return float(max(vals) - min(vals))


def disparate_impact_ratio(rates: dict[str, float]) -> float | None:
    """Min/max selection-rate ratio (None when undefined)."""
    vals = [v for v in rates.values() if v == v]
    if len(vals) < 2:
        return None
    hi = max(vals)
    lo = min(vals)
    if hi <= 0:
        return None
    return float(lo / hi)


def equalized_odds_gaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    *,
    positive_label: Any = 1,
) -> tuple[dict[str, float], dict[str, float], float | None, float | None]:
    """Per-group TPR/FPR and max-min gaps."""
    yt = _as_bool_pred(y_true, positive_label)
    yp = _as_bool_pred(y_pred, positive_label)
    tpr: dict[str, float] = {}
    fpr: dict[str, float] = {}
    for group in sorted({str(g) for g in sensitive}):
        mask = np.asarray([str(g) == group for g in sensitive])
        pos = mask & yt
        neg = mask & ~yt
        tpr[group] = float(yp[pos].mean()) if int(pos.sum()) else float("nan")
        fpr[group] = float(yp[neg].mean()) if int(neg.sum()) else float("nan")
    tpr_vals = [v for v in tpr.values() if v == v]
    fpr_vals = [v for v in fpr.values() if v == v]
    tpr_gap = float(max(tpr_vals) - min(tpr_vals)) if tpr_vals else None
    fpr_gap = float(max(fpr_vals) - min(fpr_vals)) if fpr_vals else None
    return tpr, fpr, tpr_gap, fpr_gap
