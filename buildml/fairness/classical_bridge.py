"""Per-group classical classification metrics bridged into fairness reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.fairness.metrics import _as_bool_pred


def per_group_classical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    *,
    positive_label: Any = 1,
    y_score: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute accuracy / precision / recall / F1 (and optional AUC) per group.

    Metrics are standard binary classification definitions conditioned on the
    caller-declared ``positive_label``. AUC is included only when ``y_score``
    is provided and a group has both classes in truth.

    Parameters
    ----------
    y_true, y_pred, sensitive:
        Aligned arrays.
    positive_label:
        Positive class encoding.
    y_score:
        Optional positive-class scores/probabilities for ROC-AUC.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Mapping group key → metric name → value (``None`` when undefined).
    """
    yt = _as_bool_pred(np.asarray(y_true), positive_label)
    yp = _as_bool_pred(np.asarray(y_pred), positive_label)
    sens = np.asarray(sensitive)
    scores = None if y_score is None else np.asarray(y_score, dtype=float)
    if scores is not None and len(scores) != len(yt):
        raise ValidationError("y_score length must match y_true.")

    out: dict[str, dict[str, float | None]] = {}
    for group in sorted({str(g) for g in sens}):
        mask = np.asarray([str(g) == group for g in sens])
        n = int(mask.sum())
        if n == 0:
            out[group] = {
                "n": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
            }
            continue
        yt_g = yt[mask]
        yp_g = yp[mask]
        tp = int(np.sum(yt_g & yp_g))
        fp = int(np.sum(~yt_g & yp_g))
        fn = int(np.sum(yt_g & ~yp_g))
        tn = int(np.sum(~yt_g & ~yp_g))
        accuracy = float((tp + tn) / n)
        precision = float(tp / (tp + fp)) if (tp + fp) else None
        recall = float(tp / (tp + fn)) if (tp + fn) else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))
        auc: float | None = None
        if scores is not None:
            auc = _safe_roc_auc(yt_g, scores[mask])
        out[group] = {
            "n": n,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
        }
    return out


def _safe_roc_auc(y_true_bool: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC when both classes are present; else ``None``."""
    n_pos = int(np.sum(y_true_bool))
    n_neg = int(len(y_true_bool) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    if not np.all(np.isfinite(scores)):
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true_bool.astype(int), scores))
    except Exception:  # noqa: BLE001 — keep report resilient
        return None
