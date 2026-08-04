"""Bootstrap / stratified-subsample stability bands for disparity gaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.fairness.metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_gaps,
    group_selection_rates,
)

StabilityMethod = Literal["bootstrap", "stratified_subsample"]


@dataclass(slots=True)
class FairnessStability:
    """Disclosed resampling bands for observational gap metrics."""

    method: StabilityMethod
    n_resamples: int
    random_state: int | None
    confidence_level: float
    subsample_fraction: float | None = None
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        return asdict(self)


def estimate_gap_stability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    *,
    positive_label: Any = 1,
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    method: StabilityMethod = "bootstrap",
    subsample_fraction: float = 0.8,
    random_state: int | None = 0,
) -> FairnessStability:
    """Estimate percentile CI / bands for DP, DI, and equalized-odds gaps.

    Resampling is over rows of the evaluated partition only. Bands describe
    sampling variability of observational gaps — they are not causal
    uncertainty and do not certify fairness.

    Parameters
    ----------
    y_true, y_pred, sensitive:
        Aligned arrays for the audit partition.
    positive_label:
        Positive class encoding.
    n_resamples:
        Number of bootstrap or subsample draws (must be ≥ 2).
    confidence_level:
        Central percentile interval width in ``(0, 1)``.
    method:
        ``bootstrap`` (with-replacement) or ``stratified_subsample``
        (without-replacement, stratified by sensitive group).
    subsample_fraction:
        Fraction of rows kept per stratified subsample (ignored for bootstrap).
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    FairnessStability
        Point estimates plus ``ci_low`` / ``ci_high`` / ``std`` per gap metric.

    Raises
    ------
    ValidationError
        On invalid resampling configuration or empty inputs.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    sens = np.asarray(sensitive)
    n = len(yt)
    if n == 0:
        raise ValidationError("Stability estimation requires at least one row.")
    if n_resamples < 2:
        raise ValidationError("n_resamples must be >= 2.")
    if not (0.0 < confidence_level < 1.0):
        raise ValidationError("confidence_level must be in (0, 1).")
    if method not in ("bootstrap", "stratified_subsample"):
        raise ValidationError(
            f"Unknown stability method {method!r}; "
            "use 'bootstrap' or 'stratified_subsample'."
        )
    if method == "stratified_subsample" and not (0.0 < subsample_fraction <= 1.0):
        raise ValidationError("subsample_fraction must be in (0, 1].")

    rng = np.random.default_rng(random_state)
    point = _gap_vector(yt, yp, sens, positive_label=positive_label)
    draws: dict[str, list[float]] = {k: [] for k in point}

    if method == "bootstrap":
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            vec = _gap_vector(yt[idx], yp[idx], sens[idx], positive_label=positive_label)
            for key, val in vec.items():
                if val == val:  # not NaN
                    draws[key].append(float(val))
    else:
        groups = sorted({str(g) for g in sens})
        group_indices = {
            g: np.flatnonzero(np.asarray([str(s) == g for s in sens])) for g in groups
        }
        for _ in range(n_resamples):
            parts: list[np.ndarray] = []
            for g in groups:
                gidx = group_indices[g]
                if len(gidx) == 0:
                    continue
                k = max(1, int(round(len(gidx) * subsample_fraction)))
                k = min(k, len(gidx))
                chosen = rng.choice(gidx, size=k, replace=False)
                parts.append(chosen)
            if not parts:
                continue
            idx = np.concatenate(parts)
            rng.shuffle(idx)
            vec = _gap_vector(yt[idx], yp[idx], sens[idx], positive_label=positive_label)
            for key, val in vec.items():
                if val == val:
                    draws[key].append(float(val))

    alpha = 1.0 - confidence_level
    metrics: dict[str, dict[str, float | None]] = {}
    for key, point_val in point.items():
        samples = draws[key]
        if not samples:
            metrics[key] = {
                "point": _finite_or_none(point_val),
                "ci_low": None,
                "ci_high": None,
                "std": None,
                "n_finite_draws": 0,
            }
            continue
        arr = np.asarray(samples, dtype=float)
        metrics[key] = {
            "point": _finite_or_none(point_val),
            "ci_low": float(np.quantile(arr, alpha / 2.0)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_finite_draws": int(len(arr)),
        }

    disclosures = (
        "Stability bands reflect resampling variability of observational gaps "
        "on one partition; they are not causal uncertainty intervals.",
        "Undefined gap draws (e.g. empty positives in a resample) are dropped "
        "before percentile aggregation; check n_finite_draws.",
        f"Method={method!r}, n_resamples={n_resamples}, "
        f"confidence_level={confidence_level}.",
    )
    return FairnessStability(
        method=method,
        n_resamples=int(n_resamples),
        random_state=random_state,
        confidence_level=float(confidence_level),
        subsample_fraction=(
            float(subsample_fraction) if method == "stratified_subsample" else None
        ),
        metrics=metrics,
        disclosures=disclosures,
    )


def _gap_vector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    *,
    positive_label: Any,
) -> dict[str, float]:
    rates = group_selection_rates(y_pred, sensitive, positive_label=positive_label)
    _tpr, _fpr, tpr_gap, fpr_gap = equalized_odds_gaps(
        y_true, y_pred, sensitive, positive_label=positive_label
    )
    di = disparate_impact_ratio(rates)
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(rates)
        ),
        "disparate_impact_ratio": float(di) if di is not None else float("nan"),
        "equalized_odds_tpr_difference": (
            float(tpr_gap) if tpr_gap is not None else float("nan")
        ),
        "equalized_odds_fpr_difference": (
            float(fpr_gap) if fpr_gap is not None else float("nan")
        ),
    }


def _finite_or_none(value: float) -> float | None:
    if value != value:  # NaN
        return None
    return float(value)
