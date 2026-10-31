"""Changepoint detection (ruptures or lightweight CUSUM)."""

from __future__ import annotations

from numbers import Real

import numpy as np

from buildml.core.errors import ValidationError
from buildml.timeseries.extras import require_ruptures, ruptures_available
from buildml.timeseries.results import TSChangepointResult
from buildml.timeseries.types import ChangepointMethod


def detect_changepoints(
    y: np.ndarray,
    *,
    method: ChangepointMethod = "pelt",
    penalty: float = 10.0,
    target_column: str = "target",
) -> TSChangepointResult:
    """Detect mean-shift changepoints in a univariate ordered series.

    Runs PELT or binary segmentation via ruptures when installed, otherwise a
    lightweight CUSUM heuristic. Changepoints are descriptive on the analyzed
    scope: refit forecasts after structural breaks rather than treating breaks
    as labels.

    Parameters
    ----------
    y:
        One-dimensional observation vector in temporal order.
    method:
        ``pelt`` or ``binseg`` (ruptures) or ``cusum`` (core fallback).
    penalty:
        Ruptures penalty for PELT/BinSeg, or CUSUM threshold scale for ``cusum``.
    target_column:
        Name recorded on the result for traceability in combined reports.

    Returns
    -------
    TSChangepointResult
        Detected index boundaries, per-segment means, disclosures, and warnings
        when a requested method fell back to CUSUM.

    Raises
    ------
    ValidationError
        When the series is not finite, real and one-dimensional, has fewer
        than four points, the penalty is not finite and non-negative, or the
        method is unsupported.
    """
    if isinstance(penalty, bool) or not isinstance(penalty, Real):
        raise ValidationError("penalty must be a finite non-negative real number")
    try:
        penalty = float(penalty)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("penalty must be a finite non-negative real number") from exc
    if not np.isfinite(penalty) or penalty < 0:
        raise ValidationError("penalty must be a finite non-negative real number")
    try:
        values = np.asarray(y)
        if values.ndim != 1 or np.iscomplexobj(values):
            raise ValueError("expected a real one-dimensional vector")
        y = np.asarray(values, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("non-finite observations")
    except (TypeError, ValueError) as exc:
        raise ValidationError("series must be a finite real one-dimensional vector") from exc
    n = int(y.shape[0])
    if n < 4:
        raise ValidationError("Need at least 4 points for changepoint detection")

    warnings: list[str] = []
    disclosures: list[str] = [
        f"Changepoint method={method}, n={n}, penalty={penalty}.",
        "Changepoints are descriptive on the analyzed scope: refit forecasts after breaks.",
    ]

    if method in {"pelt", "binseg"}:
        if not ruptures_available():
            warnings.append(
                f"{method} requires ruptures; falling back to cusum. "
                "Install buildml[timeseries]."
            )
            method = "cusum"

    if method == "pelt":
        require_ruptures(feature="PELT changepoint detection")
        import ruptures as rpt

        algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(y.reshape(-1, 1))
        bkps = algo.predict(pen=float(penalty))
        # ruptures returns end indices (exclusive); convert to changepoint starts
        indices = tuple(int(b) for b in bkps[:-1])
        disclosures.append("PELT changepoint search via ruptures (RBF cost, min_size=2).")
        used = "pelt"
    elif method == "binseg":
        require_ruptures(feature="Binary segmentation changepoint detection")
        import ruptures as rpt

        algo = rpt.Binseg(model="l2", min_size=2, jump=1).fit(y.reshape(-1, 1))
        bkps = algo.predict(pen=float(penalty))
        indices = tuple(int(b) for b in bkps[:-1])
        disclosures.append(
            f"Binary segmentation via ruptures (penalty={penalty}, L2 cost)."
        )
        used = "binseg"
    elif method == "cusum":
        indices = _cusum_changepoints(y, threshold=float(penalty))
        disclosures.append(
            "Lightweight CUSUM-style changepoints (core fallback): "
            "threshold scales with series std."
        )
        used = "cusum"
    else:
        raise ValidationError(f"Unsupported changepoint method '{method}'")

    segments = _segment_means(y, indices)
    return TSChangepointResult(
        method=used,
        target_column=target_column,
        n_points=n,
        changepoint_indices=indices,
        n_segments=len(segments),
        segment_means=tuple(segments),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _cusum_changepoints(y: np.ndarray, *, threshold: float) -> tuple[int, ...]:
    """Simple mean-shift CUSUM heuristic."""
    n = len(y)
    std = float(np.std(y)) or 1.0
    thresh = threshold * std / np.sqrt(n)
    cusum = np.zeros(n, dtype=float)
    mean = float(np.mean(y))
    for i in range(1, n):
        cusum[i] = cusum[i - 1] + (y[i] - mean)
    points: list[int] = []
    last = 0
    for i in range(2, n - 1):
        if abs(cusum[i] - cusum[last]) > thresh:
            points.append(i)
            last = i
            mean = float(np.mean(y[last:]))
    return tuple(points)


def _segment_means(y: np.ndarray, changepoints: tuple[int, ...]) -> list[float]:
    bounds = [0, *changepoints, len(y)]
    return [float(np.mean(y[bounds[i] : bounds[i + 1]])) for i in range(len(bounds) - 1)]
