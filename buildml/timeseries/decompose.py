"""Seasonal decomposition (STL / classical / core fallback)."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.timeseries.extras import require_statsmodels, statsmodels_available
from buildml.timeseries.results import TSDecomposeResult
from buildml.timeseries.series import infer_seasonal_period
from buildml.timeseries.types import DecomposeMethod


def decompose_series(
    y: np.ndarray,
    *,
    method: DecomposeMethod = "stl",
    seasonal_period: int | None = None,
    target_column: str = "target",
    time_column: str = "time",
    timestamps: tuple[str, ...] = (),
) -> TSDecomposeResult:
    """Decompose a univariate series into trend, seasonal, and residual components.

    Prefers STL or classical additive decomposition via statsmodels when
    ``buildml[timeseries]`` is installed; otherwise uses a centered moving-average
    trend with a repeating seasonal profile. Seasonal period is inferred when not
    supplied.

    Parameters
    ----------
    y:
        One-dimensional observation vector in temporal order.
    method:
        ``stl``, ``classical``, or ``moving_average``.
    seasonal_period:
        Cycle length (e.g. 7 for weekly seasonality on daily data). Inferred from
        series length when ``None``.
    target_column, time_column:
        Names recorded on the result for downstream reports.
    timestamps:
        Optional string stamps aligned with ``y`` for export and display.

    Returns
    -------
    TSDecomposeResult
        Component vectors, method actually used (after fallback), disclosures, and
        warnings when the series is short relative to the seasonal period.

    Raises
    ------
    ValidationError
        When ``method`` is unsupported.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    period = infer_seasonal_period(y, seasonal_period=seasonal_period)
    warnings: list[str] = []
    disclosures: list[str] = [
        f"Decomposition method={method}, seasonal_period={period}, n={n}.",
        "Train-only scope is recommended before forecasting to avoid peeking at holdout.",
    ]

    if method == "moving_average" or (
        method in {"stl", "classical"} and not statsmodels_available()
    ):
        if method in {"stl", "classical"}:
            warnings.append(
                f"{method} requested but statsmodels is not installed; "
                "falling back to moving_average. Install buildml[timeseries]."
            )
        trend, seasonal, resid = _moving_average_decompose(y, period)
        disclosures.append(
            "Core moving-average decomposition: centered rolling mean trend, "
            "seasonal profile from train-period averages, residual = observed - trend - seasonal."
        )
        used = "moving_average"
    elif method == "stl":
        require_statsmodels(feature="STL decomposition")
        from statsmodels.tsa.seasonal import STL

        stl = STL(y, period=period, robust=True)
        result = stl.fit()
        trend = np.asarray(result.trend, dtype=float)
        seasonal = np.asarray(result.seasonal, dtype=float)
        resid = np.asarray(result.resid, dtype=float)
        disclosures.append(
            "STL (Seasonal-Trend decomposition using Loess) via statsmodels; "
            "robust=True down-weights outliers in trend/seasonal estimation."
        )
        used = "stl"
    elif method == "classical":
        require_statsmodels(feature="classical seasonal decomposition")
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")
        trend = np.asarray(result.trend, dtype=float)
        seasonal = np.asarray(result.seasonal, dtype=float)
        resid = np.asarray(result.resid, dtype=float)
        # seasonal_decompose may leave NaN at edges
        mask = np.isnan(trend) | np.isnan(seasonal) | np.isnan(resid)
        if mask.any():
            warnings.append(
                f"classical decomposition has {int(mask.sum())} edge NaN(s); "
                "filled with 0 for export."
            )
            trend = np.nan_to_num(trend, nan=0.0)
            seasonal = np.nan_to_num(seasonal, nan=0.0)
            resid = np.nan_to_num(resid, nan=0.0)
        disclosures.append(
            "Classical additive seasonal_decompose via statsmodels "
            "(extrapolate_trend='freq')."
        )
        used = "classical"
    else:
        raise ValidationError(f"Unsupported decompose method '{method}'")

    if n < 2 * period:
        warnings.append(
            f"n={n} is short relative to seasonal_period={period}; "
            "seasonal structure may be unstable."
        )

    return TSDecomposeResult(
        method=used,
        target_column=target_column,
        time_column=time_column,
        n_points=n,
        seasonal_period=period,
        trend=tuple(float(v) for v in trend.tolist()),
        seasonal=tuple(float(v) for v in seasonal.tolist()),
        residual=tuple(float(v) for v in resid.tolist()),
        observed=tuple(float(v) for v in y.tolist()),
        timestamps=timestamps,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _moving_average_decompose(
    y: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(y.shape[0])
    window = max(3, min(period, n))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    trend = np.convolve(padded, kernel, mode="valid")[:n]
    detrended = y - trend
    seasonal_profile = np.zeros(period, dtype=float)
    counts = np.zeros(period, dtype=float)
    for i, val in enumerate(detrended):
        idx = i % period
        seasonal_profile[idx] += val
        counts[idx] += 1.0
    counts = np.where(counts == 0, 1.0, counts)
    seasonal_profile /= counts
    seasonal = np.array([seasonal_profile[i % period] for i in range(n)], dtype=float)
    resid = y - trend - seasonal
    return trend, seasonal, resid
