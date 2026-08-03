"""Forecast method catalog — backends, defaults, install hints."""

from __future__ import annotations

from typing import Any

from buildml.forecasting.extras import (
    industry_forecast_available,
    neuralforecast_available,
    prophet_available,
    statsmodels_available,
)

# Core sklearn / baseline methods (always available)
CORE_BASELINE_METHODS: frozenset[str] = frozenset(
    {"naive", "seasonal_naive", "drift", "mean", "lag_ridge", "lag_hgb"}
)

# statsmodels methods (buildml[timeseries])
STATSMODELS_METHODS: frozenset[str] = frozenset({"arima", "ets", "sarimax", "auto_arima"})

PROPHET_METHOD = "prophet"
NEURAL_METHOD = "nbeats"

ALL_FORECAST_METHODS: frozenset[str] = (
    CORE_BASELINE_METHODS | STATSMODELS_METHODS | {PROPHET_METHOD, NEURAL_METHOD}
)

DEFAULT_INDUSTRY_METHOD = "ets" if statsmodels_available() else "lag_ridge"
DEFAULT_TABULAR_METHOD = "lag_ridge"

# Methods that accept contemporaneous exog at fit time (others are univariate-only).
EXOG_COMPATIBLE_METHODS: frozenset[str] = frozenset(
    {"lag_ridge", "lag_hgb", "arima", "sarimax"}
)


def method_supports_exog(method: str) -> bool:
    """Return whether a forecast method can use exogenous columns at fit time.

    Univariate-only industry paths (ETS, auto_arima, Prophet, N-BEATS) and
    simple baselines ignore ``exog_columns``; lag models and ARIMA/SARIMAX wire
    contemporaneous exog into the fitter. Call before accepting ``exog_columns``
    from a caller so invalid combinations fail at the boundary.

    Parameters
    ----------
    method:
        Forecast method key from the catalog.

    Returns
    -------
    bool
        ``True`` when exogenous columns are wired into the fitter; ``False`` for
        univariate-only paths.
    """
    return str(method) in EXOG_COMPATIBLE_METHODS


def method_requires_extra(method: str) -> str | None:
    """Return the BuildML extra name required by a forecast method.

    Checks optional-backend install probes so callers can raise
    :class:`MissingExtraError` before attempting fit.

    Parameters
    ----------
    method:
        Forecast method key from the catalog.

    Returns
    -------
    str or None
        Extra name such as ``timeseries-prophet``, or ``None`` when the method
        is available without an extra.
    """
    if method in STATSMODELS_METHODS:
        return None if statsmodels_available() else "timeseries"
    if method == PROPHET_METHOD:
        return None if prophet_available() else "timeseries-prophet"
    if method == NEURAL_METHOD:
        return None if neuralforecast_available() else "timeseries-ml"
    return None


def method_backend(method: str) -> str:
    """Map a forecast method to its implementation backend label.

    Used by the capability matrix and history logs to disclose whether a fit
    used baseline, sklearn lag, statsmodels, Prophet, or neuralforecast code.

    Parameters
    ----------
    method:
        Forecast method key from the catalog.

    Returns
    -------
    str
        Backend label such as ``baseline``, ``sklearn``, ``statsmodels``,
        ``prophet``, ``neuralforecast``, or ``unknown``.
    """
    if method in CORE_BASELINE_METHODS:
        if method in {"lag_ridge", "lag_hgb"}:
            return "sklearn"
        return "baseline"
    if method in STATSMODELS_METHODS:
        return "statsmodels"
    if method == PROPHET_METHOD:
        return "prophet"
    if method == NEURAL_METHOD:
        return "neuralforecast"
    return "unknown"


def resolve_default_method(requested: str | None = None) -> str:
    """Resolve ``auto`` or ``None`` to the honest industry default method.

    Prefers ETS via statsmodels when ``buildml[timeseries]`` is installed;
    otherwise falls back to the lag_ridge tabular baseline.

    Parameters
    ----------
    requested:
        Caller method key, or ``None`` / ``auto`` for the default.

    Returns
    -------
    str
        Concrete method name ready for :func:`fit_forecaster`.
    """
    if requested is None or requested == "auto":
        return DEFAULT_INDUSTRY_METHOD
    return requested


def list_forecast_methods(*, include_neural: bool = True) -> tuple[dict[str, Any], ...]:
    """List catalogued forecast methods with backend and install metadata.

    Builds rows from core baselines, statsmodels methods, Prophet, and optional
    N-BEATS so Session walkthroughs can show honest availability.

    Parameters
    ----------
    include_neural:
        When ``False``, omit the N-BEATS neuralforecast entry.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Per-method dicts with ``method``, ``backend``, ``requires_extra``,
        and ``default`` flags.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(CORE_BASELINE_METHODS):
        rows.append(
            {
                "method": name,
                "backend": method_backend(name),
                "requires_extra": None,
                "default": name == DEFAULT_TABULAR_METHOD and not statsmodels_available(),
            }
        )
    for name in sorted(STATSMODELS_METHODS):
        extra = method_requires_extra(name)
        rows.append(
            {
                "method": name,
                "backend": "statsmodels",
                "requires_extra": extra,
                "default": name == DEFAULT_INDUSTRY_METHOD,
            }
        )
    rows.append(
        {
            "method": PROPHET_METHOD,
            "backend": "prophet",
            "requires_extra": method_requires_extra(PROPHET_METHOD),
            "default": False,
        }
    )
    if include_neural:
        rows.append(
            {
                "method": NEURAL_METHOD,
                "backend": "neuralforecast",
                "requires_extra": method_requires_extra(NEURAL_METHOD),
                "default": False,
            }
        )
    return tuple(rows)


def forecast_status_payload() -> dict[str, Any]:
    """Build install and default-method status for forecasting walkthroughs.

    Reports optional-backend availability, recommended extras, and boundary
    disclosures about temporal splits and industry fallbacks.

    Returns
    -------
    dict[str, Any]
        Backend availability flags, default/fallback methods, and disclosures.
    """
    return {
        "statsmodels_available": statsmodels_available(),
        "prophet_available": prophet_available(),
        "neuralforecast_available": neuralforecast_available(),
        "default_method": DEFAULT_INDUSTRY_METHOD,
        "fallback_method": DEFAULT_TABULAR_METHOD,
        "recommended_extra": "timeseries",
        "disclosures": [
            "Forecasting defaults to ETS/ARIMA via statsmodels when buildml[timeseries] "
            "is installed; lag_ridge/lag_hgb baselines remain as core fallback.",
            "Prophet requires buildml[timeseries-prophet]; N-BEATS requires buildml[timeseries-ml].",
            "Random/stratified splits are refused; use time_split.",
        ],
    }


def forecast_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for forecasting backends and methods.

    Reports baseline, sklearn lag, statsmodels, Prophet, and neuralforecast
    availability, install hints, and explicit non-goals for teaching overlays.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, method rows, install hints, and non-goals.
    """
    return {
        "backends": {
            "baseline": {
                "available": True,
                "extra": None,
                "methods": sorted(
                    m for m in CORE_BASELINE_METHODS if m not in {"lag_ridge", "lag_hgb"}
                ),
                "notes": "Naive / seasonal / drift / mean baselines — always available.",
            },
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": ["lag_ridge", "lag_hgb"],
                "notes": "Lag-feature tabular regressors — always available fallback.",
            },
            "statsmodels": {
                "available": statsmodels_available(),
                "extra": "timeseries",
                "methods": sorted(STATSMODELS_METHODS),
                "notes": "ETS/ARIMA/SARIMAX when buildml[timeseries] is installed.",
            },
            "prophet": {
                "available": prophet_available(),
                "extra": "timeseries-prophet",
                "methods": [PROPHET_METHOD],
                "notes": "Prophet backend (buildml[timeseries-prophet]).",
            },
            "neuralforecast": {
                "available": neuralforecast_available(),
                "extra": "timeseries-ml",
                "methods": [NEURAL_METHOD],
                "notes": (
                    "N-BEATS via NeuralForecast (buildml[timeseries-ml]); "
                    "Python/platform markers may skip the pin on Py3.13."
                ),
            },
        },
        "default_method_when_installed": DEFAULT_INDUSTRY_METHOD,
        "fallback_method": DEFAULT_TABULAR_METHOD,
        "methods": list(list_forecast_methods()),
        "install_hints": {
            "timeseries": "pip install 'buildml[timeseries]'",
            "timeseries-prophet": "pip install 'buildml[timeseries-prophet]'",
            "timeseries-ml": "pip install 'buildml[timeseries-ml]'",
        },
        "non_goals": [
            "Full Nixtla research zoo",
            "Streaming / online forecasting product",
        ],
        "industry_forecast_present": industry_forecast_available(),
    }
