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


def method_requires_extra(method: str) -> str | None:
    if method in STATSMODELS_METHODS:
        return None if statsmodels_available() else "timeseries"
    if method == PROPHET_METHOD:
        return None if prophet_available() else "timeseries-prophet"
    if method == NEURAL_METHOD:
        return None if neuralforecast_available() else "timeseries-ml"
    return None


def method_backend(method: str) -> str:
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
    """Pick industry default when caller passes None or 'auto'."""
    if requested is None or requested == "auto":
        return DEFAULT_INDUSTRY_METHOD
    return requested


def list_forecast_methods(*, include_neural: bool = True) -> tuple[dict[str, Any], ...]:
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
