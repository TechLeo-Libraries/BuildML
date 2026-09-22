"""Classical time-series forecasting domain (lag/baseline Session path).

Behavior and limitations
------------------------
Industry defaults when ``buildml[timeseries]`` installed (ETS/ARIMA/SARIMAX).
Core lag/baseline fallback with clear MissingExtraError when industry methods
requested without extras. Prophet/N-BEATS behind separate extras.
Refuses shuffled random splits. Not a digital twin.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Optional:
``buildml[timeseries]``, ``buildml[timeseries-prophet]``, ``buildml[timeseries-ml]``.

Lazy imports: core never grows heavy forecast stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ForecastConfig",
    "ForecastEvalResult",
    "ForecastEvalStrategy",
    "ForecastFitResult",
    "ForecastGenerateResult",
    "ForecastMethod",
    "ForecastPlan",
    "evaluate_forecast",
    "fit_forecaster",
    "forecast_capability_matrix",
    "forecasting_status",
    "forecasting_status_for_session",
    "generate_forecast",
    "load_forecast_bundle",
    "save_forecast_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"ForecastMethod", "ForecastEvalStrategy", "ForecastConfig"}:
        from buildml.forecasting import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ForecastPlan",
        "ForecastFitResult",
        "ForecastGenerateResult",
        "ForecastEvalResult",
    }:
        from buildml.forecasting import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_forecaster":
        from buildml.forecasting.fit import fit_forecaster

        return fit_forecaster
    if name == "generate_forecast":
        from buildml.forecasting.predict import generate_forecast

        return generate_forecast
    if name == "evaluate_forecast":
        from buildml.forecasting.evaluate import evaluate_forecast

        return evaluate_forecast
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_forecast_bundle",
        "load_forecast_bundle",
    }:
        from buildml.forecasting import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"forecasting_status", "forecasting_status_for_session"}:
        from buildml.forecasting import explain_hooks as hooks

        return getattr(hooks, name)
    if name == "forecast_capability_matrix":
        from buildml.forecasting.catalog import forecast_capability_matrix

        return forecast_capability_matrix
    raise AttributeError(f"module 'buildml.forecasting' has no attribute {name!r}")
