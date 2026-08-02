"""Optional dependency gates for industry forecasting backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_statsmodels(*, feature: str = "statsmodels forecasting") -> Any:
    try:
        import statsmodels
    except ImportError as exc:
        raise MissingExtraError("timeseries", feature) from exc
    return statsmodels


def require_prophet(*, feature: str = "Prophet forecasting") -> Any:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise MissingExtraError("timeseries-prophet", feature) from exc
    return Prophet


def require_neuralforecast(*, feature: str = "neural forecasting") -> Any:
    try:
        import neuralforecast
    except ImportError as exc:
        raise MissingExtraError("timeseries-ml", feature) from exc
    return neuralforecast


def statsmodels_available() -> bool:
    return importlib.util.find_spec("statsmodels") is not None


def prophet_available() -> bool:
    return importlib.util.find_spec("prophet") is not None


def neuralforecast_available() -> bool:
    return importlib.util.find_spec("neuralforecast") is not None


def industry_forecast_available() -> bool:
    return statsmodels_available()
