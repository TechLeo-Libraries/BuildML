"""Optional dependency gates for time-series analysis and forecasting extras."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_statsmodels(*, feature: str = "time-series analysis") -> Any:
    """Import and return ``statsmodels``, or raise :class:`MissingExtraError`."""
    try:
        import statsmodels
    except ImportError as exc:
        raise MissingExtraError("timeseries", feature) from exc
    return statsmodels


def require_ruptures(*, feature: str = "changepoint detection") -> Any:
    """Import and return ``ruptures``, or raise :class:`MissingExtraError`."""
    try:
        import ruptures
    except ImportError as exc:
        raise MissingExtraError("timeseries", feature) from exc
    return ruptures


def require_prophet(*, feature: str = "Prophet forecasting") -> Any:
    """Import and return ``prophet``, or raise :class:`MissingExtraError`."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise MissingExtraError("timeseries-prophet", feature) from exc
    return Prophet


def require_neuralforecast(*, feature: str = "neural forecasting") -> Any:
    """Import and return ``neuralforecast``, or raise :class:`MissingExtraError`."""
    try:
        import neuralforecast
    except ImportError as exc:
        raise MissingExtraError("timeseries-ml", feature) from exc
    return neuralforecast


def statsmodels_available() -> bool:
    return importlib.util.find_spec("statsmodels") is not None


def ruptures_available() -> bool:
    return importlib.util.find_spec("ruptures") is not None


def prophet_available() -> bool:
    return importlib.util.find_spec("prophet") is not None


def neuralforecast_available() -> bool:
    return importlib.util.find_spec("neuralforecast") is not None


def scipy_available() -> bool:
    return importlib.util.find_spec("scipy") is not None


def timeseries_extra_available() -> bool:
    """True when the recommended analysis stack (statsmodels) is importable."""
    return statsmodels_available()
