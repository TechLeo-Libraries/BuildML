"""Optional dependency gates for industry forecasting backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_statsmodels(*, feature: str = "statsmodels forecasting") -> Any:
    """Import and return ``statsmodels``, or raise :class:`MissingExtraError`.

    Called by industry backend fitters at runtime so missing
    ``buildml[timeseries]`` extras surface as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported statsmodels module.

    Raises
    ------
    MissingExtraError
        When statsmodels is not installed. Install with
        ``pip install 'buildml[timeseries]'``.
    """
    try:
        import statsmodels
    except ImportError as exc:
        raise MissingExtraError("timeseries", feature) from exc
    return statsmodels


def require_prophet(*, feature: str = "Prophet forecasting") -> Any:
    """Import and return the Prophet class, or raise :class:`MissingExtraError`.

    Called by the Prophet backend when ``buildml[timeseries-prophet]`` is
    required but not installed.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    type
        The imported ``prophet.Prophet`` class.

    Raises
    ------
    MissingExtraError
        When Prophet is not installed. Install with
        ``pip install 'buildml[timeseries-prophet]'``.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise MissingExtraError("timeseries-prophet", feature) from exc
    return Prophet


def require_neuralforecast(*, feature: str = "neural forecasting") -> Any:
    """Import and return ``neuralforecast``, or raise :class:`MissingExtraError`.

    Called by the N-BEATS backend when ``buildml[timeseries-ml]`` is required
    but not installed.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported neuralforecast module.

    Raises
    ------
    MissingExtraError
        When neuralforecast is not installed. Install with
        ``pip install 'buildml[timeseries-ml]'``.
    """
    try:
        import neuralforecast
    except ImportError as exc:
        raise MissingExtraError("timeseries-ml", feature) from exc
    return neuralforecast


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def statsmodels_spec_present() -> bool:
    """Cheap find_spec discovery for statsmodels."""
    return importlib.util.find_spec("statsmodels") is not None


def prophet_spec_present() -> bool:
    """Cheap find_spec discovery for Prophet."""
    return importlib.util.find_spec("prophet") is not None


def neuralforecast_spec_present() -> bool:
    """Cheap find_spec discovery for neuralforecast."""
    return importlib.util.find_spec("neuralforecast") is not None


def statsmodels_available() -> bool:
    """Return whether statsmodels imports cleanly for industry forecast paths."""
    if not statsmodels_spec_present():
        return False
    return _runtime_ok("statsmodels")


def prophet_available() -> bool:
    """Return whether Prophet imports cleanly for the Prophet backend path."""
    if not prophet_spec_present():
        return False
    return _runtime_ok("prophet")


def neuralforecast_available() -> bool:
    """Return whether neuralforecast imports cleanly for N-BEATS paths."""
    if not neuralforecast_spec_present():
        return False
    return _runtime_ok("neuralforecast")


def industry_forecast_available() -> bool:
    """Return whether any industry statsmodels forecast backend is usable."""
    return statsmodels_available()
