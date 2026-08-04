"""Optional dependency gates for time-series analysis and forecasting extras.

BuildML's time-series analysis path always works on numpy alone: moving-average
decomposition, lightweight CUSUM changepoints, and numpy ACF. STL decomposition,
ADF/KPSS stationarity tests, and ruptures changepoints require
``buildml[timeseries]`` (statsmodels, scipy, ruptures). Forecasting extras
(Prophet, neuralforecast) live behind separate install hints.

``require_*`` functions raise :class:`~buildml.core.errors.MissingExtraError`
with the install command; ``*_available`` predicates return bool and never raise.
Imports happen inside functions so core import stays fast.

See Also
--------
buildml.timeseries.catalog.timeseries_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_statsmodels(*, feature: str = "time-series analysis") -> Any:
    """Import and return ``statsmodels``, or raise a helpful :class:`MissingExtraError`.

    Call at the point of use when STL decomposition, classical seasonal
    decomposition, or ADF/KPSS diagnostics are requested and the cheap
    ``find_spec`` probe is not enough.

    Parameters
    ----------
    feature:
        Human-readable capability name embedded in the error message so the
        reader knows which operation failed.

    Returns
    -------
    module
        The imported ``statsmodels`` module.

    Raises
    ------
    MissingExtraError
        When ``statsmodels`` is not installed. Install with
        ``pip install 'buildml[timeseries]'``.
    """
    try:
        import statsmodels
    except ImportError as exc:
        raise MissingExtraError("timeseries", feature) from exc
    return statsmodels


def require_ruptures(*, feature: str = "changepoint detection") -> Any:
    """Import and return ``ruptures``, or raise a helpful :class:`MissingExtraError`.

    Call when PELT or binary-segmentation changepoint methods are requested.
    Core CUSUM fallback does not need this gate.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``ruptures`` module.

    Raises
    ------
    MissingExtraError
        When ``ruptures`` is not installed. Install with
        ``pip install 'buildml[timeseries]'``.
    """
    try:
        import ruptures
    except ImportError as exc:
        raise MissingExtraError("timeseries", feature) from exc
    return ruptures


def require_prophet(*, feature: str = "Prophet forecasting") -> Any:
    """Import and return ``prophet.Prophet``, or raise a helpful :class:`MissingExtraError`.

    Used by the forecasting domain, not core time-series analysis. Kept here so
    one extras module covers the temporal stack.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    type
        The ``Prophet`` forecaster class.

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
    """Import and return ``neuralforecast``, or raise a helpful :class:`MissingExtraError`.

    Used by the forecasting domain for deep learning forecasters. Not required
    for descriptive time-series analysis.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``neuralforecast`` module.

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


def ruptures_spec_present() -> bool:
    """Cheap find_spec discovery for ruptures."""
    return importlib.util.find_spec("ruptures") is not None


def prophet_spec_present() -> bool:
    """Cheap find_spec discovery for Prophet."""
    return importlib.util.find_spec("prophet") is not None


def neuralforecast_spec_present() -> bool:
    """Cheap find_spec discovery for neuralforecast."""
    return importlib.util.find_spec("neuralforecast") is not None


def statsmodels_available() -> bool:
    """Return whether statsmodels imports cleanly (subprocess probe)."""
    if not statsmodels_spec_present():
        return False
    return _runtime_ok("statsmodels")


def ruptures_available() -> bool:
    """Return whether ruptures imports cleanly (subprocess probe)."""
    if not ruptures_spec_present():
        return False
    return _runtime_ok("ruptures")


def prophet_available() -> bool:
    """Return whether Prophet imports cleanly (subprocess probe)."""
    if not prophet_spec_present():
        return False
    return _runtime_ok("prophet")


def neuralforecast_available() -> bool:
    """Return whether neuralforecast imports cleanly (subprocess probe)."""
    if not neuralforecast_spec_present():
        return False
    return _runtime_ok("neuralforecast")


def scipy_available() -> bool:
    """Return whether scipy imports cleanly for Welch spectral features."""
    if importlib.util.find_spec("scipy") is None:
        return False
    return _runtime_ok("scipy")


def timeseries_extra_available() -> bool:
    """Return whether the recommended analysis stack imports cleanly."""
    return statsmodels_available()
