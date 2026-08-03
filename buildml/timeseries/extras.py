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


def statsmodels_available() -> bool:
    """Return whether ``statsmodels`` is importable in the active environment.

    Cheap ``find_spec`` probe used by catalog defaults and capability matrices.
    Never raises: missing packages return ``False``.

    Returns
    -------
    bool
        ``True`` when statsmodels can be imported.
    """
    return importlib.util.find_spec("statsmodels") is not None


def ruptures_available() -> bool:
    """Return whether ``ruptures`` is importable in the active environment.

    Cheap ``find_spec`` probe for PELT/BinSeg changepoint defaults. Never raises.

    Returns
    -------
    bool
        ``True`` when ruptures can be imported.
    """
    return importlib.util.find_spec("ruptures") is not None


def prophet_available() -> bool:
    """Return whether ``prophet`` is importable in the active environment.

    Used by forecasting extras, not core analysis. Never raises.

    Returns
    -------
    bool
        ``True`` when Prophet can be imported.
    """
    return importlib.util.find_spec("prophet") is not None


def neuralforecast_available() -> bool:
    """Return whether ``neuralforecast`` is importable in the active environment.

    Used by forecasting extras for deep learning forecasters. Never raises.

    Returns
    -------
    bool
        ``True`` when neuralforecast can be imported.
    """
    return importlib.util.find_spec("neuralforecast") is not None


def scipy_available() -> bool:
    """Return whether ``scipy`` is importable for Welch spectral features.

    Spectral features in :func:`compute_features` degrade gracefully when
    ``False``. Never raises.

    Returns
    -------
    bool
        ``True`` when scipy can be imported.
    """
    return importlib.util.find_spec("scipy") is not None


def timeseries_extra_available() -> bool:
    """Return whether the recommended analysis stack (statsmodels) is importable.

    This is the predicate :func:`timeseries_capability_matrix` uses to decide
    whether STL/ADF defaults are safe on this machine.

    Returns
    -------
    bool
        ``True`` when ``statsmodels`` can be imported; ``False`` otherwise.
    """
    return statsmodels_available()
