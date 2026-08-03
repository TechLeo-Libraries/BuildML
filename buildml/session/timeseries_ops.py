"""Thin Session facades over buildml.timeseries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from buildml.timeseries.analyze import analyze_timeseries
from buildml.timeseries.explain_hooks import analysis_result_summary
from buildml.timeseries.types import AnalysisScope, DecomposeMethod


def analyze_timeseries_op(
    session,
    *,
    target_column: str | None = None,
    time_column: str | None = None,
    scope: AnalysisScope = "train",
    seasonal_period: int | None = None,
    decompose_method: DecomposeMethod | None = None,
    include_decompose: bool = True,
    include_diagnostics: bool = True,
    include_changepoints: bool = True,
    include_features: bool = True,
    acf_lags: int = 40,
    pacf_lags: int = 40,
    adf_regression: str = "c",
    kpss_regression: str = "c",
    changepoint_method: str | None = None,
    changepoint_penalty: float = 10.0,
    rolling_window: int = 7,
    spectral_n_fft: int | None = None,
) -> Any:
    """Run time-series analysis on train-only or full-dataset scope.

    Delegates to :func:`buildml.timeseries.analyze.analyze_timeseries`, stores
    the result on Session, and records the operation. Default scope is
    ``train`` to avoid peeking at holdout data during EDA.

    Parameters
    ----------
    session:
        Active Session with a time-ordered dataset and optional split plan.
    target_column:
        Series to analyze; defaults to the target role column.
    time_column:
        Timestamp or index column; inferred when omitted.
    scope:
        ``train`` (default) restricts to train indices; ``all`` uses full data.
    seasonal_period:
        Seasonal period for decomposition and diagnostics.
    decompose_method:
        Decomposition algorithm (STL, classical, etc.).
    include_decompose:
        When True, run seasonal decomposition.
    include_diagnostics:
        When True, run stationarity and autocorrelation diagnostics.
    include_changepoints:
        When True, detect structural changepoints.
    include_features:
        When True, extract lag/rolling/spectral features.
    acf_lags:
        Maximum lag for autocorrelation function plots.
    pacf_lags:
        Maximum lag for partial autocorrelation function plots.
    adf_regression:
        Regression term for Augmented Dickey-Fuller test.
    kpss_regression:
        Regression term for KPSS stationarity test.
    changepoint_method:
        Changepoint detection algorithm override.
    changepoint_penalty:
        Penalty controlling changepoint count.
    rolling_window:
        Window size for rolling statistics.
    spectral_n_fft:
        FFT size for spectral analysis (``None`` uses series length).

    Returns
    -------
    TimeseriesAnalysisResult
        Decomposition, diagnostics, changepoints, and feature summaries.
        Use :func:`ts_decompose_op` or :func:`ts_diagnostics_op` for focused runs.
    """
    if scope == "train":
        session.assert_can_fit("train")
    result = analyze_timeseries(
        session.dataset,
        session._split_plan,
        target_column=target_column,
        time_column=time_column,
        scope=scope,
        seasonal_period=seasonal_period,
        decompose_method=decompose_method,
        include_decompose=include_decompose,
        include_diagnostics=include_diagnostics,
        include_changepoints=include_changepoints,
        include_features=include_features,
        acf_lags=acf_lags,
        pacf_lags=pacf_lags,
        adf_regression=adf_regression,
        kpss_regression=kpss_regression,
        changepoint_method=changepoint_method,
        changepoint_penalty=changepoint_penalty,
        rolling_window=rolling_window,
        spectral_n_fft=spectral_n_fft,
    )
    session._ts_analysis_result = result
    session._record(
        "analyze_timeseries",
        {
            "scope": scope,
            "target_column": target_column,
            "time_column": time_column,
            "include_decompose": include_decompose,
            "include_diagnostics": include_diagnostics,
        },
        warnings=tuple(result.warnings),
        result_summary=analysis_result_summary(result),
    )
    return result


def ts_decompose_op(session, **kwargs: Any) -> Any:
    """Run decomposition-only time-series analysis on Session data.

    Convenience wrapper around :meth:`analyze_timeseries` that enables
    seasonal decomposition and disables diagnostics, changepoints, and
    feature extraction. Use this when you only need trend/seasonal/residual
    components before choosing a forecast method.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    **kwargs:
        Forwarded to :func:`analyze_timeseries` (for example ``target_column``,
        ``time_column``, ``scope``, ``seasonal_period``, ``decompose_method``).
        Decomposition is forced on; diagnostics/changepoints/features are off.

    Returns
    -------
    TimeseriesAnalysisResult
        Result with decomposition components populated.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("include_decompose", True)
    kwargs.setdefault("include_diagnostics", False)
    kwargs.setdefault("include_changepoints", False)
    kwargs.setdefault("include_features", False)
    if kwargs.get("scope", "train") == "train":
        session.assert_can_fit("train")
    result = analyze_timeseries(session.dataset, session._split_plan, **kwargs)
    session._ts_analysis_result = result
    session._record(
        "ts_decompose",
        {"scope": kwargs.get("scope", "train")},
        warnings=tuple(result.warnings),
        result_summary=analysis_result_summary(result),
    )
    return result


def ts_diagnostics_op(session, **kwargs: Any) -> Any:
    """Run diagnostics-only time-series analysis on Session data.

    Convenience wrapper around :meth:`analyze_timeseries` that runs ACF/PACF
    and ADF/KPSS stationarity tests while skipping decomposition,
    changepoints, and feature extraction.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    **kwargs:
        Forwarded to :func:`analyze_timeseries` (for example ``target_column``,
        ``time_column``, ``scope``, ``acf_lags``, ``pacf_lags``). Diagnostics
        are forced on; decomposition/changepoints/features are off.

    Returns
    -------
    TimeseriesAnalysisResult
        Result with diagnostic tests and ACF/PACF summaries populated.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("include_decompose", False)
    kwargs.setdefault("include_diagnostics", True)
    kwargs.setdefault("include_changepoints", False)
    kwargs.setdefault("include_features", False)
    if kwargs.get("scope", "train") == "train":
        session.assert_can_fit("train")
    result = analyze_timeseries(session.dataset, session._split_plan, **kwargs)
    session._ts_analysis_result = result
    session._record(
        "ts_diagnostics",
        {"scope": kwargs.get("scope", "train")},
        warnings=tuple(result.warnings),
        result_summary=analysis_result_summary(result),
    )
    return result
