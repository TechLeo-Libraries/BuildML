"""Thin Session facades over buildml.timeseries."""

from __future__ import annotations

from typing import Any

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
    """Run time-series analysis on train-only (default) or all scope."""
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
    """Decomposition-only Session entry."""
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
    """Diagnostics-only Session entry."""
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
