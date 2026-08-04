# ruff: noqa: E501, F401
"""Time-series analysis Session operation overlays."""

from __future__ import annotations

from buildml.explain.overlays._common import (
    DATASET,
    ROLES,
    SPLIT,
    OperationKind,
    _operation,
    _p,
)
from buildml.explain.schemas import OperationSpec

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "analyze_timeseries",
        OperationKind.DIAGNOSTIC,
        "Run decomposition, diagnostics, changepoints, and spectral features.",
        "Describe temporal structure on train-only scope before forecasting.",
        "Time-series analysis step.",
        (
            "Require temporal SplitPlan; default scope=train.",
            "STL/classical decomposition when statsmodels installed.",
            "ACF/PACF, ADF, KPSS stationarity tests.",
            "Changepoints via ruptures or CUSUM fallback.",
        ),
        parameters=(
            _p("scope", "train | all", "Partition scope for analysis.", "train"),
            _p("decompose_method", "stl | classical | moving_average", "Decomposition algorithm."),
            _p("seasonal_period", "int | None", "Season length for decomposition."),
            _p("include_decompose", "bool", "Run decomposition.", True),
            _p("include_diagnostics", "bool", "Run ACF/PACF/ADF/KPSS.", True),
            _p("include_changepoints", "bool", "Run changepoint detection.", True),
            _p("include_features", "bool", "Rolling stats and spectral features.", True),
        ),
        inputs=("Session with target + time roles and time_split.",),
        outputs=("TSAnalysisResult stored on Session.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After time_split; before session.forecast.fit for honest EDA.",),
        alternatives=("session.timeseries.decompose or session.timeseries.diagnostics for focused outputs.",),
        rationale=("Use to inspect seasonality and stationarity before choosing a forecaster.",),
        assumptions=("Chronological order; numeric target.",),
        failures=("Random split refused; null targets; series too short.",),
        leakage=("scope=all includes holdout: do not tune on it without disclosure.",),
        anti_patterns=("Using holdout decomposition to pick forecast hyperparameters silently.",),
        state_changes=("Stores session.timeseries.analysis_result.",),
        result_reading=("Read decompose, diagnostics, changepoints, features sub-results.",),
        next_steps=("session.forecast.fit with method informed by diagnostics.",),
        concepts=(
            "ts-analysis-before-forecast",
            "ts-decomposition",
            "forecast-temporal-leakage",
            "leakage-boundary",
        ),
    ),
    _operation(
        "ts_decompose",
        OperationKind.DIAGNOSTIC,
        "STL or classical seasonal decomposition (train-only default).",
        "Isolate trend and seasonal components.",
        "Decomposition-only step.",
        ("Wraps session.timeseries.analyze with decomposition enabled only.",),
        parameters=(
            _p("scope", "train | all", "Analysis scope.", "train"),
            _p("decompose_method", "stl | classical | moving_average", "Algorithm."),
            _p("seasonal_period", "int | None", "Season length."),
        ),
        inputs=("Temporal Session.",),
        outputs=("TSAnalysisResult with decompose populated.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After time_split.",),
        alternatives=("session.timeseries.analyze for full report.",),
        rationale=("Use when only trend/seasonal/residual views are needed.",),
        assumptions=("Numeric target.",),
        failures=("Missing statsmodels for stl without fallback acceptance.",),
        leakage=(
            "scope='train' is the honest default; scope='all' decomposes holdout rows too, and anything you learn from that shape has informed you.",
        ),
        anti_patterns=(
            "Reading a scope='all' decomposition and then choosing a seasonal period for the forecaster.",
        ),
        state_changes=("Stores session.timeseries.analysis_result.",),
        result_reading=("Inspect trend, seasonal, residual tuples.",),
        next_steps=("session.timeseries.diagnostics or session.forecast.fit.",),
        concepts=("forecast-temporal-leakage",),
    ),
    _operation(
        "ts_diagnostics",
        OperationKind.DIAGNOSTIC,
        "ACF/PACF and ADF/KPSS stationarity tests.",
        "Quantify autocorrelation and unit-root behavior.",
        "Diagnostics-only step.",
        ("Wraps session.timeseries.analyze with diagnostics enabled only.",),
        parameters=(
            _p("acf_lags", "int", "ACF lag count.", 40),
            _p("pacf_lags", "int", "PACF lag count.", 40),
        ),
        inputs=("Temporal Session.",),
        outputs=("TSAnalysisResult with diagnostics populated.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After time_split.",),
        alternatives=("session.timeseries.analyze for full report.",),
        rationale=("Use to choose ARIMA orders or differencing.",),
        assumptions=("Numeric target.",),
        failures=("ADF/KPSS unavailable without buildml[timeseries].",),
        leakage=(
            "scope='train' is the honest default; running diagnostics on holdout tells you about rows you are supposed to be measured against.",
        ),
        anti_patterns=(
            "Choosing differencing orders from a scope='all' stationarity test and then reporting test error as untouched.",
        ),
        state_changes=("Stores session.timeseries.analysis_result.",),
        result_reading=("Read adf_pvalue, kpss_pvalue, acf_values.",),
        next_steps=("session.forecast.fit(method='arima'| 'ets'| 'auto').",),
        concepts=("forecast-temporal-leakage",),
    ),
)
