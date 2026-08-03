"""Orchestrate time-series analysis on train-only (or all) scope."""

from __future__ import annotations

from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.timeseries.changepoints import detect_changepoints
from buildml.timeseries.decompose import decompose_series
from buildml.timeseries.diagnostics import compute_diagnostics
from buildml.timeseries.features import compute_features
from buildml.timeseries.results import TSAnalysisResult
from buildml.timeseries.series import analysis_frame
from buildml.timeseries.types import AnalysisScope, DecomposeMethod, TSAnalysisConfig


def analyze_timeseries(
    dataset: Dataset,
    split_plan: SplitPlan | None,
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
) -> TSAnalysisResult:
    """Run decomposition, diagnostics, changepoints, and features on a temporal split.

    This is descriptive EDA on ordered observations — not a forecast fit. After
    :meth:`~buildml.session.session.Session.time_split`, call with
    ``scope='train'`` to summarize structure before tuning forecast models.
    Each sub-analysis can be toggled off when you only need one view.

    Parameters
    ----------
    dataset:
        Tabular frame with a time column and numeric target.
    split_plan:
        Temporal split from :meth:`~buildml.session.session.Session.time_split`.
        Random splits are refused.
    target_column:
        Column to analyze. Defaults to the dataset's resolved target when omitted.
    time_column:
        Timestamp or sort key column. Defaults to the dataset's time column.
    scope:
        ``train`` analyzes only the train partition (recommended before forecast
        tuning). ``all`` includes validation and test rows for exploratory views.
    seasonal_period:
        Seasonal cycle length for decomposition. Inferred when ``None``.
    decompose_method:
        ``stl``, ``classical``, or ``moving_average``. Defaults to the best
        available backend from :func:`timeseries_capability_matrix`.
    include_decompose, include_diagnostics, include_changepoints, include_features:
        Toggle each analysis block. Convenience wrappers ``ts_decompose`` and
        ``ts_diagnostics`` set these for you.
    acf_lags, pacf_lags:
        Maximum lags for autocorrelation and partial autocorrelation plots.
    adf_regression, kpss_regression:
        Deterministic terms for ADF and KPSS stationarity tests (statsmodels).
    changepoint_method:
        ``pelt``, ``binseg``, or ``cusum``. Defaults to ruptures PELT when installed.
    changepoint_penalty:
        Penalty/threshold for changepoint search (method-specific scaling).
    rolling_window:
        Window size for rolling mean/std features.
    spectral_n_fft:
        FFT length for Welch spectral density. Auto-selected when ``None``.

    Returns
    -------
    TSAnalysisResult
        Combined report with optional decomposition, diagnostics, changepoint, and
        feature sub-results. Call ``show()`` or inspect ``disclosures`` for how
        each backend was chosen.

    Notes
    -----
    **Leakage:** Requires ``time_split``. Default ``scope='train'`` analyzes only
    the train partition. ``scope='all'`` is for EDA — do not use holdout rows to
    tune forecast hyperparameters without disclosure.
    """
    if scope == "train":
        assert_fit_partition(split_plan, "train")

    y, stamps, target_col, time_col = analysis_frame(
        dataset,
        split_plan,
        scope=scope,
        time_column=time_column,
        target_column=target_column,
    )

    from buildml.timeseries.catalog import DEFAULT_CHANGEPOINT, DEFAULT_DECOMPOSE

    decomp_method = decompose_method or DEFAULT_DECOMPOSE  # type: ignore[assignment]
    cp_method = changepoint_method or DEFAULT_CHANGEPOINT

    disclosures: list[str] = [
        f"Time-series analysis scope={scope} on target={target_col}, n={y.shape[0]}.",
        "Random/stratified splits are refused; chronological order enforced.",
        "This is descriptive analysis — not a forecast fit and not a digital twin.",
    ]
    warnings: list[str] = []
    if scope == "all":
        warnings.append(
            "scope='all' includes validation/test rows — use for EDA only, "
            "not for tuning before holdout evaluation."
        )

    decompose = None
    if include_decompose:
        decompose = decompose_series(
            y,
            method=decomp_method,  # type: ignore[arg-type]
            seasonal_period=seasonal_period,
            target_column=target_col,
            time_column=time_col,
            timestamps=stamps,
        )
        warnings.extend(decompose.warnings)

    diagnostics = None
    if include_diagnostics:
        diagnostics = compute_diagnostics(
            y,
            acf_lags=acf_lags,
            pacf_lags=pacf_lags,
            adf_regression=adf_regression,
            kpss_regression=kpss_regression,
            target_column=target_col,
            time_column=time_col,
        )
        warnings.extend(diagnostics.warnings)

    changepoints = None
    if include_changepoints:
        changepoints = detect_changepoints(
            y,
            method=cp_method,  # type: ignore[arg-type]
            penalty=changepoint_penalty,
            target_column=target_col,
        )
        warnings.extend(changepoints.warnings)

    features = None
    if include_features:
        features = compute_features(
            y,
            rolling_window=rolling_window,
            spectral_n_fft=spectral_n_fft,
            target_column=target_col,
        )
        warnings.extend(features.warnings)

    return TSAnalysisResult(
        target_column=target_col,
        time_column=time_col,
        scope=scope,
        n_points=int(y.shape[0]),
        decompose=decompose,
        diagnostics=diagnostics,
        changepoints=changepoints,
        features=features,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def ts_decompose(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    **kwargs: object,
) -> TSAnalysisResult:
    """Run seasonal decomposition only, forwarding kwargs to :func:`analyze_timeseries`.

    Convenience entry when you need trend/seasonal/residual components without
    ACF, changepoints, or rolling features. Accepts the same keyword arguments
    as :func:`analyze_timeseries` (``target_column``, ``scope``, etc.).

    Parameters
    ----------
    dataset:
        Tabular frame with time and target columns.
    split_plan:
        Temporal split plan from Session ``time_split``.
    **kwargs:
        Forwarded to :func:`analyze_timeseries` with decomposition enabled and
        other blocks disabled.

    Returns
    -------
    TSAnalysisResult
        Result with ``decompose`` populated and other blocks omitted.
    """
    return analyze_timeseries(
        dataset,
        split_plan,
        include_decompose=True,
        include_diagnostics=False,
        include_changepoints=False,
        include_features=False,
        **kwargs,  # type: ignore[arg-type]
    )


def ts_diagnostics(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    **kwargs: object,
) -> TSAnalysisResult:
    """Run ACF, PACF, and stationarity diagnostics without other analysis blocks.

    Convenience entry when you only need autocorrelation structure and ADF/KPSS
    tests. Accepts the same keyword arguments as :func:`analyze_timeseries`.

    Parameters
    ----------
    dataset:
        Tabular frame with time and target columns.
    split_plan:
        Temporal split plan from Session ``time_split``.
    **kwargs:
        Forwarded to :func:`analyze_timeseries` with diagnostics enabled and
        decomposition, changepoints, and features disabled.

    Returns
    -------
    TSAnalysisResult
        Result with ``diagnostics`` populated and other blocks omitted.
    """
    return analyze_timeseries(
        dataset,
        split_plan,
        include_decompose=False,
        include_diagnostics=True,
        include_changepoints=False,
        include_features=False,
        **kwargs,  # type: ignore[arg-type]
    )


def config_from_kwargs(**kwargs: object) -> TSAnalysisConfig:
    """Build a :class:`TSAnalysisConfig` from Session or notebook keyword arguments.

    Filters unknown keys so partial kwargs from interactive calls do not raise
    type errors. Used when persisting analysis settings on a Session plan.

    Parameters
    ----------
    **kwargs:
        Any field name accepted by :class:`TSAnalysisConfig`. Unknown keys are
        silently dropped.

    Returns
    -------
    TSAnalysisConfig
        Frozen configuration object for reuse across analyze calls.
    """
    fields = TSAnalysisConfig.__dataclass_fields__
    filtered = {k: v for k, v in kwargs.items() if k in fields}
    return TSAnalysisConfig(**filtered)  # type: ignore[arg-type]
