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
    """Run decomposition, diagnostics, changepoints, and feature extraction.

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
    """Decomposition-only entry (wraps analyze_timeseries)."""
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
    """Diagnostics-only entry (ACF/PACF/ADF/KPSS)."""
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
    """Build TSAnalysisConfig from keyword arguments."""
    fields = TSAnalysisConfig.__dataclass_fields__
    filtered = {k: v for k, v in kwargs.items() if k in fields}
    return TSAnalysisConfig(**filtered)  # type: ignore[arg-type]
