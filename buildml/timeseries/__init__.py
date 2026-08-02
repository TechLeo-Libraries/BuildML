"""Time-series analysis domain (Phase R3).

Industry-depth coverage:
  - Decomposition: STL / classical (statsmodels default), moving-average fallback
  - Diagnostics: ACF/PACF, ADF, KPSS
  - Changepoints: PELT/binseg (ruptures), CUSUM core fallback
  - Features: rolling stats, Welch spectral density

Dependency policy: core numpy/pandas/sklearn temporal guards. Industry defaults
when ``buildml[timeseries]`` installed (statsmodels, scipy, ruptures).

Lazy imports — core never grows heavy time-series stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TSAnalysisConfig",
    "TSAnalysisResult",
    "TSChangepointResult",
    "TSDecomposeResult",
    "TSDiagnosticsResult",
    "TSFeatureResult",
    "analyze_timeseries",
    "config_from_kwargs",
    "list_changepoint_methods",
    "list_decompose_methods",
    "timeseries_capability_matrix",
    "timeseries_status",
    "timeseries_status_for_session",
    "ts_decompose",
    "ts_diagnostics",
]


def __getattr__(name: str) -> Any:
    if name == "TSAnalysisConfig":
        from buildml.timeseries.types import TSAnalysisConfig

        return TSAnalysisConfig
    if name in {
        "TSAnalysisResult",
        "TSDecomposeResult",
        "TSDiagnosticsResult",
        "TSChangepointResult",
        "TSFeatureResult",
    }:
        from buildml.timeseries import results as results_mod

        return getattr(results_mod, name)
    if name in {"analyze_timeseries", "ts_decompose", "ts_diagnostics", "config_from_kwargs"}:
        from buildml.timeseries import analyze as analyze_mod

        return getattr(analyze_mod, name)
    if name in {
        "list_decompose_methods",
        "list_changepoint_methods",
        "timeseries_capability_matrix",
    }:
        from buildml.timeseries import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {"timeseries_status", "timeseries_status_for_session"}:
        from buildml.timeseries import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.timeseries' has no attribute {name!r}")
