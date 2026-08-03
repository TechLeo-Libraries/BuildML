"""Typed results for time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TSDecomposeResult:
    """Seasonal decomposition output (trend / seasonal / residual)."""

    method: str
    target_column: str
    time_column: str
    n_points: int
    seasonal_period: int | None
    trend: tuple[float, ...]
    seasonal: tuple[float, ...]
    residual: tuple[float, ...]
    observed: tuple[float, ...]
    timestamps: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise decomposition output as a JSON-safe mapping for history logs.

        Omits full trend/seasonal/residual vectors so Session history stays small
        while preserving method, scope metadata, and disclosure strings.

        Returns
        -------
        dict[str, Any]
            Method, column names, point count, seasonal period, and disclosure
            strings: not the full component vectors.
        """
        return {
            "method": self.method,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "n_points": self.n_points,
            "seasonal_period": self.seasonal_period,
            "n_trend": len(self.trend),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a one-line summary and the first few disclosure bullets."""
        print(
            f"TSDecompose · {self.method} · n={self.n_points} · "
            f"period={self.seasonal_period}"
        )
        for tip in self.disclosures[:5]:
            print(f"  · {tip}")


@dataclass(slots=True)
class TSDiagnosticsResult:
    """ACF/PACF and stationarity test outputs."""

    target_column: str
    time_column: str
    n_points: int
    acf_lags: int
    pacf_lags: int
    acf_values: tuple[float, ...]
    acf_confint: tuple[tuple[float, float], ...] = ()
    pacf_values: tuple[float, ...] = ()
    pacf_confint: tuple[tuple[float, float], ...] = ()
    adf_statistic: float | None = None
    adf_pvalue: float | None = None
    adf_used_lags: int | None = None
    adf_critical_values: dict[str, float] = field(default_factory=dict)
    kpss_statistic: float | None = None
    kpss_pvalue: float | None = None
    kpss_critical_values: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise diagnostics output as a JSON-safe mapping for history logs.

        Keeps test statistics and lag counts but drops full ACF/PACF vectors for
        compact audit trails.

        Returns
        -------
        dict[str, Any]
            Lag counts, ADF/KPSS statistics when computed, and disclosure strings.
        """
        return {
            "target_column": self.target_column,
            "time_column": self.time_column,
            "n_points": self.n_points,
            "acf_lags": self.acf_lags,
            "pacf_lags": self.pacf_lags,
            "adf_statistic": self.adf_statistic,
            "adf_pvalue": self.adf_pvalue,
            "kpss_statistic": self.kpss_statistic,
            "kpss_pvalue": self.kpss_pvalue,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print lag counts and stationarity test p-values when available."""
        print(f"TSDiagnostics · n={self.n_points} · acf_lags={self.acf_lags}")
        if self.adf_pvalue is not None:
            print(f"  ADF p={self.adf_pvalue:.4g}")
        if self.kpss_pvalue is not None:
            print(f"  KPSS p={self.kpss_pvalue:.4g}")


@dataclass(slots=True)
class TSChangepointResult:
    """Detected changepoint indices and segments."""

    method: str
    target_column: str
    n_points: int
    changepoint_indices: tuple[int, ...]
    n_segments: int
    segment_means: tuple[float, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise changepoint output as a JSON-safe mapping for history logs.

        Records indices and segment counts without embedding per-segment means in
        history payloads.

        Returns
        -------
        dict[str, Any]
            Method, indices, segment count, and disclosure strings.
        """
        return {
            "method": self.method,
            "target_column": self.target_column,
            "n_points": self.n_points,
            "changepoint_indices": list(self.changepoint_indices),
            "n_segments": self.n_segments,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class TSFeatureResult:
    """Rolling statistics and spectral summary features."""

    target_column: str
    n_points: int
    rolling_window: int
    rolling_mean: tuple[float, ...]
    rolling_std: tuple[float, ...]
    spectral_frequencies: tuple[float, ...] = ()
    spectral_power: tuple[float, ...] = ()
    dominant_period: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise feature output as a JSON-safe mapping for history logs.

        Keeps rolling window metadata and dominant period estimate without full
        rolling or spectral vectors.

        Returns
        -------
        dict[str, Any]
            Rolling window, dominant period estimate, and disclosure strings.
        """
        return {
            "target_column": self.target_column,
            "n_points": self.n_points,
            "rolling_window": self.rolling_window,
            "dominant_period": self.dominant_period,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class TSAnalysisResult:
    """Combined analysis report from analyze_timeseries."""

    target_column: str
    time_column: str
    scope: str
    n_points: int
    decompose: TSDecomposeResult | None = None
    diagnostics: TSDiagnosticsResult | None = None
    changepoints: TSChangepointResult | None = None
    features: TSFeatureResult | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the combined analysis as a JSON-safe mapping for history logs.

        Flags which sub-blocks ran and preserves top-level disclosures without
        nesting full component payloads.

        Returns
        -------
        dict[str, Any]
            Scope, column names, point count, flags for which blocks ran, and
            top-level disclosures: not nested component vectors.
        """
        return {
            "target_column": self.target_column,
            "time_column": self.time_column,
            "scope": self.scope,
            "n_points": self.n_points,
            "has_decompose": self.decompose is not None,
            "has_diagnostics": self.diagnostics is not None,
            "has_changepoints": self.changepoints is not None,
            "has_features": self.features is not None,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print the analysis header and delegate to nested result ``show`` methods."""
        print(
            f"TSAnalysis · {self.target_column} · scope={self.scope} · n={self.n_points}"
        )
        if self.decompose is not None:
            self.decompose.show()
        if self.diagnostics is not None:
            self.diagnostics.show()
