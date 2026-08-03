"""Configuration types for time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DecomposeMethod = Literal["stl", "classical", "moving_average"]
ChangepointMethod = Literal["pelt", "binseg", "cusum"]
AnalysisScope = Literal["train", "all"]


@dataclass(slots=True)
class TSAnalysisConfig:
    """User-facing knobs for time-series analysis."""

    target_column: str | None = None
    time_column: str | None = None
    scope: AnalysisScope = "train"
    seasonal_period: int | None = None
    decompose_method: DecomposeMethod = "stl"
    acf_lags: int = 40
    pacf_lags: int = 40
    adf_regression: str = "c"
    kpss_regression: str = "c"
    changepoint_method: ChangepointMethod = "pelt"
    changepoint_penalty: float = 10.0
    rolling_window: int = 7
    spectral_n_fft: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise analysis configuration knobs for plans and history records.

        Every dataclass field is copied into a plain dict suitable for JSON
        export and Session plan persistence.

        Returns
        -------
        dict[str, Any]
            Plain mapping of every :class:`TSAnalysisConfig` field.
        """
        return {
            "target_column": self.target_column,
            "time_column": self.time_column,
            "scope": self.scope,
            "seasonal_period": self.seasonal_period,
            "decompose_method": self.decompose_method,
            "acf_lags": self.acf_lags,
            "pacf_lags": self.pacf_lags,
            "adf_regression": self.adf_regression,
            "kpss_regression": self.kpss_regression,
            "changepoint_method": self.changepoint_method,
            "changepoint_penalty": self.changepoint_penalty,
            "rolling_window": self.rolling_window,
            "spectral_n_fft": self.spectral_n_fft,
        }
