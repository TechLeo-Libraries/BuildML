"""Configuration types for the forecasting Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ForecastMethod = Literal[
    "naive",
    "seasonal_naive",
    "drift",
    "mean",
    "lag_ridge",
    "lag_hgb",
]
ForecastEvalStrategy = Literal["rolling_one_step", "origin"]


@dataclass(slots=True)
class ForecastConfig:
    """User-facing forecasting knobs (serializable summary)."""

    method: ForecastMethod = "lag_ridge"
    horizon: int = 1
    lags: tuple[int, ...] = (1, 2, 3, 7)
    seasonal_period: int | None = None
    exog_columns: tuple[str, ...] = ()
    target_column: str | None = None
    time_column: str | None = None
    random_state: int | None = 0
    # lag_ridge
    alpha: float = 1.0
    # lag_hgb
    max_iter: int = 100
    max_depth: int | None = 3
    learning_rate: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "horizon": self.horizon,
            "lags": list(self.lags),
            "seasonal_period": self.seasonal_period,
            "exog_columns": list(self.exog_columns),
            "target_column": self.target_column,
            "time_column": self.time_column,
            "random_state": self.random_state,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
