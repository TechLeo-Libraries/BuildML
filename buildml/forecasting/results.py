"""Typed results for classical time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ForecastPlan:
    """Train-fitted forecast plan (estimator/baseline + lag contract).

    Distinct from classical supervised FitResult and from Session checkpoints.
    Persist via ``buildml.forecast_bundle.v1``.
    """

    method: str
    target_column: str
    time_column: str
    horizon: int
    lags: tuple[int, ...]
    seasonal_period: int | None
    exog_columns: tuple[str, ...]
    n_train_rows: int
    n_fit_rows: int
    train_end_stamp: str | None
    estimator_: Any = field(default=None, repr=False)
    baseline_value_: float | None = None
    drift_slope_: float | None = None
    seasonal_history_: tuple[float, ...] = ()
    last_train_values_: tuple[float, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    univariate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "horizon": self.horizon,
            "lags": list(self.lags),
            "seasonal_period": self.seasonal_period,
            "exog_columns": list(self.exog_columns),
            "n_train_rows": self.n_train_rows,
            "n_fit_rows": self.n_fit_rows,
            "train_end_stamp": self.train_end_stamp,
            "baseline_value": self.baseline_value_,
            "drift_slope": self.drift_slope_,
            "n_seasonal_history": len(self.seasonal_history_),
            "n_last_train_values": len(self.last_train_values_),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
            "univariate": self.univariate,
            "has_estimator": self.estimator_ is not None,
        }


@dataclass(slots=True)
class ForecastFitResult:
    """Outcome of fitting a forecaster on the train partition."""

    method: str
    target_column: str
    time_column: str
    n_train_rows: int
    n_fit_rows: int
    horizon: int
    lags: tuple[int, ...]
    univariate: bool
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    train_end_stamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "n_train_rows": self.n_train_rows,
            "n_fit_rows": self.n_fit_rows,
            "horizon": self.horizon,
            "lags": list(self.lags),
            "univariate": self.univariate,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "train_end_stamp": self.train_end_stamp,
        }

    def show(self) -> None:
        mode = "univariate" if self.univariate else "with exogenous"
        print(
            f"ForecastFit · {self.method} · {mode} · "
            f"n_train={self.n_train_rows} · n_fit={self.n_fit_rows} · "
            f"horizon={self.horizon}"
        )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class ForecastGenerateResult:
    """Horizon forecast values generated from a frozen plan."""

    method: str
    horizon: int
    origin: str
    predictions: tuple[float, ...]
    timestamps: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "horizon": self.horizon,
            "origin": self.origin,
            "n_predictions": len(self.predictions),
            "predictions": list(self.predictions),
            "timestamps": list(self.timestamps),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"ForecastGenerate · {self.method} · horizon={self.horizon} · "
            f"origin={self.origin}"
        )
        preview = ", ".join(f"{v:.4g}" for v in self.predictions[:8])
        suffix = "…" if len(self.predictions) > 8 else ""
        print(f"  ŷ: [{preview}{suffix}]")


@dataclass(slots=True)
class ForecastEvalResult:
    """Holdout forecast evaluation with leakage-safe metrics."""

    partition: str
    method: str
    strategy: str
    n_points: int
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: tuple[float, ...] = ()
    actuals: tuple[float, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "strategy": self.strategy,
            "n_points": self.n_points,
            "metrics": dict(self.metrics),
            "n_predictions": len(self.predictions),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        print(
            f"ForecastEval · {self.method} · partition={self.partition} · "
            f"strategy={self.strategy} · n={self.n_points}"
        )
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for tip in self.recommendations[:6]:
            print(f"  - {tip}")
