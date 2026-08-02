"""Configuration types for Session-facing Bayesian / probabilistic ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProbabilisticTask = Literal["classification", "regression"]

ProbabilisticEstimator = Literal[
    "bayesian_ridge",
    "gaussian_process_regressor",
    "gaussian_process_classifier",
    "gaussian_nb",
]

IntervalMethod = Literal["posterior_std", "split_conformal", "both", "none"]


@dataclass(slots=True)
class ProbabilisticConfig:
    """User-facing probabilistic / Bayesian knobs (serializable summary)."""

    estimator: ProbabilisticEstimator = "bayesian_ridge"
    task: ProbabilisticTask = "regression"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    alpha: float = 0.1
    conformal: bool = True
    conformal_calibration_fraction: float = 0.2
    interval_method: IntervalMethod = "both"
    prefer_reduce_components: bool = True
    n_restarts_optimizer: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "task": self.task,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "alpha": self.alpha,
            "conformal": self.conformal,
            "conformal_calibration_fraction": self.conformal_calibration_fraction,
            "interval_method": self.interval_method,
            "prefer_reduce_components": self.prefer_reduce_components,
            "n_restarts_optimizer": self.n_restarts_optimizer,
        }
