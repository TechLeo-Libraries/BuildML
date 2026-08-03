"""Configuration types for Session-facing Bayesian / probabilistic ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProbabilisticTask = Literal["classification", "regression"]

ProbabilisticBackend = Literal["native", "mapie", "ngboost"]

ProbabilisticEstimator = Literal[
    "bayesian_ridge",
    "gaussian_process_regressor",
    "gaussian_process_classifier",
    "gaussian_nb",
    "split",
    "cv_plus",
    "jackknife_plus",
    "mapie_split",
    "mapie_cv_plus",
    "mapie_jackknife_plus",
    "ngboost_regressor",
    "ngboost_classifier",
]

IntervalMethod = Literal[
    "posterior_std",
    "split_conformal",
    "both",
    "none",
    "mapie",
    "mapie_cv_plus",
    "mapie_jackknife_plus",
]

MapieConformalMethod = Literal["split", "cv_plus", "jackknife_plus"]


@dataclass(slots=True)
class ProbabilisticConfig:
    """User-facing probabilistic / Bayesian knobs (serializable summary)."""

    backend: ProbabilisticBackend = "native"
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
    n_estimators: int = 100
    learning_rate: float = 0.05
    mapie_method: MapieConformalMethod | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise probabilistic configuration knobs for plans and history.

        Snapshots backend, estimator, conformal, and interval settings used
        when :func:`buildml.probabilistic.fit.fit_probabilistic` runs.

        Returns
        -------
        dict[str, Any]
            Plain mapping of every :class:`ProbabilisticConfig` field.
        """
        return {
            "backend": self.backend,
            "estimator": self.estimator,
            "task": self.task,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "alpha": float(self.alpha),
            "conformal": self.conformal,
            "conformal_calibration_fraction": self.conformal_calibration_fraction,
            "interval_method": self.interval_method,
            "prefer_reduce_components": self.prefer_reduce_components,
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "mapie_method": self.mapie_method,
        }
