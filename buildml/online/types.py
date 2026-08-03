"""Configuration types for the online / continual Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OnlineTask = Literal["classification", "regression"]
OnlineBackend = Literal["sklearn", "industry", "torch"]

SklearnOnlineEstimator = Literal[
    "sgd_classifier",
    "sgd_regressor",
    "passive_aggressive_classifier",
    "passive_aggressive_regressor",
    "perceptron",
    "multinomial_nb",
    "bernoulli_nb",
]
IndustryOnlineEstimator = Literal[
    "river_logistic",
    "river_hoeffding",
    "river_pa",
    "river_linear_regression",
    "river_hoeffding_regressor",
]
TorchContinualMethod = Literal["replay_mlp", "ewc_mlp"]

OnlineEstimator = SklearnOnlineEstimator | IndustryOnlineEstimator | TorchContinualMethod
OnlineDriftDetector = Literal["mean_shift", "adwin", "page_hinkley", "none"]


@dataclass(slots=True)
class OnlineConfig:
    """User-facing online-learning knobs (serializable summary)."""

    estimator: str = "sgd_classifier"
    backend: OnlineBackend | None = None
    task: OnlineTask = "classification"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    chunk_size: int = 50
    n_init: int | None = None
    classes: tuple[Any, ...] | None = None
    prefer_reduce_components: bool = True
    allow_refit_fallback: bool = False
    drift_disclose: bool = True
    drift_detector: OnlineDriftDetector = "mean_shift"
    buffer_size: int = 512
    epochs_per_update: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    ewc_lambda: float = 100.0
    hidden_dim: int = 64
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize user-facing online config knobs for plan metadata.

        Captures chunk protocol, drift, and continual-learning hyperparameters
        stored on :class:`OnlinePlan.config`.

        Returns
        -------
        dict[str, Any]
            Estimator, backend, chunk protocol, and continual-learning knobs.
        """
        return {
            "estimator": self.estimator,
            "backend": self.backend,
            "task": self.task,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "chunk_size": self.chunk_size,
            "n_init": self.n_init,
            "classes": None if self.classes is None else list(self.classes),
            "prefer_reduce_components": self.prefer_reduce_components,
            "allow_refit_fallback": self.allow_refit_fallback,
            "drift_disclose": self.drift_disclose,
            "drift_detector": self.drift_detector,
            "buffer_size": self.buffer_size,
            "epochs_per_update": self.epochs_per_update,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "ewc_lambda": self.ewc_lambda,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
        }
