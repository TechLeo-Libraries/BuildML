"""Configuration types for the online / continual Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OnlineTask = Literal["classification", "regression"]

OnlineEstimator = Literal[
    "sgd_classifier",
    "sgd_regressor",
    "passive_aggressive_classifier",
    "passive_aggressive_regressor",
    "perceptron",
    "multinomial_nb",
    "bernoulli_nb",
]


@dataclass(slots=True)
class OnlineConfig:
    """User-facing online-learning knobs (serializable summary)."""

    estimator: OnlineEstimator = "sgd_classifier"
    task: OnlineTask = "classification"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    chunk_size: int = 50
    n_init: int | None = None
    classes: tuple[Any, ...] | None = None
    prefer_reduce_components: bool = True
    allow_refit_fallback: bool = False
    drift_disclose: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "task": self.task,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "chunk_size": self.chunk_size,
            "n_init": self.n_init,
            "classes": None if self.classes is None else list(self.classes),
            "prefer_reduce_components": self.prefer_reduce_components,
            "allow_refit_fallback": self.allow_refit_fallback,
            "drift_disclose": self.drift_disclose,
        }
