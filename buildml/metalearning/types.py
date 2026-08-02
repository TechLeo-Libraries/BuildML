"""Configuration types for Session-facing meta-learning (few-shot / episodic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MetaLearningMethod = Literal["prototypical", "warm_start"]

MetaLearningBaseEstimator = Literal[
    "logistic_regression",
    "sgd_classifier",
]


@dataclass(slots=True)
class MetaLearningConfig:
    """User-facing meta-learning knobs (serializable summary)."""

    method: MetaLearningMethod = "prototypical"
    task_column: str | None = None
    columns: tuple[str, ...] | None = None
    n_way: int | None = None
    k_shot: int = 5
    n_query: int = 10
    n_episodes: int = 20
    base_estimator: MetaLearningBaseEstimator = "logistic_regression"
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    task_holdout_fraction: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task_column": self.task_column,
            "columns": None if self.columns is None else list(self.columns),
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_query": self.n_query,
            "n_episodes": self.n_episodes,
            "base_estimator": self.base_estimator,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "task_holdout_fraction": self.task_holdout_fraction,
        }
