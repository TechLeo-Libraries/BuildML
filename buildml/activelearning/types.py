"""Configuration types for the active-learning Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ActiveLearningStrategy = Literal[
    "least_confidence",
    "margin",
    "entropy",
    "committee",
    "expected_model_change_lite",
]

ActiveLearningEstimator = Literal[
    "logistic_regression",
    "hist_gradient_boosting",
]


@dataclass(slots=True)
class ActiveLearningConfig:
    """User-facing active-learning knobs (serializable summary)."""

    strategy: ActiveLearningStrategy = "margin"
    base_estimator: ActiveLearningEstimator = "logistic_regression"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    batch_size: int = 5
    label_budget: int | None = 50
    unlabeled_marker: Any = None  # None → treat NaN/NA/None as unlabeled pool
    prefer_reduce_components: bool = True
    committee_size: int = 5
    auto_refit: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "base_estimator": self.base_estimator,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "label_budget": self.label_budget,
            "unlabeled_marker": self.unlabeled_marker,
            "prefer_reduce_components": self.prefer_reduce_components,
            "committee_size": self.committee_size,
            "auto_refit": self.auto_refit,
        }
