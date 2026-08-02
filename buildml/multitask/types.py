"""Configuration types for the multi-task / multi-output Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MultiTaskMethod = Literal[
    "multi_output",
    "classifier_chain",
    "regressor_chain",
]

MultiTaskTask = Literal["classification", "regression", "auto"]

MultiTaskBaseEstimator = Literal[
    "logistic_regression",
    "hist_gradient_boosting",
    "ridge",
    "hist_gradient_boosting_regressor",
]


@dataclass(slots=True)
class MultiTaskConfig:
    """User-facing multi-task knobs (serializable summary)."""

    method: MultiTaskMethod = "multi_output"
    task: MultiTaskTask = "auto"
    targets: tuple[str, ...] | None = None
    columns: tuple[str, ...] | None = None
    base_estimator: MultiTaskBaseEstimator = "logistic_regression"
    random_state: int | None = 0
    order: tuple[str, ...] | None = None
    prefer_reduce_components: bool = True
    prediction_prefix: str = "multitask_pred"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "targets": None if self.targets is None else list(self.targets),
            "columns": None if self.columns is None else list(self.columns),
            "base_estimator": self.base_estimator,
            "random_state": self.random_state,
            "order": None if self.order is None else list(self.order),
            "prefer_reduce_components": self.prefer_reduce_components,
            "prediction_prefix": self.prediction_prefix,
        }
