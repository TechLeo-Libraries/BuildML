"""Configuration types for the multi-task / multi-output Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SklearnMultiTaskMethod = Literal[
    "multi_output",
    "classifier_chain",
    "regressor_chain",
]
IndustryMultiTaskMethod = Literal[
    "multi_output_xgb",
    "multi_output_lgbm",
    "multi_output_catboost",
]
TorchMultiTaskMethod = Literal["shared_trunk_multihead"]

MultiTaskMethod = SklearnMultiTaskMethod | IndustryMultiTaskMethod | TorchMultiTaskMethod

MultiTaskTask = Literal["classification", "regression", "auto", "mixed"]

MultiTaskBackend = Literal["sklearn", "industry", "torch"]

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
    backend: MultiTaskBackend | None = None
    task: MultiTaskTask = "auto"
    targets: tuple[str, ...] | None = None
    columns: tuple[str, ...] | None = None
    base_estimator: MultiTaskBaseEstimator = "logistic_regression"
    random_state: int | None = 0
    order: tuple[str, ...] | None = None
    prefer_reduce_components: bool = True
    prediction_prefix: str = "multitask_pred"
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize user-facing multi-task knobs for history and bundle metadata.

        Captures backend, method, targets, and torch training hyperparameters.

        Returns
        -------
        dict[str, Any]
            Method, task, columns, targets, and training configuration summary.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "task": self.task,
            "targets": None if self.targets is None else list(self.targets),
            "columns": None if self.columns is None else list(self.columns),
            "base_estimator": self.base_estimator,
            "random_state": self.random_state,
            "order": None if self.order is None else list(self.order),
            "prefer_reduce_components": self.prefer_reduce_components,
            "prediction_prefix": self.prediction_prefix,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
        }
