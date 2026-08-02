"""Configuration types for the semi-supervised Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SemiSupervisedMethod = Literal[
    "label_propagation",
    "label_spreading",
    "self_training",
]

# sklearn semi-supervised convention: unlabeled == -1
SKLEARN_UNLABELED = -1


@dataclass(slots=True)
class SemiSupervisedConfig:
    """User-facing semi-supervised knobs (serializable summary)."""

    method: SemiSupervisedMethod = "label_propagation"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    # Graph-based
    kernel: str = "knn"
    n_neighbors: int = 7
    max_iter: int = 1000
    alpha: float = 0.2  # LabelSpreading only
    # Self-training
    base_estimator: str = "logistic_regression"
    threshold: float = 0.75
    criterion: str = "threshold"
    k_best: int = 10
    max_self_train_iter: int = 10
    # Label missingness
    unlabeled_marker: Any = None  # None → treat NaN/NA/None as unlabeled
    prefer_reduce_components: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "kernel": self.kernel,
            "n_neighbors": self.n_neighbors,
            "max_iter": self.max_iter,
            "alpha": self.alpha,
            "base_estimator": self.base_estimator,
            "threshold": self.threshold,
            "criterion": self.criterion,
            "k_best": self.k_best,
            "max_self_train_iter": self.max_self_train_iter,
            "unlabeled_marker": self.unlabeled_marker,
            "prefer_reduce_components": self.prefer_reduce_components,
        }
