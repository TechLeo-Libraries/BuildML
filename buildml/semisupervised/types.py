"""Configuration types for the semi-supervised Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SklearnSemiSupervisedMethod = Literal[
    "label_propagation",
    "label_spreading",
    "self_training",
]
IndustrySemiSupervisedMethod = Literal["pseudo_label_xgb", "pseudo_label_lgbm"]
TorchSemiSupervisedMethod = Literal["fixmatch_tabular", "mixmatch_tabular"]
HFSemiSupervisedMethod = Literal["text_pseudo_label"]

SemiSupervisedMethod = (
    SklearnSemiSupervisedMethod
    | IndustrySemiSupervisedMethod
    | TorchSemiSupervisedMethod
    | HFSemiSupervisedMethod
)
SemiSupervisedBackend = Literal["sklearn", "industry", "torch", "hf"]

# sklearn semi-supervised convention: unlabeled == -1
SKLEARN_UNLABELED = -1


@dataclass(slots=True)
class SemiSupervisedConfig:
    """User-facing semi-supervised knobs (serializable summary)."""

    method: SemiSupervisedMethod = "label_propagation"
    backend: SemiSupervisedBackend | None = None
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    # Graph-based (sklearn)
    kernel: str = "knn"
    n_neighbors: int = 7
    max_iter: int = 1000
    alpha: float = 0.2  # LabelSpreading only
    # Self-training / pseudo-label
    base_estimator: str = "logistic_regression"
    threshold: float = 0.75
    criterion: str = "threshold"
    k_best: int = 10
    max_self_train_iter: int = 10
    # Torch consistency
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    consistency_weight: float = 1.0
    mixup_alpha: float = 0.75
    device: str = "cpu"
    # HF text
    text_column: str | None = None
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Label missingness
    unlabeled_marker: Any = None  # None → treat NaN/NA/None as unlabeled
    prefer_reduce_components: bool = True
    modality: str = "tabular"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
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
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "consistency_weight": self.consistency_weight,
            "mixup_alpha": self.mixup_alpha,
            "device": self.device,
            "text_column": self.text_column,
            "text_model_name": self.text_model_name,
            "unlabeled_marker": self.unlabeled_marker,
            "prefer_reduce_components": self.prefer_reduce_components,
            "modality": self.modality,
        }
