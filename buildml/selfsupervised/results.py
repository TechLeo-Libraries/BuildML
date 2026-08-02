"""Typed results for self-supervised learning hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SelfSupervisedPlan:
    """Train-fitted self-supervised pretext plan (encoder + feature contract).

    Persist via ``buildml.selfsupervised_bundle.v1``. Distinct from Session
    checkpoints, Torch trainer bundles, semi-supervised plans, and zoo
    pretrained backbones (vision/audio/speech transfer hooks).
    """

    method: str
    columns: tuple[str, ...]
    n_train_rows: int
    latent_dim: int
    representation_prefix: str
    representation_columns: tuple[str, ...]
    encoder_: Any = field(repr=False)
    reconstruction_mae_: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "columns": list(self.columns),
            "n_train_rows": self.n_train_rows,
            "latent_dim": self.latent_dim,
            "representation_prefix": self.representation_prefix,
            "representation_columns": list(self.representation_columns),
            "reconstruction_mae": self.reconstruction_mae_,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class SelfSupervisedFitResult:
    """Outcome of fitting a self-supervised pretext on the train partition."""

    method: str
    n_train_rows: int
    columns: tuple[str, ...]
    latent_dim: int
    reconstruction_mae: float | None
    representation_columns: tuple[str, ...]
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_train_rows": self.n_train_rows,
            "columns": list(self.columns),
            "latent_dim": self.latent_dim,
            "reconstruction_mae": self.reconstruction_mae,
            "representation_columns": list(self.representation_columns),
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SelfSupervisedTransformResult:
    """Exported representations from a frozen SSL plan (no refit)."""

    partition: str
    n_rows: int
    method: str
    representation_columns: tuple[str, ...]
    attached: bool = False
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "method": self.method,
            "representation_columns": list(self.representation_columns),
            "attached": self.attached,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class SSLHeadPlan:
    """Supervised head fitted on SSL representations (labeled train only)."""

    estimator_name: str
    target_column: str
    representation_columns: tuple[str, ...]
    n_labeled_train: int
    n_unlabeled_skipped: int
    classes_: tuple[Any, ...]
    estimator_: Any = field(repr=False)
    task: str = "classification"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_name": self.estimator_name,
            "target_column": self.target_column,
            "representation_columns": list(self.representation_columns),
            "n_labeled_train": self.n_labeled_train,
            "n_unlabeled_skipped": self.n_unlabeled_skipped,
            "classes": list(self.classes_),
            "task": self.task,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SSLHeadFitResult:
    """Outcome of attaching/fitting a supervised head on SSL embeddings."""

    estimator_name: str
    n_labeled_train: int
    n_unlabeled_skipped: int
    representation_columns: tuple[str, ...]
    target_column: str
    classes: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_name": self.estimator_name,
            "n_labeled_train": self.n_labeled_train,
            "n_unlabeled_skipped": self.n_unlabeled_skipped,
            "representation_columns": list(self.representation_columns),
            "target_column": self.target_column,
            "classes": list(self.classes),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SelfSupervisedEvalResult:
    """Holdout evaluation of an SSL head on labeled partition rows."""

    partition: str
    n_rows: int
    n_labeled_eval: int
    n_unlabeled_eval: int
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "n_labeled_eval": self.n_labeled_eval,
            "n_unlabeled_eval": self.n_unlabeled_eval,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
