"""Typed results for semi-supervised learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemiSupervisedPlan:
    """Train-fitted semi-supervised plan (estimator + label contract).

    Persist via ``buildml.semisupervised_bundle.v1``. Distinct from Session
    checkpoints, classical ``FitResult``, anomaly novelty (normal-only), and
    self-supervised pretext plans.
    """

    method: str
    columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    n_labeled_train: int
    n_unlabeled_train: int
    classes_: tuple[Any, ...]
    backend: str = "sklearn"
    modality: str = "tabular"
    estimator_: Any = field(default=None, repr=False)
    label_encoder_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-friendly dict for history and bundles.

Omits private estimator and encoder fields so bundles and history records stay lightweight while preserving teaching disclosures.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "modality": self.modality,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_labeled_train": self.n_labeled_train,
            "n_unlabeled_train": self.n_unlabeled_train,
            "classes": list(self.classes_),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class SemiSupervisedFitResult:
    """Outcome of fitting a semi-supervised estimator on the train partition."""

    method: str
    n_train_rows: int
    n_labeled_train: int
    n_unlabeled_train: int
    columns: tuple[str, ...]
    target_column: str
    classes: tuple[Any, ...]
    backend: str = "sklearn"
    modality: str = "tabular"
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-friendly dict for history and bundles.

Omits private estimator and encoder fields so bundles and history records stay lightweight while preserving teaching disclosures.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "modality": self.modality,
            "n_train_rows": self.n_train_rows,
            "n_labeled_train": self.n_labeled_train,
            "n_unlabeled_train": self.n_unlabeled_train,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "classes": list(self.classes),
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SemiSupervisedPredictResult:
    """Predictions from a frozen semi-supervised plan (no refit)."""

    partition: str
    n_rows: int
    predictions: tuple[Any, ...]
    method: str
    attached: bool = False
    prediction_column: str = "semisupervised_prediction"
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-friendly dict for history and bundles.

Omits private estimator and encoder fields so bundles and history records stay lightweight while preserving teaching disclosures.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
        """
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "method": self.method,
            "attached": self.attached,
            "prediction_column": self.prediction_column,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class SemiSupervisedEvalResult:
    """Holdout evaluation on labeled rows only (no invented labels for scoring)."""

    partition: str
    method: str
    n_rows: int
    n_labeled_eval: int
    n_unlabeled_eval: int
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-friendly dict for history and bundles.

Omits private estimator and encoder fields so bundles and history records stay lightweight while preserving teaching disclosures.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "n_rows": self.n_rows,
            "n_labeled_eval": self.n_labeled_eval,
            "n_unlabeled_eval": self.n_unlabeled_eval,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
