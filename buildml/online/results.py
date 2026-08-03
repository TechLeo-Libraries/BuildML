"""Typed results for online / continual learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OnlinePlan:
    """Incremental model state + update ledger (partial_fit family).

    Persist via ``buildml.online_bundle.v1``. Distinct from Session checkpoints
    and from full-batch classical FitResult / active-learning / semi-supervised
    plans. This is batch/stream-chunk updates on Session data: not a
    distributed streaming platform.
    """

    estimator_name: str
    task: str
    columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    n_seen_rows: int
    n_updates: int
    cursor: int
    chunk_size: int
    classes_: tuple[Any, ...] | None
    seen_train_indices: tuple[Any, ...]
    update_history: tuple[dict[str, Any], ...]
    estimator_: Any = field(repr=False)
    label_encoder_: Any = field(repr=False, default=None)
    init_feature_means_: tuple[float, ...] | None = field(default=None, repr=False)
    used_refit_fallback: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    backend: str = "sklearn"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-friendly dict (no private estimators).

        Omits ``estimator_`` and ``label_encoder_`` so bundles and history stay
        lightweight.

        Returns
        -------
        dict[str, Any]
            Incremental state, cursor, update history, and disclosure fields.
        """
        return {
            "estimator_name": self.estimator_name,
            "backend": self.backend,
            "task": self.task,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_seen_rows": self.n_seen_rows,
            "n_updates": self.n_updates,
            "cursor": self.cursor,
            "chunk_size": self.chunk_size,
            "classes": None if self.classes_ is None else list(self.classes_),
            "n_seen_indices": len(self.seen_train_indices),
            "n_update_rounds": len(self.update_history),
            "used_refit_fallback": self.used_refit_fallback,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
            "update_history": list(self.update_history),
        }


@dataclass(slots=True)
class OnlineFitResult:
    """Outcome of the initial warm-start fit on the first train chunk."""

    estimator_name: str
    task: str
    n_init_rows: int
    n_train_rows: int
    n_remaining_train: int
    columns: tuple[str, ...]
    target_column: str
    classes: tuple[Any, ...] | None
    used_reduce_components: bool = False
    used_refit_fallback: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    backend: str = "sklearn"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fit result for history and bundle metadata.

        Includes init chunk counters and disclosure fields only; omits private
        estimator objects.

        Returns
        -------
        dict[str, Any]
            Init chunk size, remaining train rows, and disclosure fields.
        """
        return {
            "estimator_name": self.estimator_name,
            "backend": self.backend,
            "task": self.task,
            "n_init_rows": self.n_init_rows,
            "n_train_rows": self.n_train_rows,
            "n_remaining_train": self.n_remaining_train,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "classes": None if self.classes is None else list(self.classes),
            "used_reduce_components": self.used_reduce_components,
            "used_refit_fallback": self.used_refit_fallback,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class OnlineUpdateResult:
    """Outcome of one incremental ``partial_fit`` update on a train chunk."""

    estimator_name: str
    task: str
    n_chunk_rows: int
    n_seen_rows: int
    n_updates: int
    n_remaining_train: int
    update_mode: str
    drift_notes: tuple[str, ...] = ()
    used_refit_fallback: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the update result for history and explain overlays.

        Includes chunk counters, drift notes, and update mode without embedding
        the full OnlinePlan.

        Returns
        -------
        dict[str, Any]
            Chunk size, cumulative counters, drift notes, and update mode.
        """
        return {
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_chunk_rows": self.n_chunk_rows,
            "n_seen_rows": self.n_seen_rows,
            "n_updates": self.n_updates,
            "n_remaining_train": self.n_remaining_train,
            "update_mode": self.update_mode,
            "drift_notes": list(self.drift_notes),
            "used_refit_fallback": self.used_refit_fallback,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class OnlineEvalResult:
    """Holdout evaluation after incremental updates (never used for updates)."""

    partition: str
    estimator_name: str
    task: str
    n_rows: int
    n_seen_rows: int
    n_updates: int
    metrics: dict[str, float]
    drift_detected: bool = False
    drift_notes: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation result for history and explain overlays.

        Includes holdout metrics and drift flags without raw predictions.

        Returns
        -------
        dict[str, Any]
            Holdout partition, metrics, drift flags, and disclosure fields.
        """
        return {
            "partition": self.partition,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_seen_rows": self.n_seen_rows,
            "n_updates": self.n_updates,
            "metrics": dict(self.metrics),
            "drift_detected": self.drift_detected,
            "drift_notes": list(self.drift_notes),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class OnlinePredictResult:
    """Predictions from the incremental online estimator."""

    partition: str
    estimator_name: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the predict result for history (predictions omitted).

        Records partition and prediction count so history stays lightweight.

        Returns
        -------
        dict[str, Any]
            Partition, row count, prediction count, and disclosure fields.
        """
        return {
            "partition": self.partition,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
