"""Typed results for multi-task / multi-output learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MultiTaskPlan:
    """Fitted multi-target estimator + feature/target contract.

    Persist via ``buildml.multitask_bundle.v1``. Distinct from Session
    checkpoints and from classical single-target ``FitResult``. Classical
    ``Session.fit`` still requires exactly one target.
    """

    method: str
    task: str
    columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    n_train_rows: int
    classes_per_task_: dict[str, tuple[Any, ...]]
    estimator_: Any = field(repr=False)
    label_encoders_: dict[str, Any] = field(repr=False, default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "columns": list(self.columns),
            "target_columns": list(self.target_columns),
            "n_tasks": len(self.target_columns),
            "n_train_rows": self.n_train_rows,
            "classes_per_task": {
                k: list(v) for k, v in self.classes_per_task_.items()
            },
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class MultiTaskFitResult:
    """Outcome of a train-only multi-task fit."""

    method: str
    task: str
    n_train_rows: int
    columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    n_tasks: int
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "n_train_rows": self.n_train_rows,
            "columns": list(self.columns),
            "target_columns": list(self.target_columns),
            "n_tasks": self.n_tasks,
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class MultiTaskPredictResult:
    """Per-task predictions from a frozen multi-task plan."""

    partition: str
    n_rows: int
    method: str
    task: str
    target_columns: tuple[str, ...]
    predictions: dict[str, tuple[Any, ...]]
    attached: bool = False
    prediction_prefix: str = "multitask_pred"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "method": self.method,
            "task": self.task,
            "target_columns": list(self.target_columns),
            "n_tasks": len(self.target_columns),
            "n_predictions_per_task": {
                k: len(v) for k, v in self.predictions.items()
            },
            "attached": self.attached,
            "prediction_prefix": self.prediction_prefix,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class MultiTaskEvalResult:
    """Holdout evaluation with per-task and aggregate metrics."""

    partition: str
    method: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    per_task_metrics: dict[str, dict[str, float]]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "per_task_metrics": {
                k: dict(v) for k, v in self.per_task_metrics.items()
            },
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
