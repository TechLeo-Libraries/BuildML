"""Typed results for practical Session-facing meta-learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MetaLearningPlan:
    """Meta-trained few-shot / episodic learner + feature/task contract.

    Persist via ``buildml.metalearning_bundle.v1``. Distinct from Session
    checkpoints and from multi-task / classical FitResult. This is tabular
    few-shot / episodic meta-learning — not foundation-model MAML-at-scale.
    """

    backend: str
    method: str
    columns: tuple[str, ...]
    target_column: str
    task_column: str
    train_task_ids: tuple[Any, ...]
    held_out_task_ids: tuple[Any, ...]
    classes_: tuple[Any, ...]
    n_train_rows: int
    n_way: int
    k_shot: int
    n_query: int
    n_episodes: int
    meta_train_accuracy: float | None
    label_encoder_: Any = field(repr=False)
    init_estimator_: Any = field(repr=False, default=None)
    meta_learner_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-friendly dict (no private estimators).

        Omits ``label_encoder_``, ``init_estimator_``, and ``meta_learner_`` so
        bundles and history stay lightweight.

        Returns
        -------
        dict[str, Any]
            Episodic protocol, task ids, and disclosure fields for history/bundles.
        """
        return {
            "backend": self.backend,
            "method": self.method,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "task_column": self.task_column,
            "train_task_ids": list(self.train_task_ids),
            "held_out_task_ids": list(self.held_out_task_ids),
            "n_meta_train_tasks": len(self.train_task_ids),
            "n_held_out_tasks": len(self.held_out_task_ids),
            "classes": list(self.classes_),
            "n_train_rows": self.n_train_rows,
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_query": self.n_query,
            "n_episodes": self.n_episodes,
            "meta_train_accuracy": self.meta_train_accuracy,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class MetaLearningFitResult:
    """Outcome of a train-only meta-learning fit."""

    backend: str
    method: str
    n_train_rows: int
    columns: tuple[str, ...]
    target_column: str
    task_column: str
    n_meta_train_tasks: int
    n_held_out_tasks: int
    n_way: int
    k_shot: int
    n_episodes: int
    meta_train_accuracy: float | None
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fit result for history and bundle metadata.

        Captures episodic protocol knobs and meta-train accuracy without model
        weights.

        Returns
        -------
        dict[str, Any]
            Backend, method, episodic knobs, and meta-train accuracy summary.
        """
        return {
            "backend": self.backend,
            "method": self.method,
            "n_train_rows": self.n_train_rows,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "task_column": self.task_column,
            "n_meta_train_tasks": self.n_meta_train_tasks,
            "n_held_out_tasks": self.n_held_out_tasks,
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_episodes": self.n_episodes,
            "meta_train_accuracy": self.meta_train_accuracy,
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class MetaAdaptResult:
    """Fast adaptation of a meta-learner to one task's support set."""

    method: str
    task_id: Any
    n_support: int
    n_classes_adapted: int
    classes_: tuple[Any, ...]
    prototypes_: dict[Any, tuple[float, ...]] | None
    adapted_estimator_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the adapt result without embedding adapted model weights.

        Prototype vectors are summarized by dimension only.

        Returns
        -------
        dict[str, Any]
            Method, task id, support size, and adaptation summary fields.
        """
        proto_summary = None
        if self.prototypes_ is not None:
            proto_summary = {
                str(k): {"dim": len(v)} for k, v in self.prototypes_.items()
            }
        return {
            "method": self.method,
            "task_id": self.task_id,
            "n_support": self.n_support,
            "n_classes_adapted": self.n_classes_adapted,
            "classes": list(self.classes_),
            "prototypes": proto_summary,
            "has_adapted_estimator": self.adapted_estimator_ is not None,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class MetaLearningEvalResult:
    """Episodic holdout evaluation across tasks (never for meta-train)."""

    partition: str
    method: str
    n_tasks_evaluated: int
    n_query_rows: int
    metrics: dict[str, float]
    per_task_metrics: dict[str, dict[str, float]]
    novel_task_ids: tuple[Any, ...]
    overlapping_task_ids: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation metrics and per-task episodic scores.

        Includes novel vs overlapping task id lists for walkthrough honesty.

        Returns
        -------
        dict[str, Any]
            Partition, aggregate metrics, novel/overlapping task ids, and disclosures.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "n_tasks_evaluated": self.n_tasks_evaluated,
            "n_query_rows": self.n_query_rows,
            "metrics": dict(self.metrics),
            "per_task_metrics": {
                str(k): dict(v) for k, v in self.per_task_metrics.items()
            },
            "novel_task_ids": list(self.novel_task_ids),
            "overlapping_task_ids": list(self.overlapping_task_ids),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
