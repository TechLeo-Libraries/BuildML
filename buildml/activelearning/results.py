"""Typed results for active learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ActiveLearningPlan:
    """Train-fitted active-learning plan (model + pool contract + query history).

    Persist via ``buildml.activelearning_bundle.v1``. Distinct from Session
    checkpoints, semi-supervised propagation plans, and self-supervised pretext.
    """

    strategy: str
    backend: str
    base_estimator: str
    columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    n_labeled_train: int
    n_unlabeled_pool: int
    classes_: tuple[Any, ...]
    labeled_train_indices: tuple[Any, ...]
    unlabeled_pool_indices: tuple[Any, ...]
    query_history: tuple[dict[str, Any], ...]
    n_queries_used: int
    label_budget: int | None
    estimator_: Any = field(repr=False)
    label_encoder_: Any = field(repr=False, default=None)
    committee_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-friendly dict (no private estimators).

        Omits ``estimator_``, ``label_encoder_``, and ``committee_`` so bundles
        and history stay lightweight.

        Returns
        -------
        dict[str, Any]
            Pool contract, query history, budget, and disclosure fields.
        """
        return {
            "strategy": self.strategy,
            "backend": self.backend,
            "base_estimator": self.base_estimator,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_labeled_train": self.n_labeled_train,
            "n_unlabeled_pool": self.n_unlabeled_pool,
            "classes": list(self.classes_),
            "n_labeled_indices": len(self.labeled_train_indices),
            "n_pool_indices": len(self.unlabeled_pool_indices),
            "n_query_rounds": len(self.query_history),
            "n_queries_used": self.n_queries_used,
            "label_budget": self.label_budget,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
            "query_history": list(self.query_history),
        }


@dataclass(slots=True)
class ActiveLearningFitResult:
    """Outcome of fitting / refitting the active learner on labeled train rows."""

    strategy: str
    backend: str
    base_estimator: str
    n_train_rows: int
    n_labeled_train: int
    n_unlabeled_pool: int
    n_queries_used: int
    label_budget: int | None
    columns: tuple[str, ...]
    target_column: str
    classes: tuple[Any, ...]
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fit result for history and bundle metadata.

        Captures pool sizes, query budget usage, and strategy summary without
        model weights.

        Returns
        -------
        dict[str, Any]
            Strategy, backend, pool counts, and disclosure fields.
        """
        return {
            "strategy": self.strategy,
            "backend": self.backend,
            "base_estimator": self.base_estimator,
            "n_train_rows": self.n_train_rows,
            "n_labeled_train": self.n_labeled_train,
            "n_unlabeled_pool": self.n_unlabeled_pool,
            "n_queries_used": self.n_queries_used,
            "label_budget": self.label_budget,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "classes": list(self.classes),
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ActiveLearningQueryResult:
    """Batch of train-pool indices suggested for human labeling (no oracle)."""

    strategy: str
    batch_size_requested: int
    indices: tuple[Any, ...]
    scores: tuple[float, ...]
    n_unlabeled_pool: int
    n_queries_used: int
    label_budget: int | None
    budget_remaining: int | None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the query result for history and walkthrough overlays.

        Includes suggested indices and scores for the latest query round.

        Returns
        -------
        dict[str, Any]
            Batch size, indices, scores, pool size, and budget remaining.
        """
        return {
            "strategy": self.strategy,
            "batch_size_requested": self.batch_size_requested,
            "n_suggested": len(self.indices),
            "indices": list(self.indices),
            "scores": list(self.scores),
            "n_unlabeled_pool": self.n_unlabeled_pool,
            "n_queries_used": self.n_queries_used,
            "label_budget": self.label_budget,
            "budget_remaining": self.budget_remaining,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ActiveLearningLabelResult:
    """Outcome of incorporating user-provided labels into the Session target."""

    n_labeled_now: int
    n_newly_labeled: int
    indices: tuple[Any, ...]
    n_queries_used: int
    label_budget: int | None
    budget_remaining: int | None
    refit: bool
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the label result without embedding updated target values.

        Records how many rows were labeled and whether a refit occurred.

        Returns
        -------
        dict[str, Any]
            Newly labeled count, budget state, refit flag, and indices.
        """
        return {
            "n_labeled_now": self.n_labeled_now,
            "n_newly_labeled": self.n_newly_labeled,
            "n_indices": len(self.indices),
            "indices": list(self.indices),
            "n_queries_used": self.n_queries_used,
            "label_budget": self.label_budget,
            "budget_remaining": self.budget_remaining,
            "refit": self.refit,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ActiveLearningEvalResult:
    """Holdout evaluation on labeled rows only (no invented labels for scoring)."""

    partition: str
    strategy: str
    n_rows: int
    n_labeled_eval: int
    n_unlabeled_eval: int
    n_queries_used: int
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation metrics and labeled/unlabeled mix on a partition.

        Metrics reflect labeled rows only; unlabeled holdout rows are counted
        separately for walkthrough honesty.

        Returns
        -------
        dict[str, Any]
            Partition, metrics, query usage, and disclosure fields.
        """
        return {
            "partition": self.partition,
            "strategy": self.strategy,
            "n_rows": self.n_rows,
            "n_labeled_eval": self.n_labeled_eval,
            "n_unlabeled_eval": self.n_unlabeled_eval,
            "n_queries_used": self.n_queries_used,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
