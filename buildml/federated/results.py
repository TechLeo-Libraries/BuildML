"""Typed results for Session-facing federated learning simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FederatedPlan:
    """Global federated model + client partition contract + round history.

    Persist via ``buildml.federated_bundle.v1``. Distinct from Session
    checkpoints. This is a **local FedAvg-style simulation** on partitioned
    Session data — not a distributed FL network stack (Flower/OpenFL), and
    not cryptographic secure aggregation.
    """

    method: str
    estimator_name: str
    task: str
    columns: tuple[str, ...]
    target_column: str
    client_column: str
    client_ids: tuple[Any, ...]
    n_train_rows: int
    n_rounds: int
    local_epochs: int
    client_fraction: float
    mu: float
    classes_: tuple[Any, ...] | None
    round_history: tuple[dict[str, Any], ...]
    estimator_: Any = field(repr=False)
    label_encoder_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "client_column": self.client_column,
            "client_ids": list(self.client_ids),
            "n_clients": len(self.client_ids),
            "n_train_rows": self.n_train_rows,
            "n_rounds": self.n_rounds,
            "local_epochs": self.local_epochs,
            "client_fraction": self.client_fraction,
            "mu": self.mu,
            "classes": None if self.classes_ is None else list(self.classes_),
            "n_rounds_completed": len(self.round_history),
            "round_history": [dict(r) for r in self.round_history],
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class FederatedFitResult:
    """Outcome of a train-only federated simulation fit."""

    method: str
    estimator_name: str
    task: str
    n_train_rows: int
    n_clients: int
    n_rounds: int
    local_epochs: int
    client_column: str
    columns: tuple[str, ...]
    target_column: str
    final_train_metric: float | None
    round_history: tuple[dict[str, Any], ...]
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_train_rows": self.n_train_rows,
            "n_clients": self.n_clients,
            "n_rounds": self.n_rounds,
            "local_epochs": self.local_epochs,
            "client_column": self.client_column,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "final_train_metric": self.final_train_metric,
            "n_rounds_completed": len(self.round_history),
            "round_history": [dict(r) for r in self.round_history],
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class FederatedEvalResult:
    """Holdout evaluation of the global federated model (never for training)."""

    partition: str
    method: str
    estimator_name: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    per_client_metrics: dict[str, dict[str, float]]
    n_clients_evaluated: int
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "per_client_metrics": {
                str(k): dict(v) for k, v in self.per_client_metrics.items()
            },
            "n_clients_evaluated": self.n_clients_evaluated,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class FederatedPredictResult:
    """Predictions from the global federated model (no update)."""

    partition: str
    method: str
    estimator_name: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
