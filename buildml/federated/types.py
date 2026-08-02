"""Configuration types for Session-facing federated learning simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FederatedBackend = Literal["native", "flower"]

FederatedMethod = Literal["fedavg", "fedprox"]

FederatedTask = Literal["classification", "regression"]

FederatedEstimator = Literal[
    "sgd_classifier",
    "sgd_regressor",
    "logistic_regression",
    "ridge",
    "linear_regression",
]


@dataclass(slots=True)
class FederatedConfig:
    """User-facing federated-learning knobs (serializable summary)."""

    backend: FederatedBackend = "native"
    method: FederatedMethod = "fedavg"
    estimator: FederatedEstimator = "sgd_classifier"
    task: FederatedTask = "classification"
    client_column: str | None = None
    columns: tuple[str, ...] | None = None
    n_rounds: int = 5
    local_epochs: int = 1
    client_fraction: float = 1.0
    mu: float = 0.0
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    min_client_rows: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method": self.method,
            "estimator": self.estimator,
            "task": self.task,
            "client_column": self.client_column,
            "columns": None if self.columns is None else list(self.columns),
            "n_rounds": self.n_rounds,
            "local_epochs": self.local_epochs,
            "client_fraction": self.client_fraction,
            "mu": self.mu,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "min_client_rows": self.min_client_rows,
        }
