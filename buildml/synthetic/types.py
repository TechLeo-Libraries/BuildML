"""Configuration types for Session-facing synthetic-data systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SynthesizerMethod = Literal[
    "bootstrap",
    "gaussian_copula",
    "smote",
    "ctgan",
    "tvae",
    "copulagan",
]
SyntheticBackend = Literal["native", "sdv"]
ColumnKind = Literal["continuous", "categorical", "integer"]
EvalMode = Literal["fidelity", "tstr"]
EvalBackend = Literal["builtin", "sdmetrics", "auto"]
MergeMode = Literal["none", "extend_train"]
FitPartition = Literal["train"]


@dataclass(slots=True)
class SynthesizerConfig:
    """User-facing synthesizer knobs (serializable summary)."""

    method: SynthesizerMethod = "gaussian_copula"
    backend: SyntheticBackend | None = None
    partition: FitPartition = "train"
    columns: list[str] | None = None
    random_state: int = 42
    # Bootstrap
    smooth_sigma: float = 0.0
    # Gaussian copula
    correlation_ridge: float = 1e-3
    # SMOTE (requires buildml[imbalanced])
    target_column: str | None = None
    k_neighbors: int = 5
    sampling_strategy: str | float | dict[str, float] = "auto"
    # SDV (requires buildml[synthetic-industry])
    epochs: int = 300
    batch_size: int = 500

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
            "partition": self.partition,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "smooth_sigma": self.smooth_sigma,
            "correlation_ridge": self.correlation_ridge,
            "target_column": self.target_column,
            "k_neighbors": self.k_neighbors,
            "sampling_strategy": self.sampling_strategy,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }


@dataclass(slots=True)
class ColumnSchemaSpec:
    """Per-column kind + metadata fitted on train only."""

    name: str
    kind: ColumnKind
    n_unique: int = 0
    n_null: int = 0
    categories: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_unique": self.n_unique,
            "n_null": self.n_null,
            "categories": list(self.categories),
            "extras": dict(self.extras),
        }
