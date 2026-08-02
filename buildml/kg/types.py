"""Configuration types for Session-facing knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

KgBackend = Literal["native", "pykeen"]
KgMethod = Literal["transe", "distmult", "rotate", "complex"]
KgNorm = Literal["l1", "l2"]
LinkPredictionMode = Literal["tail", "head", "relation"]
KgQueryMode = Literal["neighbors", "path", "typed"]
EmbeddingKind = Literal["real", "rotate", "complex"]


@dataclass(slots=True)
class KgConfig:
    """User-facing knowledge-graph knobs (serializable summary)."""

    backend: KgBackend = "native"
    method: KgMethod = "transe"
    head_column: str | None = None
    relation_column: str | None = None
    tail_column: str | None = None
    embedding_dim: int = 50
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 0.01
    margin: float = 1.0
    neg_ratio: int = 1
    norm: KgNorm = "l1"
    random_state: int | None = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method": self.method,
            "head_column": self.head_column,
            "relation_column": self.relation_column,
            "tail_column": self.tail_column,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "margin": self.margin,
            "neg_ratio": self.neg_ratio,
            "norm": self.norm,
            "random_state": self.random_state,
        }
