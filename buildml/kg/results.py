"""Typed results for knowledge-graph learning and query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class KgPlan:
    """Stores train-fitted knowledge-graph embeddings and adjacency state.

    Persist via ``buildml.kg_bundle.v1``. Distinct from Session checkpoints,
    Graph ML (node classification), and RAG retrieve/generate.
    """

    method: str
    head_column: str
    relation_column: str
    tail_column: str
    embedding_dim: int
    n_train_triples: int
    n_entities: int
    n_relations: int
    entity_ids: tuple[Any, ...]
    relation_ids: tuple[Any, ...]
    backend: str = "native"
    embedding_kind: str = "real"
    entity_index_: dict[Any, int] = field(default_factory=dict, repr=False)
    relation_index_: dict[Any, int] = field(default_factory=dict, repr=False)
    # Train triple index arrays (entity/relation integer ids)
    train_heads_: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False
    )
    train_relations_: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False
    )
    train_tails_: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False
    )
    entity_embeddings_: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)), repr=False
    )
    relation_embeddings_: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)), repr=False
    )
    # Known true triples for filtered ranking (train catalog)
    true_triple_set_: frozenset[tuple[int, int, int]] = field(
        default_factory=frozenset, repr=False
    )
    # Adjacency for symbolic query (train graph only)
    out_edges_: dict[int, list[tuple[int, int]]] = field(
        default_factory=dict, repr=False
    )
    in_edges_: dict[int, list[tuple[int, int]]] = field(
        default_factory=dict, repr=False
    )
    epochs_run: int = 0
    final_loss: float | None = None
    neg_ratio: int = 1
    norm: str = "l1"
    margin: float = 1.0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the KG plan for bundles and history logs.

        Captures backend, method, vocabulary sizes, and training metadata
        without embedding full embedding weight arrays.

        Returns
        -------
        dict[str, Any]
            Plan metadata, column contract, and honesty disclosures.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "embedding_kind": self.embedding_kind,
            "head_column": self.head_column,
            "relation_column": self.relation_column,
            "tail_column": self.tail_column,
            "embedding_dim": self.embedding_dim,
            "n_train_triples": self.n_train_triples,
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
            "n_entity_ids": len(self.entity_ids),
            "n_relation_ids": len(self.relation_ids),
            "epochs_run": self.epochs_run,
            "final_loss": self.final_loss,
            "neg_ratio": self.neg_ratio,
            "norm": self.norm,
            "margin": self.margin,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class KgFitResult:
    """Outcome of fitting a KG embedding model on train triples."""

    method: str
    n_train_triples: int
    n_entities: int
    n_relations: int
    embedding_dim: int
    head_column: str
    relation_column: str
    tail_column: str
    backend: str = "native"
    epochs_run: int = 0
    final_loss: float | None = None
    neg_ratio: int = 1
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise KG fit output for history logs.

        Records backend, method, vocabulary sizes, and training loss after fit
        on Session train triples completes.

        Returns
        -------
        dict[str, Any]
            Fit metadata, triple counts, and honesty disclosures.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "n_train_triples": self.n_train_triples,
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
            "embedding_dim": self.embedding_dim,
            "head_column": self.head_column,
            "relation_column": self.relation_column,
            "tail_column": self.tail_column,
            "epochs_run": self.epochs_run,
            "final_loss": self.final_loss,
            "neg_ratio": self.neg_ratio,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact human-readable summary of the KG fit result."""
        loss = "n/a" if self.final_loss is None else f"{self.final_loss:.6f}"
        print(
            f"KgFit · {self.backend}/{self.method} · entities={self.n_entities} · "
            f"relations={self.n_relations} · triples={self.n_train_triples} · "
            f"dim={self.embedding_dim} · loss={loss}"
        )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class ScoreTriplesResult:
    """Scores for an explicit list of (head, relation, tail) triples."""

    method: str
    n_triples: int
    scores: tuple[float, ...]
    heads: tuple[Any, ...]
    relations: tuple[Any, ...]
    tails: tuple[Any, ...]
    unknown_entities: int = 0
    unknown_relations: int = 0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise triple-scoring output without listing every score.

        Keeps history payloads compact while recording OOV entity/relation counts.

        Returns
        -------
        dict[str, Any]
            Method, triple count, unknown counts, and disclosures.
        """
        return {
            "method": self.method,
            "n_triples": self.n_triples,
            "unknown_entities": self.unknown_entities,
            "unknown_relations": self.unknown_relations,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact summary of triple-scoring output."""
        print(f"ScoreTriples · {self.method} · n={self.n_triples}")


@dataclass(slots=True)
class PredictLinksResult:
    """Top-K link predictions for incomplete triples."""

    mode: str
    method: str
    k: int
    n_queries: int
    # Parallel lists: for each query, predicted entity/relation ids + scores
    predictions: tuple[tuple[Any, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    query_heads: tuple[Any, ...] = ()
    query_relations: tuple[Any, ...] = ()
    query_tails: tuple[Any, ...] = ()
    filtered: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise link-prediction output without listing every candidate.

        Records mode, k, query counts, and filtered-ranking flag for history logs.

        Returns
        -------
        dict[str, Any]
            Prediction metadata and disclosures.
        """
        return {
            "mode": self.mode,
            "method": self.method,
            "k": self.k,
            "n_queries": self.n_queries,
            "n_predictions": sum(len(p) for p in self.predictions),
            "filtered": self.filtered,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact summary of link-prediction output."""
        print(
            f"PredictLinks · {self.method} · mode={self.mode} · "
            f"k={self.k} · queries={self.n_queries}"
        )


@dataclass(slots=True)
class KgQueryResult:
    """Symbolic query result over the train triple graph (not LLM)."""

    mode: str
    n_results: int
    # neighbors/typed: list of (neighbor, relation) or entity ids
    # path: list of (entity, relation) steps, or empty if none
    results: tuple[Any, ...]
    source: Any = None
    target: Any = None
    relation: Any = None
    max_hops: int | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise symbolic query output for history logs.

        Records mode, endpoints, hop limit, and result count without embedding
        full neighbor or path lists.

        Returns
        -------
        dict[str, Any]
            Query metadata and disclosures.
        """
        return {
            "mode": self.mode,
            "n_results": self.n_results,
            "source": self.source,
            "target": self.target,
            "relation": None if self.relation is None else str(self.relation),
            "max_hops": self.max_hops,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact summary of symbolic query output."""
        print(f"KgQuery · mode={self.mode} · n={self.n_results}")


@dataclass(slots=True)
class KgEvalResult:
    """Holdout link-prediction metrics (filtered ranking protocol)."""

    partition: str
    method: str
    k: int
    n_triples_scored: int
    n_skipped_unknown: int
    metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise holdout link-prediction evaluation metrics.

        Produced by :func:`buildml.kg.evaluate.evaluate_kg` after filtered
        ranking on a validation or test partition.

        Returns
        -------
        dict[str, Any]
            Partition, MRR/Hits metrics, OOV skip counts, and disclosures.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "k": self.k,
            "n_triples_scored": self.n_triples_scored,
            "n_skipped_unknown": self.n_skipped_unknown,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact summary of holdout evaluation metrics."""
        print(
            f"KgEval · {self.method} · k={self.k} · "
            f"partition={self.partition} · triples={self.n_triples_scored}"
        )
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
