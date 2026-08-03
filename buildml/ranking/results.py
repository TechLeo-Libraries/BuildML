"""Typed results for tabular learning-to-rank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RankerPlan:
    """Stores train-fitted tabular learning-to-rank state.

    Persist via ``buildml.ranker_bundle.v1``. Distinct from Session checkpoints,
    RAG retrieve/index bundles, and recommender CF bundles.
    """

    method: str
    backend: str = "sklearn"
    query_column: str = ""
    item_column: str = ""
    relevance_column: str = ""
    feature_columns: tuple[str, ...] = ()
    pointwise_estimator: str = "ridge"
    pairwise_estimator: str = "ranksvm"
    n_train_rows: int = 0
    n_train_queries: int = 0
    n_features: int = 0
    feature_mean_: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False
    )
    feature_scale_: np.ndarray = field(
        default_factory=lambda: np.ones(0), repr=False
    )
    estimator_: Any = field(default=None, repr=False)
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0
    max_pairs_per_query: int = 80
    relevance_threshold: float = 0.0
    alpha: float = 1.0
    C: float = 1.0
    random_state: int | None = 0
    n_estimators: int = 120
    learning_rate: float = 0.08
    hidden_dim: int = 64
    epochs: int = 40
    device: str = "cpu"
    group_split_disclosed: bool = False
    split_kind: str | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the ranker plan for bundles and history logs.

        Captures backend, method, column contract, and training metadata
        without embedding full estimator objects or weight arrays.

        Returns
        -------
        dict[str, Any]
            Plan metadata, feature contract, and honesty disclosures.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "query_column": self.query_column,
            "item_column": self.item_column,
            "relevance_column": self.relevance_column,
            "feature_columns": list(self.feature_columns),
            "pointwise_estimator": self.pointwise_estimator,
            "pairwise_estimator": self.pairwise_estimator,
            "n_train_rows": self.n_train_rows,
            "n_train_queries": self.n_train_queries,
            "n_features": self.n_features,
            "max_pairs_per_query": self.max_pairs_per_query,
            "relevance_threshold": self.relevance_threshold,
            "alpha": self.alpha,
            "C": self.C,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "device": self.device,
            "group_split_disclosed": self.group_split_disclosed,
            "split_kind": self.split_kind,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RankerFitResult:
    """Outcome of fitting a ranker on train query–item rows."""

    method: str
    n_train_queries: int
    n_features: int
    query_column: str
    item_column: str
    relevance_column: str
    backend: str = "sklearn"
    n_train_rows: int = 0
    feature_columns: tuple[str, ...] = ()
    pointwise_estimator: str | None = None
    pairwise_estimator: str | None = None
    n_pairwise_examples: int | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise ranker fit output for history logs.

        Records backend, method, train query/row counts, and estimator choices
        after fit on Session train rows completes.

        Returns
        -------
        dict[str, Any]
            Fit metadata, column contract, and honesty disclosures.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "n_train_rows": self.n_train_rows,
            "n_train_queries": self.n_train_queries,
            "n_features": self.n_features,
            "query_column": self.query_column,
            "item_column": self.item_column,
            "relevance_column": self.relevance_column,
            "feature_columns": list(self.feature_columns),
            "pointwise_estimator": self.pointwise_estimator,
            "pairwise_estimator": self.pairwise_estimator,
            "n_pairwise_examples": self.n_pairwise_examples,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact human-readable summary of the ranker fit result."""
        print(
            f"RankerFit · {self.backend}/{self.method} · queries={self.n_train_queries} · "
            f"rows={self.n_train_rows} · features={self.n_features}"
        )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class RankResult:
    """Per-query ranked item lists from a frozen RankerPlan."""

    k: int
    n_queries: int
    method: str
    query_ids: tuple[Any, ...]
    rankings: tuple[tuple[Any, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    n_candidates: tuple[int, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise rank output without listing every ranked item.

        Keeps history payloads compact while recording k, query counts, and
        total ranked items.

        Returns
        -------
        dict[str, Any]
            Rank metadata and honesty disclosures.
        """
        return {
            "k": self.k,
            "n_queries": self.n_queries,
            "method": self.method,
            "n_ranked_items": sum(len(r) for r in self.rankings),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact summary of per-query rank output."""
        print(
            f"Rank · {self.method} · k={self.k} · queries={self.n_queries}"
        )


@dataclass(slots=True)
class RankerEvalResult:
    """Holdout ranking metrics averaged over queries."""

    partition: str
    method: str
    k: int
    n_queries_scored: int
    n_holdout_rows: int
    metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise holdout ranking evaluation metrics.

        Produced by :func:`buildml.ranking.evaluate.evaluate_ranker` after
        macro-averaging nDCG, MAP, and MRR over holdout queries.

        Returns
        -------
        dict[str, Any]
            Partition, metric dictionary, query counts, and disclosures.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "k": self.k,
            "n_queries_scored": self.n_queries_scored,
            "n_holdout_rows": self.n_holdout_rows,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact summary of holdout ranking evaluation metrics."""
        print(
            f"RankerEval · {self.method} · k={self.k} · "
            f"partition={self.partition} · queries={self.n_queries_scored}"
        )
        for key, value in self.metrics.items():
            print(
                f"  {key}: {value:.6f}"
                if isinstance(value, float)
                else f"  {key}: {value}"
            )
