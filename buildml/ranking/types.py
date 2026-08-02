"""Configuration types for Session-facing learning-to-rank (LTR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RankerMethod = Literal["pointwise", "pairwise"]
PointwiseEstimator = Literal["ridge", "hgb"]
PairwiseEstimator = Literal["ranksvm"]


@dataclass(slots=True)
class RankerConfig:
    """User-facing ranker knobs (serializable summary)."""

    method: RankerMethod = "pointwise"
    query_column: str | None = None
    item_column: str | None = None
    relevance_column: str | None = None
    feature_columns: tuple[str, ...] | None = None
    pointwise_estimator: PointwiseEstimator = "ridge"
    pairwise_estimator: PairwiseEstimator = "ranksvm"
    k: int = 10
    max_pairs_per_query: int = 80
    relevance_threshold: float = 0.0
    random_state: int | None = 0
    alpha: float = 1.0
    C: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "query_column": self.query_column,
            "item_column": self.item_column,
            "relevance_column": self.relevance_column,
            "feature_columns": (
                None if self.feature_columns is None else list(self.feature_columns)
            ),
            "pointwise_estimator": self.pointwise_estimator,
            "pairwise_estimator": self.pairwise_estimator,
            "k": self.k,
            "max_pairs_per_query": self.max_pairs_per_query,
            "relevance_threshold": self.relevance_threshold,
            "random_state": self.random_state,
            "alpha": self.alpha,
            "C": self.C,
        }
