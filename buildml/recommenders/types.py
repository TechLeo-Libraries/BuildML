"""Configuration types for Session-facing recommendation systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RecommenderMethod = Literal[
    "item_knn",
    "user_knn",
    "svd",
    "nmf",
    "content",
]
FeedbackMode = Literal["explicit", "implicit"]
ColdStartPolicy = Literal["skip", "popularity"]


@dataclass(slots=True)
class RecommenderConfig:
    """User-facing recommender knobs (serializable summary)."""

    method: RecommenderMethod = "item_knn"
    user_column: str | None = None
    item_column: str | None = None
    rating_column: str | None = None
    feedback: FeedbackMode = "explicit"
    n_neighbors: int = 40
    n_factors: int = 32
    k: int = 10
    min_rating: float | None = None
    item_feature_columns: tuple[str, ...] | None = None
    cold_start: ColdStartPolicy = "popularity"
    random_state: int | None = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "user_column": self.user_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "feedback": self.feedback,
            "n_neighbors": self.n_neighbors,
            "n_factors": self.n_factors,
            "k": self.k,
            "min_rating": self.min_rating,
            "item_feature_columns": (
                None
                if self.item_feature_columns is None
                else list(self.item_feature_columns)
            ),
            "cold_start": self.cold_start,
            "random_state": self.random_state,
        }
