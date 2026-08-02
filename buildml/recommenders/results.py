"""Typed results for recommendation systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RecommenderPlan:
    """Train-fitted recommender state.

    Persist via ``buildml.recommender_bundle.v1``. Distinct from Session
    checkpoints and from RAG / diagnostic EDA Recommendation objects.
    """

    method: str
    backend: str
    user_column: str
    item_column: str
    rating_column: str | None
    feedback: str
    n_neighbors: int
    n_factors: int
    n_train_interactions: int
    n_users: int
    n_items: int
    user_ids: tuple[Any, ...]
    item_ids: tuple[Any, ...]
    user_index_: dict[Any, int] = field(default_factory=dict, repr=False)
    item_index_: dict[Any, int] = field(default_factory=dict, repr=False)
    # Dense or CSR-compatible float matrix users × items (train only)
    matrix_: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)), repr=False
    )
    # Method-specific fitted state
    similarity_: np.ndarray | None = field(default=None, repr=False)
    user_factors_: np.ndarray | None = field(default=None, repr=False)
    item_factors_: np.ndarray | None = field(default=None, repr=False)
    global_mean_: float = 0.0
    item_popularity_: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False
    )
    item_feature_columns: tuple[str, ...] = ()
    user_feature_columns: tuple[str, ...] = ()
    item_features_: np.ndarray | None = field(default=None, repr=False)
    item_feature_mean_: np.ndarray | None = field(default=None, repr=False)
    item_feature_scale_: np.ndarray | None = field(default=None, repr=False)
    # Industry backend fitted state (implicit ALS/BPR, LightFM)
    backend_model_: Any = field(default=None, repr=False)
    user_item_csr_: Any = field(default=None, repr=False)
    lightfm_user_features_: Any = field(default=None, repr=False)
    lightfm_item_features_: Any = field(default=None, repr=False)
    cold_start: str = "popularity"
    min_rating: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
            "user_column": self.user_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "feedback": self.feedback,
            "n_neighbors": self.n_neighbors,
            "n_factors": self.n_factors,
            "n_train_interactions": self.n_train_interactions,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_user_ids": len(self.user_ids),
            "n_item_ids": len(self.item_ids),
            "item_feature_columns": list(self.item_feature_columns),
            "user_feature_columns": list(self.user_feature_columns),
            "cold_start": self.cold_start,
            "min_rating": self.min_rating,
            "global_mean": self.global_mean_,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RecommenderFitResult:
    """Outcome of fitting a recommender on train interactions."""

    method: str
    backend: str
    n_train_interactions: int
    n_users: int
    n_items: int
    feedback: str
    user_column: str
    item_column: str
    rating_column: str | None = None
    n_neighbors: int | None = None
    n_factors: int | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
            "n_train_interactions": self.n_train_interactions,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "feedback": self.feedback,
            "user_column": self.user_column,
            "item_column": self.item_column,
            "rating_column": self.rating_column,
            "n_neighbors": self.n_neighbors,
            "n_factors": self.n_factors,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"RecommenderFit · {self.method} · backend={self.backend} · "
            f"feedback={self.feedback} · "
            f"users={self.n_users} · items={self.n_items} · "
            f"interactions={self.n_train_interactions}"
        )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class RecommendResult:
    """Top-K item recommendations for one or more users."""

    k: int
    n_users: int
    method: str
    user_ids: tuple[Any, ...]
    # Parallel lists: for each user, recommended item ids and scores
    recommendations: tuple[tuple[Any, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    cold_start_users: tuple[Any, ...] = ()
    excluded_train_items: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_users": self.n_users,
            "method": self.method,
            "n_recommendations": sum(len(r) for r in self.recommendations),
            "n_cold_start_users": len(self.cold_start_users),
            "excluded_train_items": self.excluded_train_items,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"Recommend · {self.method} · k={self.k} · users={self.n_users} · "
            f"cold_start={len(self.cold_start_users)}"
        )


@dataclass(slots=True)
class RecommenderEvalResult:
    """Holdout ranking metrics (known-item protocol)."""

    partition: str
    method: str
    k: int
    n_users_scored: int
    n_cold_start_users: int
    n_holdout_interactions: int
    metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "k": self.k,
            "n_users_scored": self.n_users_scored,
            "n_cold_start_users": self.n_cold_start_users,
            "n_holdout_interactions": self.n_holdout_interactions,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"RecommenderEval · {self.method} · k={self.k} · "
            f"partition={self.partition} · users={self.n_users_scored}"
        )
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
