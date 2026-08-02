"""Configuration types for the unsupervised Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ClusterMethod = Literal["kmeans", "agglomerative", "dbscan"]
AssignStrategy = Literal["native", "nearest_centroid", "nearest_core"]


@dataclass(slots=True)
class ClusterConfig:
    """User-facing clustering knobs (serializable summary)."""

    method: ClusterMethod = "kmeans"
    n_clusters: int | None = 8
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    # KMeans
    n_init: int | str = "auto"
    max_iter: int = 300
    # Agglomerative
    linkage: str = "ward"
    # DBSCAN
    eps: float = 0.5
    min_samples: int = 5
    # Feature resolution
    prefer_reduce_components: bool = True
    label_column: str = "cluster_id"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "linkage": self.linkage,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "prefer_reduce_components": self.prefer_reduce_components,
            "label_column": self.label_column,
        }
