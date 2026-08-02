"""Configuration types for Session-facing Topological Data Analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TdaBackend = Literal["native", "giotto"]
Vectorization = Literal[
    "persistence_image",
    "landscape",
    "silhouette",
    "betti_curve",
    "persistence_landscape",
]
SubsampleStrategy = Literal["error", "random", "stratified"]
DiagramDistanceMetric = Literal["wasserstein", "bottleneck"]
TdaTask = Literal["classification", "regression"]
TdaHead = Literal[
    "logistic_regression",
    "random_forest",
    "ridge",
    "hist_gradient_boosting",
    "none",
]


@dataclass(slots=True)
class TdaConfig:
    """User-facing TDA knobs (serializable summary)."""

    backend: TdaBackend = "native"
    vectorization: Vectorization = "persistence_image"
    homology_dims: tuple[int, ...] = (0, 1)
    knn: int = 16
    maxdim: int = 1
    thresh: float | None = None
    n_bins: int = 20
    n_layers: int = 3
    pixel_size: float | None = None
    standardize: bool = True
    head: TdaHead = "logistic_regression"
    task: TdaTask | None = None
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    max_points_guard: int = 4000
    subsample_strategy: SubsampleStrategy = "error"
    mapper: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "vectorization": self.vectorization,
            "homology_dims": list(self.homology_dims),
            "knn": self.knn,
            "maxdim": self.maxdim,
            "thresh": self.thresh,
            "n_bins": self.n_bins,
            "n_layers": self.n_layers,
            "pixel_size": self.pixel_size,
            "standardize": self.standardize,
            "head": self.head,
            "task": self.task,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "max_points_guard": self.max_points_guard,
            "subsample_strategy": self.subsample_strategy,
            "mapper": self.mapper,
        }
