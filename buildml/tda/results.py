"""Typed results for Topological Data Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TdaPlan:
    """Train-fitted TDA transformer (+ optional supervised head).

    Persist via ``buildml.tda_bundle.v2`` (v1 bundles remain loadable).
    """

    vectorization: str
    columns: tuple[str, ...]
    homology_dims: tuple[int, ...]
    knn: int
    maxdim: int
    thresh: float | None
    n_bins: int
    n_layers: int
    n_train_rows: int
    feature_dim: int
    feature_names: tuple[str, ...]
    task: str | None
    head: str
    used_reduce_components: bool
    standardize: bool
    backend: str = "native"
    mean_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)
    train_x_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    nn_: Any = field(default=None, repr=False)
    vectorizer_state_: dict[str, Any] = field(default_factory=dict, repr=False)
    head_estimator_: Any = field(default=None, repr=False)
    label_encoder_: Any = field(default=None, repr=False)
    classes_: tuple[Any, ...] = ()
    train_tda_features_: np.ndarray | None = field(default=None, repr=False)
    mapper_summary_: dict[str, Any] | None = field(default=None, repr=False)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "vectorization": self.vectorization,
            "columns": list(self.columns),
            "homology_dims": list(self.homology_dims),
            "knn": self.knn,
            "maxdim": self.maxdim,
            "thresh": self.thresh,
            "n_bins": self.n_bins,
            "n_layers": self.n_layers,
            "n_train_rows": self.n_train_rows,
            "feature_dim": self.feature_dim,
            "feature_names": list(self.feature_names),
            "task": self.task,
            "head": self.head,
            "used_reduce_components": self.used_reduce_components,
            "standardize": self.standardize,
            "has_head": self.head_estimator_ is not None,
            "mapper_summary": None if self.mapper_summary_ is None else dict(self.mapper_summary_),
            "classes": list(self.classes_),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class TdaFitResult:
    """Outcome of fitting the TDA pipeline on train."""

    vectorization: str
    n_train_rows: int
    feature_dim: int
    homology_dims: tuple[int, ...]
    knn: int
    columns: tuple[str, ...]
    task: str | None
    head: str
    backend: str = "native"
    train_score: float | None = None
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "vectorization": self.vectorization,
            "n_train_rows": self.n_train_rows,
            "feature_dim": self.feature_dim,
            "homology_dims": list(self.homology_dims),
            "knn": self.knn,
            "columns": list(self.columns),
            "task": self.task,
            "head": self.head,
            "train_score": self.train_score,
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"TdaFit · {self.backend} · {self.vectorization} · knn={self.knn} · "
            f"dim={self.feature_dim} · n_train={self.n_train_rows} · head={self.head}"
        )
        if self.train_score is not None:
            print(f"  train_score: {self.train_score:.6f}")
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class TdaTransformResult:
    """Topological feature matrix for one partition."""

    partition: str
    n_rows: int
    feature_dim: int
    feature_names: tuple[str, ...]
    features: np.ndarray = field(repr=False)
    vectorization: str = "persistence_image"
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "feature_dim": self.feature_dim,
            "feature_names": list(self.feature_names),
            "vectorization": self.vectorization,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class TdaPredictResult:
    """Predictions from the optional TDA supervised head."""

    partition: str
    n_rows: int
    task: str
    predictions: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "task": self.task,
            "n_predictions": len(self.predictions),
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class TdaEvalResult:
    """Holdout scores for a head fitted on train topological features."""

    partition: str
    task: str
    n_rows: int
    metrics: dict[str, float] = field(default_factory=dict)
    diagram_distances: dict[str, float] = field(default_factory=dict)
    vectorization: str = "persistence_image"
    backend: str = "native"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "diagram_distances": dict(self.diagram_distances),
            "vectorization": self.vectorization,
            "backend": self.backend,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"TdaEval · {self.vectorization} · task={self.task} · "
            f"partition={self.partition} · n={self.n_rows}"
        )
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
