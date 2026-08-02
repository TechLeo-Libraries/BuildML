"""Typed results for unsupervised clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ClusterPlan:
    """Train-fitted clustering plan (estimator + feature contract).

    Distinct from classical preprocess plans (``ReducePlan``) and from Session
    checkpoints. Persist via ``buildml.unsupervised_bundle.v1``.
    """

    method: str
    columns: tuple[str, ...]
    label_column: str
    n_clusters: int | None
    n_train_rows: int
    train_labels_: tuple[int, ...]
    cluster_sizes_: dict[int, int]
    assign_strategy: str
    estimator_: Any = field(repr=False)
    centroids_: np.ndarray | None = field(default=None, repr=False)
    centroid_label_ids_: tuple[int, ...] = ()
    core_sample_indices_: tuple[int, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "columns": list(self.columns),
            "label_column": self.label_column,
            "n_clusters": self.n_clusters,
            "n_train_rows": self.n_train_rows,
            "n_labels_train": len(self.train_labels_),
            "cluster_sizes": {str(k): int(v) for k, v in self.cluster_sizes_.items()},
            "assign_strategy": self.assign_strategy,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
            "has_centroids": self.centroids_ is not None,
            "centroid_label_ids": list(self.centroid_label_ids_),
            "n_core_samples": len(self.core_sample_indices_),
        }


@dataclass(slots=True)
class ClusterFitResult:
    """Outcome of fitting a clusterer on the train partition."""

    method: str
    n_clusters: int | None
    n_train_rows: int
    columns: tuple[str, ...]
    cluster_sizes: dict[int, int]
    assign_strategy: str
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    inertia: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "n_train_rows": self.n_train_rows,
            "columns": list(self.columns),
            "cluster_sizes": {str(k): int(v) for k, v in self.cluster_sizes.items()},
            "assign_strategy": self.assign_strategy,
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "inertia": self.inertia,
        }

    def show(self) -> None:
        print(
            f"ClusterFit · {self.method} · n_clusters={self.n_clusters} · "
            f"n_train={self.n_train_rows}"
        )
        for label, size in sorted(self.cluster_sizes.items()):
            print(f"  cluster {label}: {size}")
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class ClusterAssignResult:
    """Cluster labels for one partition (or the full frame)."""

    partition: str
    labels: tuple[int, ...]
    n_rows: int
    label_column: str
    method: str
    assign_strategy: str
    attached: bool = False
    n_noise: int = 0
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "label_column": self.label_column,
            "method": self.method,
            "assign_strategy": self.assign_strategy,
            "attached": self.attached,
            "n_noise": self.n_noise,
            "n_unique_labels": len({int(v) for v in self.labels}),
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class ClusterEvalResult:
    """Internal (and optional external) clustering evaluation on a partition."""

    partition: str
    method: str
    n_rows: int
    n_clusters_observed: int
    metrics: dict[str, float] = field(default_factory=dict)
    external_metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "n_rows": self.n_rows,
            "n_clusters_observed": self.n_clusters_observed,
            "metrics": dict(self.metrics),
            "external_metrics": dict(self.external_metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        print(
            f"ClusterEval · {self.method} · partition={self.partition} · "
            f"n={self.n_rows} · k_obs={self.n_clusters_observed}"
        )
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for key, value in self.external_metrics.items():
            print(f"  external.{key}: {value:.6f}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")
