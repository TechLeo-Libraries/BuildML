"""Configuration types for the unsupervised Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from buildml.unsupervised.catalog import ALL_CLUSTER_METHODS, ALL_REDUCE_METHODS

ClusterMethod = Literal[
    "kmeans",
    "agglomerative",
    "dbscan",
    "gmm",
    "hdbscan",
    "spectral",
    "optics",
    "mean_shift",
    "dec",
    "idec",
]

ReduceMethod = Literal["pca", "umap", "tsne"]

AssignStrategy = Literal["native", "nearest_centroid", "nearest_core", "gmm_predict"]


@dataclass(slots=True)
class ClusterConfig:
    """User-facing clustering knobs (serializable summary)."""

    method: ClusterMethod = "kmeans"
    n_clusters: int | None = 8
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    # KMeans / general
    n_init: int | str = "auto"
    max_iter: int = 300
    # Agglomerative
    linkage: str = "ward"
    # DBSCAN / density
    eps: float = 0.5
    min_samples: int = 5
    # GMM + BIC selection
    gmm_covariance_type: str = "full"
    gmm_max_components: int = 10
    gmm_select_by: str = "bic"
    # HDBSCAN
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    # Spectral
    spectral_affinity: str = "nearest_neighbors"
    spectral_n_neighbors: int = 10
    # OPTICS
    optics_min_samples: int = 5
    optics_xi: float = 0.05
    optics_min_cluster_size: float | None = None
    # Mean shift
    bandwidth: float | None = None
    # Deep clustering (DEC/IDEC)
    latent_dim: int = 10
    pretrain_epochs: int = 50
    finetune_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    # Feature resolution
    prefer_reduce_components: bool = True
    label_column: str = "cluster_id"
    # Auto k via elbow (k-means family on train)
    auto_k: bool = False
    auto_k_min: int = 2
    auto_k_max: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-friendly dict for history and bundles.

Omits private estimator and encoder fields so bundles and history records stay lightweight while preserving teaching disclosures.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
        """
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
            "gmm_covariance_type": self.gmm_covariance_type,
            "gmm_max_components": self.gmm_max_components,
            "gmm_select_by": self.gmm_select_by,
            "hdbscan_min_cluster_size": self.hdbscan_min_cluster_size,
            "hdbscan_min_samples": self.hdbscan_min_samples,
            "spectral_affinity": self.spectral_affinity,
            "spectral_n_neighbors": self.spectral_n_neighbors,
            "optics_min_samples": self.optics_min_samples,
            "optics_xi": self.optics_xi,
            "optics_min_cluster_size": self.optics_min_cluster_size,
            "bandwidth": self.bandwidth,
            "latent_dim": self.latent_dim,
            "pretrain_epochs": self.pretrain_epochs,
            "finetune_epochs": self.finetune_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "prefer_reduce_components": self.prefer_reduce_components,
            "label_column": self.label_column,
            "auto_k": self.auto_k,
            "auto_k_min": self.auto_k_min,
            "auto_k_max": self.auto_k_max,
        }


def validate_cluster_method(method: str) -> ClusterMethod:
    """Validate cluster method against supported catalog identifiers.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
ClusterMethod
    Return value (ClusterMethod) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if method not in ALL_CLUSTER_METHODS:
        raise ValueError(f"Unknown cluster method {method!r}")
    return method  # type: ignore[return-value]


def validate_reduce_method(method: str) -> ReduceMethod:
    """Validate reduce method against supported catalog identifiers.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
ReduceMethod
    Return value (ReduceMethod) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if method not in ALL_REDUCE_METHODS:
        raise ValueError(f"Unknown reduce method {method!r}")
    return method  # type: ignore[return-value]
