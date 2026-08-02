"""Unsupervised learning domain (clustering + train-fit / holdout-assign path).

Industry-depth coverage (Phase R2):
  - Clustering: k-means, agglomerative, DBSCAN, GMM+BIC, HDBSCAN, spectral,
    OPTICS, mean-shift, DEC/IDEC (Torch).
  - Reduction: PCA (core), UMAP/t-SNE (viz + cluster pipeline input).
  - Validation: silhouette, Davies–Bouldin, Calinski–Harabasz, stability,
    elbow/inertia, transductive disclosures.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Industry defaults
when extras installed:
  - ``buildml[unsupervised]`` → hdbscan, umap-learn
  - ``buildml[torch]`` → DEC/IDEC deep clustering

Lazy imports — core never grows heavy unsupervised stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_FORMAT_V1",
    "BUNDLE_FORMAT_V2",
    "CHECKPOINT_BOUNDARY",
    "AssignStrategy",
    "ClusterAssignResult",
    "ClusterConfig",
    "ClusterEvalResult",
    "ClusterFitResult",
    "ClusterMethod",
    "ClusterPlan",
    "ReduceMethod",
    "assign_clusters",
    "evaluate_clustering",
    "fit_clusterer",
    "list_cluster_methods",
    "list_reduce_methods",
    "load_unsupervised_bundle",
    "save_unsupervised_bundle",
    "unsupervised_status",
    "unsupervised_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {"ClusterMethod", "AssignStrategy", "ClusterConfig", "ReduceMethod"}:
        from buildml.unsupervised import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ClusterPlan",
        "ClusterFitResult",
        "ClusterAssignResult",
        "ClusterEvalResult",
    }:
        from buildml.unsupervised import results as results_mod

        return getattr(results_mod, name)
    if name in {"fit_clusterer", "assign_clusters"}:
        from buildml.unsupervised import cluster as cluster_mod

        return getattr(cluster_mod, name)
    if name == "evaluate_clustering":
        from buildml.unsupervised.evaluate import evaluate_clustering

        return evaluate_clustering
    if name in {"list_cluster_methods", "list_reduce_methods"}:
        from buildml.unsupervised import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "BUNDLE_FORMAT_V1",
        "BUNDLE_FORMAT_V2",
        "CHECKPOINT_BOUNDARY",
        "save_unsupervised_bundle",
        "load_unsupervised_bundle",
    }:
        from buildml.unsupervised import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"unsupervised_status", "unsupervised_status_for_session"}:
        from buildml.unsupervised import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.unsupervised' has no attribute {name!r}")
