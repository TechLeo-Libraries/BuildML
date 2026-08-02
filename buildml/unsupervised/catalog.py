"""Clustering method catalog — backends, assign strategies, install hints."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_spec_available
from buildml.unsupervised.extras import hdbscan_available, umap_available

AssignStrategy = Literal["native", "nearest_centroid", "nearest_core", "gmm_predict"]

# Core sklearn methods (always available with scikit-learn)
CORE_SKLEARN_METHODS: frozenset[str] = frozenset(
    {
        "kmeans",
        "agglomerative",
        "dbscan",
        "gmm",
        "spectral",
        "optics",
        "mean_shift",
    }
)

# Industry-default density when hdbscan extra installed
HDBSCAN_METHOD = "hdbscan"

# Deep clustering behind buildml[torch]
TORCH_METHODS: frozenset[str] = frozenset({"dec", "idec"})

ALL_CLUSTER_METHODS: frozenset[str] = CORE_SKLEARN_METHODS | {HDBSCAN_METHOD} | TORCH_METHODS

DEFAULT_DENSITY_METHOD = HDBSCAN_METHOD if hdbscan_available() else "dbscan"
DEFAULT_TABULAR_DEEP_METHOD = "dec"

REDUCE_METHODS_CORE: frozenset[str] = frozenset({"pca", "tsne"})
REDUCE_METHODS_EXTRA: frozenset[str] = frozenset({"umap"})
ALL_REDUCE_METHODS: frozenset[str] = REDUCE_METHODS_CORE | REDUCE_METHODS_EXTRA

DEFAULT_REDUCE_VIZ = "umap" if umap_available() else "pca"


def method_assign_strategy(method: str) -> AssignStrategy:
    if method in {"kmeans", "dec", "idec"}:
        return "native"
    if method == "gmm":
        return "gmm_predict"
    if method in {"dbscan", "hdbscan", "optics"}:
        return "nearest_core"
    return "nearest_centroid"


def method_requires_extra(method: str) -> str | None:
    if method == "hdbscan":
        return "unsupervised"
    if method in TORCH_METHODS:
        return "torch"
    return None


def method_backend(method: str) -> str:
    if method in TORCH_METHODS:
        return "torch"
    if method == "hdbscan":
        return "hdbscan"
    return "sklearn"


def resolve_density_method(requested: str | None = None) -> str:
    """Pick HDBSCAN when installed unless caller explicitly requests dbscan."""
    if requested is not None:
        return requested
    return DEFAULT_DENSITY_METHOD


def list_cluster_methods(*, include_torch: bool = True) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for name in sorted(CORE_SKLEARN_METHODS):
        rows.append(
            {
                "method": name,
                "backend": "sklearn",
                "extra": None,
                "assign_strategy": method_assign_strategy(name),
                "transductive_fit": name in {"spectral", "optics"},
            }
        )
    rows.append(
        {
            "method": HDBSCAN_METHOD,
            "backend": "hdbscan" if hdbscan_available() else "unavailable",
            "extra": "unsupervised",
            "assign_strategy": "nearest_core",
            "default_when_installed": hdbscan_available(),
        }
    )
    if include_torch:
        for name in sorted(TORCH_METHODS):
            rows.append(
                {
                    "method": name,
                    "backend": "torch" if torch_spec_available() else "unavailable",
                    "extra": "torch",
                    "assign_strategy": "native",
                }
            )
    return tuple(rows)


def list_reduce_methods() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for name in sorted(REDUCE_METHODS_CORE):
        rows.append(
            {
                "method": name,
                "backend": "sklearn",
                "extra": None,
                "holdout_transform": name != "tsne",
            }
        )
    rows.append(
        {
            "method": "umap",
            "backend": "umap-learn" if umap_available() else "unavailable",
            "extra": "unsupervised",
            "holdout_transform": True,
            "default_when_installed": umap_available(),
        }
    )
    return tuple(rows)
