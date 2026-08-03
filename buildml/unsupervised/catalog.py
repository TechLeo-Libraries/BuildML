"""Clustering method catalog — backends, assign strategies, install hints."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_available, torch_spec_available
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
    """Perform method assign strategy for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
AssignStrategy
    Return value (AssignStrategy) produced by this operation.
    """
    if method in {"kmeans", "dec", "idec"}:
        return "native"
    if method == "gmm":
        return "gmm_predict"
    if method in {"dbscan", "hdbscan", "optics"}:
        return "nearest_core"
    return "nearest_centroid"


def method_requires_extra(method: str) -> str | None:
    """Perform method requires extra for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
str | None
    Return value (str | None) produced by this operation.
    """
    if method == "hdbscan":
        return "unsupervised"
    if method in TORCH_METHODS:
        return "torch"
    return None


def method_backend(method: str) -> str:
    """Perform method backend for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
str
    Return value (str) produced by this operation.
    """
    if method in TORCH_METHODS:
        return "torch"
    if method == "hdbscan":
        return "hdbscan"
    return "sklearn"


def resolve_density_method(requested: str | None = None) -> str:
    """Pick HDBSCAN when installed unless caller explicitly requests dbscan.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
requested:
    requested (str | None).

Returns
-------
str
    Return value (str) produced by this operation.
    """
    if requested is not None:
        return requested
    return DEFAULT_DENSITY_METHOD


def list_cluster_methods(*, include_torch: bool = True) -> tuple[dict[str, Any], ...]:
    """List catalog entries for cluster methods.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
include_torch:
    include torch (bool).

Returns
-------
tuple[dict[str, Any], ...]
    Tuple of results (tuple[dict[str, Any], ...]) for downstream Session steps.
    """
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
                    "backend": "torch" if torch_available() else "unavailable",
                    "extra": "torch",
                    "assign_strategy": "native",
                }
            )
    return tuple(rows)


def unsupervised_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for clustering / reduction backends.

Reports installed backends, supported methods, evaluation rules, install hints, and explicit non-goals for teaching overlays.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": sorted(CORE_SKLEARN_METHODS),
                "notes": "Core sklearn clustering — always available.",
            },
            "hdbscan": {
                "available": hdbscan_available(),
                "extra": "unsupervised",
                "methods": [HDBSCAN_METHOD],
                "notes": "HDBSCAN density clustering when buildml[unsupervised] installed.",
            },
            "torch": {
                "available": torch_available(),
                "extra": "torch",
                "methods": sorted(TORCH_METHODS),
                "notes": "DEC/IDEC deep clustering when torch imports cleanly.",
            },
        },
        "reduction": {
            "methods": list(list_reduce_methods()),
            "default_viz": DEFAULT_REDUCE_VIZ,
            "umap_present": umap_available(),
        },
        "default_density_method": DEFAULT_DENSITY_METHOD,
        "default_tabular_deep_method": DEFAULT_TABULAR_DEEP_METHOD,
        "cluster_methods": list(list_cluster_methods()),
        "torch_spec_present": torch_spec_available(),
        "install_hints": {
            "unsupervised": "pip install 'buildml[unsupervised]'  # hdbscan + umap-learn",
            "torch": "pip install 'buildml[torch]'  # DEC/IDEC",
        },
        "non_goals": [
            "Full deep clustering research zoo",
            "GPU-scale embedding clustering products",
        ],
    }


def list_reduce_methods() -> tuple[dict[str, Any], ...]:
    """List catalog entries for reduce methods.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
tuple[dict[str, Any], ...]
    Tuple of results (tuple[dict[str, Any], ...]) for downstream Session steps.
    """
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
