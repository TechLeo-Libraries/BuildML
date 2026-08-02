"""Optional dependency gates for the unsupervised domain."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_hdbscan(*, feature: str = "HDBSCAN clustering") -> Any:
    """Import and return ``hdbscan``, or raise :class:`MissingExtraError`."""
    try:
        import hdbscan
    except ImportError as exc:
        raise MissingExtraError("unsupervised", feature) from exc
    return hdbscan


def require_umap(*, feature: str = "UMAP dimensionality reduction") -> Any:
    """Import and return ``umap``, or raise :class:`MissingExtraError`."""
    try:
        import umap
    except ImportError as exc:
        raise MissingExtraError("unsupervised", feature) from exc
    return umap


def hdbscan_available() -> bool:
    return importlib.util.find_spec("hdbscan") is not None


def umap_available() -> bool:
    return importlib.util.find_spec("umap") is not None


def unsupervised_extra_available() -> bool:
    """True when the recommended unsupervised stack (hdbscan + umap-learn) is importable."""
    return hdbscan_available() and umap_available()
