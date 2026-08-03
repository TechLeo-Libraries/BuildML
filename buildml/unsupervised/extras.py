"""Optional dependency gates for the unsupervised domain."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_hdbscan(*, feature: str = "HDBSCAN clustering") -> Any:
    """Import and return ``hdbscan``, or raise :class:`MissingExtraError`.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise MissingExtraError("unsupervised", feature) from exc
    return hdbscan


def require_umap(*, feature: str = "UMAP dimensionality reduction") -> Any:
    """Import and return ``umap``, or raise :class:`MissingExtraError`.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import umap
    except ImportError as exc:
        raise MissingExtraError("unsupervised", feature) from exc
    return umap


def hdbscan_available() -> bool:
    """Return whether hdbscan optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return importlib.util.find_spec("hdbscan") is not None


def umap_available() -> bool:
    """Return whether umap optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return importlib.util.find_spec("umap") is not None


def unsupervised_extra_available() -> bool:
    """True when the recommended unsupervised stack (hdbscan + umap-learn) is importable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return hdbscan_available() and umap_available()
