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


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def hdbscan_spec_present() -> bool:
    """Cheap find_spec discovery for hdbscan."""
    return importlib.util.find_spec("hdbscan") is not None


def umap_spec_present() -> bool:
    """Cheap find_spec discovery for umap."""
    return importlib.util.find_spec("umap") is not None


def hdbscan_available() -> bool:
    """Return whether hdbscan imports cleanly (subprocess probe)."""
    if not hdbscan_spec_present():
        return False
    return _runtime_ok("hdbscan")


def umap_available() -> bool:
    """Return whether umap imports cleanly (subprocess probe)."""
    if not umap_spec_present():
        return False
    return _runtime_ok("umap")


def unsupervised_extra_available() -> bool:
    """True when the recommended unsupervised stack imports cleanly at runtime."""
    return hdbscan_available() and umap_available()
