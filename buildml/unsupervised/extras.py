"""Optional dependency gates for the unsupervised domain."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_hdbscan(*, feature: str = "HDBSCAN clustering") -> Any:
    """Import and return ``hdbscan`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``hdbscan`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise MissingExtraError("unsupervised", feature) from exc
    return hdbscan


def require_umap(*, feature: str = "UMAP dimensionality reduction") -> Any:
    """Import and return ``umap`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``umap`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
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
    """Check whether Python can locate ``hdbscan``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("hdbscan") is not None


def umap_spec_present() -> bool:
    """Check whether Python can locate ``umap``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("umap") is not None


def hdbscan_available() -> bool:
    """Check runtime availability for hdbscan.

The package must be discoverable and pass an isolated runtime import probe; an installed but broken package returns false.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not hdbscan_spec_present():
        return False
    return _runtime_ok("hdbscan")


def umap_available() -> bool:
    """Check runtime availability for umap.

The package must be discoverable and pass an isolated runtime import probe; an installed but broken package returns false.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not umap_spec_present():
        return False
    return _runtime_ok("umap")


def unsupervised_extra_available() -> bool:
    """Check runtime availability for unsupervised extra.

Both HDBSCAN and UMAP must pass their runtime import checks.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return hdbscan_available() and umap_available()
