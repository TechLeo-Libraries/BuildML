"""Optional Graph ML dependency gates."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_pyg(*, feature: str = "Graph PyG backend") -> Any:
    """Import and return ``torch_geometric``, or raise :class:`MissingExtraError`.

    Ensures torch is importable first because PyG depends on it at runtime.

    Parameters
    ----------
    feature:
        Human-readable feature name included in the install hint.

    Returns
    -------
    Any
        The imported ``torch_geometric`` module.

    Raises
    ------
    MissingExtraError
        When torch or torch-geometric is not installed or fails to load.
    """
    from buildml.dl.extras import require_torch

    require_torch(feature=f"{feature} (PyG requires torch)")
    try:
        import torch_geometric
    except ImportError as exc:
        raise MissingExtraError("graph-pyg", feature) from exc
    except OSError as exc:
        raise MissingExtraError("graph-pyg", feature) from exc
    return torch_geometric


def pyg_spec_present() -> bool:
    """Cheap find_spec discovery for torch-geometric."""
    return importlib.util.find_spec("torch_geometric") is not None


def pyg_available() -> bool:
    """Return whether torch-geometric is discoverable via importlib.

    Uses ``find_spec`` only; does not verify torch or PyG import cleanly.
    Prefer :func:`pyg_runtime_available` for backend readiness.
    """
    return pyg_spec_present()


def pyg_runtime_available() -> bool:
    """Return whether torch-geometric and torch both import cleanly.

    Subprocess-probes ``torch_geometric`` after confirming torch so a broken
    PyG wheel cannot hard-crash the parent process.
    """
    if not pyg_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok, torch_available

    if not torch_available():
        return False
    return _subprocess_import_ok("torch_geometric")


def require_networkx(*, feature: str = "Classical graph features") -> Any:
    """Import and return ``networkx``, or raise :class:`MissingExtraError`.

    Used by classical graph-feature paths that require the ``buildml[graph]``
    extra before computing NetworkX metrics.

    Parameters
    ----------
    feature:
        Human-readable feature name included in the install hint.

    Returns
    -------
    Any
        The imported ``networkx`` module.

    Raises
    ------
    MissingExtraError
        When NetworkX is not installed or fails to load.
    """
    try:
        import networkx
    except ImportError as exc:
        raise MissingExtraError("graph", feature) from exc
    except OSError as exc:
        raise MissingExtraError("graph", feature) from exc
    return networkx


def networkx_spec_present() -> bool:
    """Cheap find_spec discovery for NetworkX."""
    return importlib.util.find_spec("networkx") is not None


def networkx_available() -> bool:
    """Return whether NetworkX imports cleanly for classical graph metrics."""
    if not networkx_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("networkx")