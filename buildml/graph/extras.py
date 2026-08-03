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


def pyg_available() -> bool:
    """Return whether torch-geometric is discoverable via importlib.

    Uses ``find_spec`` only; does not verify torch imports cleanly.

    Returns
    -------
    bool
        True when the ``torch_geometric`` package is installed.
    """
    return importlib.util.find_spec("torch_geometric") is not None


def pyg_runtime_available() -> bool:
    """Return whether torch-geometric and torch both import cleanly.

    Used by the capability matrix to distinguish install presence from a
    working runtime suitable for the ``pyg`` backend.

    Returns
    -------
    bool
        True when both torch and torch-geometric import without error.
    """
    if not pyg_available():
        return False
    from buildml.dl.extras import torch_available

    return torch_available()


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


def networkx_available() -> bool:
    """Return whether NetworkX can be imported for classical graph metrics.

    Performs a lightweight import check beyond ``find_spec`` so partially
    broken installs report unavailable.

    Returns
    -------
    bool
        True when ``networkx`` imports without error.
    """
    if importlib.util.find_spec("networkx") is None:
        return False
    try:
        import networkx  # noqa: F401
    except Exception:
        return False
    return True
