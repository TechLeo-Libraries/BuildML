"""Optional Graph ML dependency gates."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_pyg(*, feature: str = "Graph PyG backend") -> Any:
    """Import and return ``torch_geometric``, or raise :class:`MissingExtraError`."""
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
    """Cheap check: is torch-geometric installed?"""
    return importlib.util.find_spec("torch_geometric") is not None


def pyg_runtime_available() -> bool:
    """True when torch-geometric and torch both import cleanly."""
    if not pyg_available():
        return False
    from buildml.dl.extras import torch_available

    return torch_available()


def require_networkx(*, feature: str = "Classical graph features") -> Any:
    """Import and return ``networkx``, or raise :class:`MissingExtraError`."""
    try:
        import networkx
    except ImportError as exc:
        raise MissingExtraError("graph", feature) from exc
    except OSError as exc:
        raise MissingExtraError("graph", feature) from exc
    return networkx


def networkx_available() -> bool:
    """Return True when ``networkx`` can be imported."""
    if importlib.util.find_spec("networkx") is None:
        return False
    try:
        import networkx  # noqa: F401
    except Exception:
        return False
    return True
