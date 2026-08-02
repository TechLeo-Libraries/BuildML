"""Optional Graph ML dependency gates."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


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
