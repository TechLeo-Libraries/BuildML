"""Optional Gymnasium dependency gate for ``buildml[rl]``."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_gymnasium(*, feature: str = "Gymnasium RL loop") -> Any:
    """Import and return ``gymnasium``, or raise :class:`MissingExtraError`."""
    try:
        import gymnasium
    except ImportError as exc:
        raise MissingExtraError("rl", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rl", feature) from exc
    return gymnasium


def gymnasium_available() -> bool:
    """Return True when ``gymnasium`` can be imported."""
    if importlib.util.find_spec("gymnasium") is None:
        return False
    try:
        import gymnasium  # noqa: F401
    except Exception:
        return False
    return True
