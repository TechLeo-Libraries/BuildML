"""Optional dependency gates for federated industry backends (Flower / flwr)."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def flwr_available() -> bool:
    return importlib.util.find_spec("flwr") is not None


def federated_industry_available() -> bool:
    """True when Flower (``flwr``) is importable."""
    return flwr_available()


def require_flwr(*, feature: str = "Flower federated backend") -> Any:
    """Import and return ``flwr``, or raise :class:`MissingExtraError`."""
    try:
        import flwr
    except ImportError as exc:
        raise MissingExtraError("federated-industry", feature) from exc
    return flwr
