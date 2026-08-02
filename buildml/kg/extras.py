"""Optional dependency gates for KG industry backends (PyKEEN)."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_pykeen(*, feature: str = "PyKEEN KG backend") -> Any:
    """Import and return ``pykeen``, or raise :class:`MissingExtraError`."""
    from buildml.dl.extras import require_torch

    require_torch(feature=f"{feature} (PyKEEN requires torch)")
    try:
        import pykeen
    except ImportError as exc:
        raise MissingExtraError("kg-industry", feature) from exc
    return pykeen


def pykeen_available() -> bool:
    """Cheap check: is a PyKEEN distribution installed?"""
    return importlib.util.find_spec("pykeen") is not None


def pykeen_runtime_available() -> bool:
    """True when PyKEEN and torch both import cleanly."""
    if not pykeen_available():
        return False
    from buildml.dl.extras import torch_available

    return torch_available()


def kg_industry_available() -> bool:
    """True when PyKEEN is importable (spec present)."""
    return pykeen_available()
