"""Optional dependency gates for online / continual industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def river_spec_present() -> bool:
    return importlib.util.find_spec("river") is not None


def river_available() -> bool:
    """True when River can be imported (not just find_spec)."""
    if not river_spec_present():
        return False
    try:
        import river  # noqa: F401
    except Exception:
        return False
    return True


def online_industry_available() -> bool:
    """Industry streaming path is available when River is importable."""
    return river_available()


def require_river(*, feature: str = "River streaming online learning") -> Any:
    try:
        import river  # noqa: F401
    except ImportError as exc:
        raise MissingExtraError("online-industry", feature) from exc
    return river


def require_torch_continual(*, feature: str = "Torch replay / EWC continual learning") -> Any:
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "online_industry_available",
    "require_river",
    "require_torch_continual",
    "river_available",
    "river_spec_present",
    "torch_available",
    "torch_spec_available",
]
