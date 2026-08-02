"""Optional dependency gates for meta-learning industry / torch backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.dl.extras import torch_available, torch_spec_available


def learn2learn_spec_present() -> bool:
    return importlib.util.find_spec("learn2learn") is not None


def learn2learn_available() -> bool:
    """True when learn2learn can be imported (MAML/Reptile industry path)."""
    if not learn2learn_spec_present():
        return False
    try:
        import learn2learn  # noqa: F401
    except Exception:
        return False
    return True


def metalearning_industry_available() -> bool:
    """Industry MAML/Reptile tabular adapters (requires buildml[torch])."""
    return torch_spec_available()


def metalearning_torch_available() -> bool:
    """Deep prototypical encoder path (buildml[torch])."""
    return torch_spec_available()


def require_learn2learn(*, feature: str = "MAML/Reptile meta-learning") -> Any:
    from buildml.core.errors import MissingExtraError

    try:
        import learn2learn
    except ImportError as exc:
        raise MissingExtraError("metalearning-industry", feature) from exc
    return learn2learn


def require_torch_metalearning(*, feature: str = "Torch meta-learning") -> Any:
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "learn2learn_available",
    "learn2learn_spec_present",
    "metalearning_industry_available",
    "metalearning_torch_available",
    "require_learn2learn",
    "require_torch_metalearning",
    "torch_available",
    "torch_spec_available",
]
