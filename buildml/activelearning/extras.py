"""Optional dependency gates for active-learning industry backends."""

from __future__ import annotations

import importlib.util

from buildml.dl.extras import torch_available, torch_spec_available


def scikit_activeml_spec_present() -> bool:
    return importlib.util.find_spec("skactiveml") is not None


def scikit_activeml_available() -> bool:
    """True when scikit-activeml is installed (find_spec only — no import probe)."""
    return scikit_activeml_spec_present()


def activelearning_industry_available() -> bool:
    """Industry CoreSet/QBC strategies are available in-tree (native scoring)."""
    return True


def require_scikit_activeml(*, feature: str = "scikit-activeml industry scoring") -> None:
    from buildml.core.errors import MissingExtraError

    if not scikit_activeml_spec_present():
        raise MissingExtraError("activelearning-industry", feature)


def require_torch_activelearning(
    *, feature: str = "Torch BALD / MC-dropout active learning"
):
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "activelearning_industry_available",
    "require_scikit_activeml",
    "require_torch_activelearning",
    "scikit_activeml_available",
    "scikit_activeml_spec_present",
    "torch_available",
    "torch_spec_available",
]
