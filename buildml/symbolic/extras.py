"""Optional dependency gates for symbolic industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def skope_rules_available() -> bool:
    """True when skope-rules (``skrules``) imports cleanly.

    find_spec alone is insufficient — skope-rules 1.x imports ``collections.Iterable``
    and fails on Python 3.13 even when the dist is installed.
    """
    if importlib.util.find_spec("skrules") is None:
        return False
    try:
        from skrules import SkopeRules  # noqa: F401
    except Exception:
        return False
    return True


def imodels_available() -> bool:
    if importlib.util.find_spec("imodels") is None:
        return False
    try:
        import imodels  # noqa: F401
    except Exception:
        return False
    return True


def z3_available() -> bool:
    if importlib.util.find_spec("z3") is None:
        return False
    try:
        import z3  # noqa: F401
    except Exception:
        return False
    return True


def symbolic_industry_available() -> bool:
    """True when skope-rules or imodels is importable (not merely installed)."""
    return skope_rules_available() or imodels_available()


def require_skope_rules(*, feature: str = "SkopeRules rule induction") -> Any:
    try:
        from skrules import SkopeRules
    except ImportError as exc:
        raise MissingExtraError("symbolic-industry", feature) from exc
    return SkopeRules


def require_imodels(*, feature: str = "imodels interpretable rule export") -> Any:
    try:
        import imodels
    except ImportError as exc:
        raise MissingExtraError("symbolic-industry", feature) from exc
    return imodels


def require_z3(*, feature: str = "Z3 rule-set constraint verification") -> Any:
    try:
        import z3
    except ImportError as exc:
        raise MissingExtraError("symbolic-industry", feature) from exc
    return z3


def require_torch_symbolic(*, feature: str = "Torch neuro-symbolic tabular models") -> Any:
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


def torch_neuro_available() -> bool:
    """True when torch imports cleanly for neuro-symbolic bases."""
    return torch_available()
