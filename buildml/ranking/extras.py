"""Optional dependency gates for learning-to-rank industry backends.

Native sklearn pointwise/pairwise rankers are always available. LightGBM
LambdaRank, XGBoost rank:ndcg, CatBoost YetiRank, and torch listwise-lite
require ``buildml[ranking-industry]`` or ``buildml[torch]``.

Industry ``*_available`` predicates use subprocess import probes so broken
native wheels are never reported as ready. Use ``*_spec_present`` for cheap
discovery disclosure in capability matrices.

See Also
--------
buildml.ranking.catalog.ranking_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def lightgbm_spec_present() -> bool:
    """Cheap find_spec discovery for LightGBM (does not prove import works)."""
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_spec_present() -> bool:
    """Cheap find_spec discovery for XGBoost (does not prove import works)."""
    return importlib.util.find_spec("xgboost") is not None


def catboost_spec_present() -> bool:
    """Cheap find_spec discovery for CatBoost (does not prove import works)."""
    return importlib.util.find_spec("catboost") is not None


def lightgbm_available() -> bool:
    """Return whether LightGBM imports cleanly for LambdaRank paths."""
    if not lightgbm_spec_present():
        return False
    return _runtime_ok("lightgbm")


def xgboost_available() -> bool:
    """Return whether XGBoost imports cleanly for rank:ndcg paths."""
    if not xgboost_spec_present():
        return False
    return _runtime_ok("xgboost")


def catboost_available() -> bool:
    """Return whether CatBoost imports cleanly for YetiRank paths."""
    if not catboost_spec_present():
        return False
    return _runtime_ok("catboost")


def gradient_boosting_ranking_available() -> bool:
    """Return whether any GBDT ranking library imports cleanly at runtime."""
    return lightgbm_available() or xgboost_available() or catboost_available()


def ranking_industry_available() -> bool:
    """Return whether industry LTR libraries import cleanly at runtime.

    Gates capability-matrix ``available`` flags. Prefer
    ``*_spec_present`` when only install discovery is needed.
    """
    return gradient_boosting_ranking_available()


def require_lightgbm(*, feature: str = "LightGBM LambdaRank LTR") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`."""
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("ranking-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost rank:ndcg LTR") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`."""
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("ranking-industry", feature) from exc
    return xgboost


def require_catboost(*, feature: str = "CatBoost YetiRank LTR") -> Any:
    """Import and return ``catboost``, or raise :class:`MissingExtraError`."""
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("ranking-industry", feature) from exc
    return catboost


def require_torch_ranking(*, feature: str = "Torch listwise-lite LTR") -> Any:
    """Import torch for listwise-lite ranking, or raise :class:`MissingExtraError`."""
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "catboost_available",
    "catboost_spec_present",
    "gradient_boosting_ranking_available",
    "lightgbm_available",
    "lightgbm_spec_present",
    "ranking_industry_available",
    "require_catboost",
    "require_lightgbm",
    "require_torch_ranking",
    "require_xgboost",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
    "xgboost_spec_present",
]
