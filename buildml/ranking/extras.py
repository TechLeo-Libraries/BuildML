"""Optional dependency gates for learning-to-rank industry backends.

Native sklearn pointwise/pairwise rankers are always available. LightGBM
LambdaRank, XGBoost rank:ndcg, CatBoost YetiRank, and torch listwise-lite
require ``buildml[ranking-industry]`` or ``buildml[torch]``.

See Also
--------
buildml.ranking.catalog.ranking_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def lightgbm_available() -> bool:
    """Return whether LightGBM is discoverable for LambdaRank paths.

    Uses ``find_spec`` for a cheap catalog probe without importing lightgbm.

    Returns
    -------
    bool
        ``True`` when the ``lightgbm`` package is discoverable.
    """
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    """Return whether XGBoost is discoverable for rank:ndcg paths.

    Uses ``find_spec`` for a cheap catalog probe without importing xgboost.

    Returns
    -------
    bool
        ``True`` when the ``xgboost`` package is discoverable.
    """
    return importlib.util.find_spec("xgboost") is not None


def catboost_available() -> bool:
    """Return whether CatBoost is discoverable for YetiRank paths.

    Uses ``find_spec`` for a cheap catalog probe without importing catboost.

    Returns
    -------
    bool
        ``True`` when the ``catboost`` package is discoverable.
    """
    return importlib.util.find_spec("catboost") is not None


def gradient_boosting_ranking_available() -> bool:
    """Return whether any GBDT ranking library is discoverable.

    True when LightGBM, XGBoost, or CatBoost is installed for industry LTR
    backends.

    Returns
    -------
    bool
        ``True`` when at least one GBDT ranker dependency is discoverable.
    """
    return lightgbm_available() or xgboost_available() or catboost_available()


def ranking_industry_available() -> bool:
    """Return whether industry LTR libraries (GBDT rankers) are importable.

    Mirrors :func:`gradient_boosting_ranking_available` for capability-matrix
    ``industry_extra_present``.

    Returns
    -------
    bool
        ``True`` when at least one GBDT ranker dependency is discoverable.
    """
    return gradient_boosting_ranking_available()


def require_lightgbm(*, feature: str = "LightGBM LambdaRank LTR") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`.

    Called by the LightGBM adapter at fit time so missing extras surface as
    actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported lightgbm module.

    Raises
    ------
    MissingExtraError
        When LightGBM is not installed. Install with
        ``pip install 'buildml[ranking-industry]'``.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("ranking-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost rank:ndcg LTR") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`.

    Called by the XGBoost adapter at fit time so missing extras surface as
    actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported xgboost module.

    Raises
    ------
    MissingExtraError
        When XGBoost is not installed. Install with
        ``pip install 'buildml[ranking-industry]'``.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("ranking-industry", feature) from exc
    return xgboost


def require_catboost(*, feature: str = "CatBoost YetiRank LTR") -> Any:
    """Import and return ``catboost``, or raise :class:`MissingExtraError`.

    Called by the CatBoost adapter at fit time so missing extras surface as
    actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported catboost module.

    Raises
    ------
    MissingExtraError
        When CatBoost is not installed. Install with
        ``pip install 'buildml[ranking-industry]'``.
    """
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("ranking-industry", feature) from exc
    return catboost


def require_torch_ranking(*, feature: str = "Torch listwise-lite LTR") -> Any:
    """Import torch for listwise-lite ranking, or raise :class:`MissingExtraError`.

    Delegates to :func:`buildml.dl.extras.require_torch` so torch install
    guidance is consistent across BuildML domains.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported torch module.

    Raises
    ------
    MissingExtraError
        When torch is not installed. Install with ``pip install 'buildml[torch]'``.
    """
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "catboost_available",
    "gradient_boosting_ranking_available",
    "lightgbm_available",
    "ranking_industry_available",
    "require_catboost",
    "require_lightgbm",
    "require_torch_ranking",
    "require_xgboost",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
]
