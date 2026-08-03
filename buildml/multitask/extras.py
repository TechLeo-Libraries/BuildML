"""Optional dependency gates for multi-task industry backends.

Native sklearn MultiOutput/Chain paths are always available. Industry GBDT
multi-target and torch shared-trunk paths require optional extras.

See Also
--------
buildml.multitask.catalog.multitask_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def lightgbm_available() -> bool:
    """Return whether ``lightgbm`` appears on the import path without importing it.

    Gates ``multi_output_lgbm`` without importing lightgbm at module load time.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('lightgbm')`` succeeds.
    """
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    """Return whether ``xgboost`` appears on the import path without importing it.

    Gates ``multi_output_xgb`` without importing xgboost at module load time.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('xgboost')`` succeeds.
    """
    return importlib.util.find_spec("xgboost") is not None


def catboost_available() -> bool:
    """Return whether ``catboost`` appears on the import path without importing it.

    Gates ``multi_output_catboost`` without importing catboost at load time.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('catboost')`` succeeds.
    """
    return importlib.util.find_spec("catboost") is not None


def gradient_boosting_extras_available() -> bool:
    """Return whether at least one industry GBDT library is importable.

    Used by the capability matrix to disclose industry multi-target paths.

    Returns
    -------
    bool
        ``True`` when any of lightgbm, xgboost, or catboost is discoverable.
    """
    return lightgbm_available() or xgboost_available() or catboost_available()


def multitask_industry_available() -> bool:
    """Return whether industry GBDT multi-target libraries are importable.

    Industry backends require same-type targets and honest MultiOutput wrappers
    for classification.

    Returns
    -------
    bool
        ``True`` when at least one GBDT extra is discoverable.
    """
    return gradient_boosting_extras_available()


def require_xgboost(*, feature: str = "XGBoost multi-target multi-task") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`.

    Called by the ``multi_output_xgb`` adapter at fit time.

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
        When xgboost is not installed. Install with
        ``pip install 'buildml[multitask-industry]'``.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM multi-target multi-task") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`.

    Called by the ``multi_output_lgbm`` adapter at fit time.

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
        When lightgbm is not installed. Install with
        ``pip install 'buildml[multitask-industry]'``.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return lightgbm


def require_catboost(*, feature: str = "CatBoost multi-target multi-task") -> Any:
    """Import and return ``catboost``, or raise :class:`MissingExtraError`.

    Called by the ``multi_output_catboost`` adapter at fit time.

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
        When catboost is not installed. Install with
        ``pip install 'buildml[multitask-industry]'``.
    """
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return catboost


def require_torch_multitask(*, feature: str = "Torch shared-trunk multi-head multi-task") -> Any:
    """Import and return ``torch`` for the shared-trunk multi-head backend.

    Delegates to :func:`buildml.dl.extras.require_torch` with multi-task wording.

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
    "gradient_boosting_extras_available",
    "lightgbm_available",
    "multitask_industry_available",
    "require_catboost",
    "require_lightgbm",
    "require_torch_multitask",
    "require_xgboost",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
]
