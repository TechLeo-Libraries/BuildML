"""Optional dependency gates for AutoML industry backends.

Native grid/randomized/evolutionary search is always available. Optuna-backed
search requires ``buildml[automl]``; FLAML and AutoGluon industry adapters
require ``buildml[automl-industry]``.

See Also
--------
buildml.automl.catalog.automl_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_optuna(*, feature: str = "Optuna AutoML search") -> Any:
    """Import and return ``optuna``, or raise :class:`MissingExtraError`.

    Called by Optuna-backed AutoML paths at fit time so missing extras surface
    as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported optuna module.

    Raises
    ------
    MissingExtraError
        When optuna is not installed. Install with ``pip install 'buildml[automl]'``.
    """
    try:
        import optuna
    except ImportError as exc:
        raise MissingExtraError("automl", feature) from exc
    return optuna


def require_flaml(*, feature: str = "FLAML tabular AutoML adapter") -> Any:
    """Import and return ``flaml``, or raise :class:`MissingExtraError`.

    Called by the FLAML industry adapter when ``backend='flaml'`` is selected.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported flaml module.

    Raises
    ------
    MissingExtraError
        When flaml is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import flaml
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return flaml


def require_autogluon(*, feature: str = "AutoGluon tabular AutoML adapter") -> Any:
    """Import ``TabularPredictor``, or raise :class:`MissingExtraError`.

    Called by the AutoGluon industry adapter when ``backend='autogluon'`` is
    selected.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    type
        The ``autogluon.tabular.TabularPredictor`` class.

    Raises
    ------
    MissingExtraError
        When autogluon is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return TabularPredictor


def optuna_available() -> bool:
    """Return whether ``optuna`` appears on the import path without importing it.

    Used for capability-matrix disclosure before attempting a real import probe.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('optuna')`` succeeds.
    """
    return importlib.util.find_spec("optuna") is not None


def flaml_available() -> bool:
    """Return whether ``flaml`` appears on the import path without importing it.

    Gates the FLAML industry adapter without importing flaml at module load time.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('flaml')`` succeeds.
    """
    return importlib.util.find_spec("flaml") is not None


def autogluon_available() -> bool:
    """Return whether ``autogluon`` appears on the import path without importing it.

    Gates the AutoGluon industry adapter without importing autogluon at load time.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('autogluon')`` succeeds.
    """
    return importlib.util.find_spec("autogluon") is not None


def lightgbm_available() -> bool:
    """Return whether ``lightgbm`` appears on the import path without importing it.

    Gates optional LightGBM estimator families in the native AutoML catalog.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('lightgbm')`` succeeds.
    """
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    """Return whether ``xgboost`` appears on the import path without importing it.

    Gates optional XGBoost estimator families in the native AutoML catalog.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('xgboost')`` succeeds.
    """
    return importlib.util.find_spec("xgboost") is not None


def catboost_available() -> bool:
    """Return whether ``catboost`` appears on the import path without importing it.

    Gates optional CatBoost estimator families in the native AutoML catalog.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('catboost')`` succeeds.
    """
    return importlib.util.find_spec("catboost") is not None


def gradient_boosting_extras_available() -> bool:
    """Return whether at least one industry GBDT library is importable.

    Native AutoML extends its family catalog with LightGBM, XGBoost, and/or
    CatBoost when the corresponding packages are installed.

    Returns
    -------
    bool
        ``True`` when any of lightgbm, xgboost, or catboost is discoverable.
    """
    return lightgbm_available() or xgboost_available() or catboost_available()


def automl_industry_available() -> bool:
    """Return whether FLAML or AutoGluon industry adapters can be imported.

    Industry backends bypass fold-local recipe search and run internal model
    selection on train-only data.

    Returns
    -------
    bool
        ``True`` when flaml or autogluon is discoverable on the import path.
    """
    return flaml_available() or autogluon_available()


def require_lightgbm(*, feature: str = "LightGBM estimator family") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`.

    Called when the native AutoML catalog includes a LightGBM family entry.

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
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost estimator family") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`.

    Called when the native AutoML catalog includes an XGBoost family entry.

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
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return xgboost


def require_catboost(*, feature: str = "CatBoost estimator family") -> Any:
    """Import and return ``catboost``, or raise :class:`MissingExtraError`.

    Called when the native AutoML catalog includes a CatBoost family entry.

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
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return catboost
