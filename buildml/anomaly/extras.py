"""Optional dependency gates for anomaly industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_pyod(*, feature: str = "PyOD anomaly detectors") -> Any:
    """Import and return ``pyod`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``pyod`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import pyod
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return pyod


def pyod_spec_present() -> bool:
    """Check whether Python can locate ``pyod``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("pyod") is not None


def pyod_available() -> bool:
    """Check runtime availability for pyod.

The package must be discoverable and pass an isolated runtime import probe; an installed but broken package returns false.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not pyod_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("pyod")


def lightgbm_spec_present() -> bool:
    """Check whether Python can locate ``lightgbm``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("lightgbm") is not None


def lightgbm_available() -> bool:
    """Check runtime availability for lightgbm.

The package must be discoverable and pass an isolated runtime import probe; an installed but broken package returns false.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not lightgbm_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("lightgbm")


def xgboost_spec_present() -> bool:
    """Check whether Python can locate ``xgboost``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("xgboost") is not None


def xgboost_available() -> bool:
    """Check runtime availability for xgboost.

The package must be discoverable and pass an isolated runtime import probe; an installed but broken package returns false.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not xgboost_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("xgboost")


def gradient_boosting_extras_available() -> bool:
    """Check runtime availability for gradient boosting extras.

At least one of LightGBM and XGBoost must pass its runtime import check.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return lightgbm_available() or xgboost_available()


def anomaly_industry_available() -> bool:
    """Check runtime availability for anomaly industry.

PyOD or at least one of LightGBM and XGBoost must pass its runtime import check.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return pyod_available() or gradient_boosting_extras_available()


def require_lightgbm(*, feature: str = "LightGBM supervised anomaly scorer") -> Any:
    """Import and return ``lightgbm`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``lightgbm`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost supervised anomaly scorer") -> Any:
    """Import and return ``xgboost`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``xgboost`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return xgboost


def require_torch_anomaly(*, feature: str = "Torch autoencoder anomaly detector") -> Any:
    """Import and return ``torch`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``torch`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)
