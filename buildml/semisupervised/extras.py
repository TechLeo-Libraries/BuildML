"""Optional dependency gates for semi-supervised industry backends.

Native label propagation / self-training paths are always available. Industry
GBDT pseudo-label and HF text paths use runtime import probes so broken wheels
are never reported as ready.

See Also
--------
buildml.semisupervised.catalog.semisupervised_capability_matrix
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
    """Check whether Python can locate ``lightgbm``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_spec_present() -> bool:
    """Check whether Python can locate ``xgboost``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("xgboost") is not None


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
    return _runtime_ok("lightgbm")


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
    return _runtime_ok("xgboost")


def gradient_boosting_extras_available() -> bool:
    """Check runtime availability for gradient boosting extras.

At least one of LightGBM and XGBoost must pass its runtime import check.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return lightgbm_available() or xgboost_available()


def semisupervised_industry_available() -> bool:
    """Check runtime availability for semisupervised industry.

At least one of LightGBM and XGBoost must pass its runtime import check.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return gradient_boosting_extras_available()


def sentence_transformers_spec_present() -> bool:
    """Check whether Python can locate ``sentence_transformers``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


def sentence_transformers_available() -> bool:
    """Check runtime availability for sentence transformers.

Torch is checked first, then sentence-transformers is imported in a subprocess to isolate native-library failures.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not sentence_transformers_spec_present():
        return False
    if not torch_available():
        return False
    return _runtime_ok("sentence_transformers")


def hf_text_available() -> bool:
    """Check runtime availability for hf text.

This delegates to the sentence-transformers check, including its Torch prerequisite.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return sentence_transformers_available()


def require_xgboost(*, feature: str = "XGBoost pseudo-label semi-supervised") -> Any:
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
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM pseudo-label semi-supervised") -> Any:
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
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return lightgbm


def require_sentence_transformers(
    *, feature: str = "HF text pseudo-label semi-supervised"
) -> Any:
    """Import and return ``sentence_transformers`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``sentence_transformers`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("ssl", feature) from exc
    return sentence_transformers


def require_torch_semisupervised(*, feature: str = "Torch consistency semi-supervised") -> Any:
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


__all__ = [
    "gradient_boosting_extras_available",
    "hf_text_available",
    "lightgbm_available",
    "lightgbm_spec_present",
    "require_lightgbm",
    "require_sentence_transformers",
    "require_torch_semisupervised",
    "require_xgboost",
    "semisupervised_industry_available",
    "sentence_transformers_available",
    "sentence_transformers_spec_present",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
    "xgboost_spec_present",
]
