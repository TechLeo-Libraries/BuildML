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
    """Cheap find_spec discovery for LightGBM."""
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_spec_present() -> bool:
    """Cheap find_spec discovery for XGBoost."""
    return importlib.util.find_spec("xgboost") is not None


def lightgbm_available() -> bool:
    """Return whether lightgbm imports cleanly for pseudo-label paths."""
    if not lightgbm_spec_present():
        return False
    return _runtime_ok("lightgbm")


def xgboost_available() -> bool:
    """Return whether xgboost imports cleanly for pseudo-label paths."""
    if not xgboost_spec_present():
        return False
    return _runtime_ok("xgboost")


def gradient_boosting_extras_available() -> bool:
    """Return whether any industry GBDT library imports cleanly."""
    return lightgbm_available() or xgboost_available()


def semisupervised_industry_available() -> bool:
    """True when industry GBDT pseudo-label libraries import cleanly."""
    return gradient_boosting_extras_available()


def sentence_transformers_spec_present() -> bool:
    """Cheap find_spec discovery for sentence-transformers."""
    return importlib.util.find_spec("sentence_transformers") is not None


def sentence_transformers_available() -> bool:
    """Return whether sentence-transformers imports cleanly (subprocess-safe).

    Defers to torch first, then probes sentence_transformers out-of-process so
    a broken torch stack cannot hard-crash the parent process.
    """
    if not sentence_transformers_spec_present():
        return False
    if not torch_available():
        return False
    return _runtime_ok("sentence_transformers")


def hf_text_available() -> bool:
    """HF text semi-supervised path needs sentence-transformers at runtime."""
    return sentence_transformers_available()


def require_xgboost(*, feature: str = "XGBoost pseudo-label semi-supervised") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`."""
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM pseudo-label semi-supervised") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`."""
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return lightgbm


def require_sentence_transformers(
    *, feature: str = "HF text pseudo-label semi-supervised"
) -> Any:
    """Import sentence-transformers, or raise :class:`MissingExtraError`."""
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("ssl", feature) from exc
    return sentence_transformers


def require_torch_semisupervised(*, feature: str = "Torch consistency semi-supervised") -> Any:
    """Import torch for consistency semi-supervised, or raise MissingExtraError."""
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
