"""SSL method catalog: modality routing and install hints."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Literal

from buildml.dl.extras import torch_available

logger = logging.getLogger(__name__)

Modality = Literal["tabular", "text", "vision"]

# Tabular Torch methods (industry contrastive + generative SSL)
TABULAR_TORCH_METHODS: frozenset[str] = frozenset(
    {
        "simclr_tabular",
        "byol_tabular",
        "vicreg_tabular",
        "mae_tabular",
        "vae_tabular",
    }
)

TEXT_TORCH_METHODS: frozenset[str] = frozenset(
    {
        "hf_text_ssl",
    }
)

VISION_TORCH_METHODS: frozenset[str] = frozenset(
    {
        "vision_ssl",
    }
)

TORCH_METHODS: frozenset[str] = (
    TABULAR_TORCH_METHODS | TEXT_TORCH_METHODS | VISION_TORCH_METHODS
)

# Legacy sklearn-only path (deprecated fallback)
LEGACY_SKLEARN_METHODS: frozenset[str] = frozenset({"masked_tabular"})

ALL_METHODS: frozenset[str] = TORCH_METHODS | LEGACY_SKLEARN_METHODS

DEFAULT_TABULAR_METHOD = "simclr_tabular"
DEFAULT_TEXT_METHOD = "hf_text_ssl"
DEFAULT_VISION_METHOD = "vision_ssl"
LEGACY_FALLBACK_METHOD = "masked_tabular"


def method_modality(method: str) -> Modality:
    """Map an SSL method key to its input modality.

    Used by fit routing to select tabular, text, or vision column contracts
    and install hints for the requested method.

    Parameters
    ----------
    method:
        Catalog method name such as ``simclr_tabular`` or ``hf_text_ssl``.

    Returns
    -------
    Modality
        ``tabular``, ``text``, or ``vision``.

    Raises
    ------
    ValueError
        When ``method`` is not registered in the SSL catalog.
    """
    if method in TABULAR_TORCH_METHODS or method in LEGACY_SKLEARN_METHODS:
        return "tabular"
    if method in TEXT_TORCH_METHODS:
        return "text"
    if method in VISION_TORCH_METHODS:
        return "vision"
    raise ValueError(f"Unknown SSL method {method!r}")


def resolve_default_tabular_method() -> str:
    """Pick the industry-default tabular SSL method when Torch is installed.

    Returns ``simclr_tabular`` when PyTorch imports cleanly; otherwise falls
    back to deprecated ``masked_tabular``.

    Returns
    -------
    str
        Default tabular SSL method key for Session fit.
    """
    if importlib.util.find_spec("torch") is None:
        return LEGACY_FALLBACK_METHOD
    try:
        from buildml.dl.extras import torch_available

        if torch_available():
            return DEFAULT_TABULAR_METHOD
    except Exception:
        # Torch probe failed; fall back to the non-torch tabular method.
        logger.debug("selfsupervised: torch availability probe failed", exc_info=True)
    return LEGACY_FALLBACK_METHOD


def list_ssl_methods(*, include_legacy: bool = True) -> tuple[dict[str, Any], ...]:
    """Return catalog rows for explain surfaces and documentation.

    Each row records method name, modality, backend, optional extra install
    hint, and deprecation status for legacy sklearn paths.

    Parameters
    ----------
    include_legacy:
        When True, append the deprecated ``masked_tabular`` sklearn method.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Sorted catalog entries suitable for walkthrough matrices.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(TORCH_METHODS):
        mod = method_modality(name)
        extra = "torch"
        if mod == "text":
            extra = "ssl"
        elif mod == "vision":
            extra = "vision"
        rows.append(
            {
                "method": name,
                "modality": mod,
                "backend": "torch",
                "extra": extra,
                "default_when_installed": name == DEFAULT_TABULAR_METHOD and mod == "tabular",
            }
        )
    if include_legacy:
        rows.append(
            {
                "method": LEGACY_FALLBACK_METHOD,
                "modality": "tabular",
                "backend": "sklearn",
                "extra": None,
                "deprecated": True,
            }
        )
    return tuple(rows)


def ssl_capability_matrix() -> dict[str, Any]:
    """Build an honest capability matrix for self-supervised backends.

    Reports which Torch, HF text, and vision methods are importable in the
    current environment together with install hints and non-goals.

    Returns
    -------
    dict[str, Any]
        Backend availability, default method, catalog rows, and install hints.
    """
    torch_ok = torch_available()
    st_ok = importlib.util.find_spec("sentence_transformers") is not None
    tv_ok = importlib.util.find_spec("torchvision") is not None
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": sorted(LEGACY_SKLEARN_METHODS),
                "notes": (
                    "Legacy masked_tabular sklearn fallback: deprecated when torch works."
                ),
            },
            "torch": {
                "available": torch_ok,
                "extra": "torch",
                "methods": sorted(TABULAR_TORCH_METHODS),
                "notes": "Tabular contrastive/generative SSL (SimCLR/BYOL/VICReg/MAE/VAE).",
            },
            "hf_text": {
                "available": torch_ok and st_ok,
                "extra": "ssl",
                "methods": sorted(TEXT_TORCH_METHODS),
                "notes": "HF sentence-transformer text SSL (buildml[ssl]).",
            },
            "vision": {
                "available": torch_ok and tv_ok,
                "extra": "vision",
                "methods": sorted(VISION_TORCH_METHODS),
                "notes": "Vision SSL hooks (buildml[vision]).",
            },
        },
        "default_tabular_method": resolve_default_tabular_method(),
        "methods": list(list_ssl_methods()),
        "install_hints": {
            "torch": "pip install 'buildml[torch]'",
            "ssl": "pip install 'buildml[ssl]'",
            "vision": "pip install 'buildml[vision]'",
        },
        "non_goals": [
            "Foundation-model pretraining from scratch at web scale",
            "Full MoCo/DINO research zoo",
        ],
    }
