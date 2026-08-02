"""SSL method catalog — modality routing and install hints."""

from __future__ import annotations

from typing import Any, Literal

import importlib.util

from buildml.dl.extras import torch_available

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
    if method in TABULAR_TORCH_METHODS or method in LEGACY_SKLEARN_METHODS:
        return "tabular"
    if method in TEXT_TORCH_METHODS:
        return "text"
    if method in VISION_TORCH_METHODS:
        return "vision"
    raise ValueError(f"Unknown SSL method {method!r}")


def resolve_default_tabular_method() -> str:
    """Pick industry-default tabular SSL when Torch is installed."""
    if importlib.util.find_spec("torch") is None:
        return LEGACY_FALLBACK_METHOD
    try:
        from buildml.dl.extras import torch_available

        if torch_available():
            return DEFAULT_TABULAR_METHOD
    except Exception:
        pass
    return LEGACY_FALLBACK_METHOD


def list_ssl_methods(*, include_legacy: bool = True) -> tuple[dict[str, Any], ...]:
    """Return catalog rows for explain/docs surfaces."""
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
