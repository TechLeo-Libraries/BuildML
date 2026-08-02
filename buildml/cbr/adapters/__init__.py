"""CBR retrieval backend adapters (lazy torch imports)."""

from __future__ import annotations

from typing import Any

from buildml.cbr.adapters.industry_ann import (
    add_vectors_to_ann_index,
    build_ann_index,
    query_ann_index,
)
from buildml.cbr.adapters.sklearn_retrieval import batch_neighbor_orders
from buildml.cbr.adapters.text_embed import (
    embed_text_cases,
    embed_text_queries,
)

__all__ = [
    "add_vectors_to_ann_index",
    "batch_neighbor_orders",
    "build_ann_index",
    "build_torch_encoder",
    "embed_text_cases",
    "embed_text_queries",
    "encode_with_torch",
    "fit_torch_encoder",
    "query_ann_index",
]


def __getattr__(name: str) -> Any:
    if name in {"build_torch_encoder", "encode_with_torch", "fit_torch_encoder"}:
        from buildml.cbr.adapters import torch_metric as torch_mod

        return getattr(torch_mod, name)
    raise AttributeError(f"module 'buildml.cbr.adapters' has no attribute {name!r}")
