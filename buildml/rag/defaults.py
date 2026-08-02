"""Runtime default resolution for the RAG path (industry vs fallback)."""

from __future__ import annotations

from buildml.rag.types import (
    DEFAULT_FUSION,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    RetrieveConfig,
    RetrieveMode,
)


def rag_semantic_stack_available() -> bool:
    """True when sentence-transformers can be imported (``buildml[rag]``)."""
    from buildml.rag.extras import rag_available

    return rag_available()


def default_embedder_spec() -> str:
    """Recommended embedder: semantic when ``buildml[rag]`` is installed, else hashing."""
    return "auto"


def default_retrieve_mode() -> RetrieveMode:
    """Hybrid BM25+dense when semantic stack is available; dense-only otherwise."""
    return "hybrid" if rag_semantic_stack_available() else "dense"


def default_retrieve_config(**overrides: object) -> RetrieveConfig:
    """Build a :class:`RetrieveConfig` with industry defaults applied at call time."""
    mode = overrides.pop("mode", default_retrieve_mode())  # type: ignore[misc]
    k = int(overrides.pop("k", DEFAULT_TOP_K))  # type: ignore[misc]
    fusion = overrides.pop("fusion", DEFAULT_FUSION)  # type: ignore[misc]
    rerank = overrides.pop("rerank", False)  # type: ignore[misc]
    cfg = RetrieveConfig(
        k=k,
        mode=mode,  # type: ignore[arg-type]
        fusion=fusion,  # type: ignore[arg-type]
        rrf_k=int(overrides.pop("rrf_k", DEFAULT_RRF_K)),  # type: ignore[misc]
        dense_weight=float(overrides.pop("dense_weight", 0.5)),  # type: ignore[misc]
        bm25_k1=float(overrides.pop("bm25_k1", 1.5)),  # type: ignore[misc]
        bm25_b=float(overrides.pop("bm25_b", 0.75)),  # type: ignore[misc]
        filters=overrides.pop("filters", None),  # type: ignore[misc]
        rerank=rerank,  # type: ignore[arg-type]
        rerank_model=overrides.pop("rerank_model", None),  # type: ignore[misc]
        rerank_candidates=int(overrides.pop("rerank_candidates", DEFAULT_RERANK_CANDIDATES)),  # type: ignore[misc]
    )
    if overrides:
        raise TypeError(f"Unexpected retrieve config overrides: {sorted(overrides)}")
    return cfg
