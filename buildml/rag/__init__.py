"""Retrieval-augmented generation domain. Lazy imports — core never requires RAG extras."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "BM25Index",
    "Chunk",
    "ChunkConfig",
    "ChunkResult",
    "ConfigCompareResult",
    "CorpusHandle",
    "Document",
    "EmbedConfig",
    "EvalConfig",
    "HashingEmbedder",
    "IndexConfig",
    "IndexResult",
    "RagEvalResult",
    "RagIndex",
    "RetrieveConfig",
    "RetrieveResult",
    "SentenceTransformerEmbedder",
    "build_index",
    "chunk_documents",
    "compare_retrieval_configs",
    "corpus_from_documents",
    "corpus_from_frame",
    "evaluate_retrieval",
    "load_rag_bundle",
    "load_text_corpus",
    "rag_available",
    "rag_status",
    "require_rag_stack",
    "require_sentence_transformers",
    "retrieve",
    "rrf_fuse",
    "save_rag_bundle",
    "weighted_fuse",
]


def __getattr__(name: str) -> Any:
    if name in {"require_rag_stack", "require_sentence_transformers", "rag_available"}:
        from buildml.rag import extras

        return getattr(extras, name)
    if name in {
        "ChunkConfig",
        "EmbedConfig",
        "EvalConfig",
        "IndexConfig",
        "RetrieveConfig",
    }:
        from buildml.rag import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "Chunk",
        "ChunkResult",
        "ConfigCompareResult",
        "CorpusHandle",
        "Document",
        "IndexResult",
        "RagEvalResult",
        "RetrieveResult",
    }:
        from buildml.rag import results

        return getattr(results, name)
    if name in {"HashingEmbedder", "SentenceTransformerEmbedder"}:
        from buildml.rag import embed

        return getattr(embed, name)
    if name in {"corpus_from_documents", "corpus_from_frame", "load_text_corpus"}:
        from buildml.rag import corpus

        return getattr(corpus, name)
    if name == "chunk_documents":
        from buildml.rag.chunk import chunk_documents

        return chunk_documents
    if name in {"RagIndex", "build_index"}:
        from buildml.rag import index as index_mod

        return getattr(index_mod, name)
    if name == "retrieve":
        from buildml.rag.retrieve import retrieve

        return retrieve
    if name in {"BM25Index", "rrf_fuse", "weighted_fuse"}:
        from buildml.rag import hybrid

        return getattr(hybrid, name)
    if name in {"evaluate_retrieval", "compare_retrieval_configs"}:
        from buildml.rag import evaluate as evaluate_mod

        return getattr(evaluate_mod, name)
    if name in {"rag_status"}:
        from buildml.rag.explain_hooks import rag_status

        return rag_status
    if name in {"BUNDLE_FORMAT", "CHECKPOINT_BOUNDARY", "save_rag_bundle", "load_rag_bundle"}:
        from buildml.rag import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    raise AttributeError(f"module 'buildml.rag' has no attribute {name!r}")
