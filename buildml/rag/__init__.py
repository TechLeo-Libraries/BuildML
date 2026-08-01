"""Retrieval-augmented generation domain. Lazy imports — core never requires RAG extras."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "Chunk",
    "ChunkConfig",
    "ChunkResult",
    "CorpusHandle",
    "Document",
    "EmbedConfig",
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
    "corpus_from_documents",
    "corpus_from_frame",
    "evaluate_retrieval",
    "load_rag_bundle",
    "load_text_corpus",
    "rag_available",
    "require_rag_stack",
    "require_sentence_transformers",
    "retrieve",
    "save_rag_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"require_rag_stack", "require_sentence_transformers", "rag_available"}:
        from buildml.rag import extras

        return getattr(extras, name)
    if name in {
        "ChunkConfig",
        "EmbedConfig",
        "IndexConfig",
        "RetrieveConfig",
    }:
        from buildml.rag import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "Chunk",
        "ChunkResult",
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
    if name == "evaluate_retrieval":
        from buildml.rag.evaluate import evaluate_retrieval

        return evaluate_retrieval
    if name in {"BUNDLE_FORMAT", "CHECKPOINT_BOUNDARY", "save_rag_bundle", "load_rag_bundle"}:
        from buildml.rag import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    raise AttributeError(f"module 'buildml.rag' has no attribute {name!r}")
