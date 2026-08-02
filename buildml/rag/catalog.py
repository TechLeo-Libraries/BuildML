"""RAG capability matrix — honest backend / embedder disclosure."""

from __future__ import annotations

from typing import Any

from buildml.rag.defaults import default_retrieve_mode
from buildml.rag.extras import rag_advanced_available, rag_available


def rag_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for RAG embed / retrieve / generate stacks."""
    semantic = rag_available()
    advanced = rag_advanced_available()
    return {
        "backends": {
            "hashing": {
                "available": True,
                "extra": None,
                "embedder": "HashingEmbedder",
                "notes": "Core numpy/sklearn hashing embeddings — always available.",
            },
            "semantic": {
                "available": semantic,
                "extra": "rag",
                "embedder": "SentenceTransformerEmbedder",
                "notes": (
                    "sentence-transformers dense embeddings when buildml[rag] imports "
                    "cleanly (real import probe, not find_spec alone)."
                ),
            },
            "langchain": {
                "available": advanced,
                "extra": "rag-advanced",
                "embedder": "LangChain community retrieve hooks",
                "notes": "Optional LangChain retrieve adapters (buildml[rag-advanced]).",
            },
        },
        "retrieve": {
            "modes": ["dense", "hybrid", "bm25"],
            "default_mode": default_retrieve_mode(),
            "fusion": ["rrf", "weighted"],
            "rerank": semantic,
        },
        "generate": {
            "echo_grounded": True,
            "external_llm": "via buildml[ai] / provider hooks — not bundled here",
        },
        "default_embedder_when_installed": "semantic" if semantic else "hashing",
        "install_hints": {
            "rag": "pip install 'buildml[rag]'  # sentence-transformers + transformers",
            "rag-advanced": "pip install 'buildml[rag-advanced]'  # LangChain community",
        },
        "non_goals": [
            "Hosted vector DB product (Pinecone/Weaviate)",
            "Managed RAG SaaS orchestration",
            "Full LangChain agent frameworks",
        ],
        "rag_extra_present": semantic,
        "rag_advanced_present": advanced,
    }
