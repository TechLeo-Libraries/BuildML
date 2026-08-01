"""Typed results for the RAG retrieve path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """One corpus document."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "index"  # "index" | "eval_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Document:
        return cls(
            doc_id=str(payload["doc_id"]),
            text=str(payload["text"]),
            metadata=dict(payload.get("metadata") or {}),
            role=str(payload.get("role") or "index"),
        )


@dataclass(slots=True)
class Chunk:
    """One text chunk with stable ids."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            text=str(payload["text"]),
            start_char=int(payload["start_char"]),
            end_char=int(payload["end_char"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class CorpusHandle:
    """In-memory corpus for the Session RAG path."""

    documents: tuple[Document, ...]
    source: str = "memory"

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_documents": self.n_documents,
            "source": self.source,
            "doc_ids": [d.doc_id for d in self.documents],
            "roles": sorted({d.role for d in self.documents}),
        }


@dataclass(slots=True)
class ChunkResult:
    """Chunking output summary."""

    chunks: tuple[Chunk, ...]
    config: dict[str, Any]
    n_documents: int

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks,
            "n_documents": self.n_documents,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class Hit:
    """One ranked retrieval hit."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "text": self.text,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class IndexResult:
    """Built vector index attached to the Session."""

    n_chunks: int
    n_documents: int
    embedder_id: str
    dim: int
    store_backend: str
    chunk_config: dict[str, Any]
    embed_config: dict[str, Any]
    warnings: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks,
            "n_documents": self.n_documents,
            "embedder_id": self.embedder_id,
            "dim": self.dim,
            "store_backend": self.store_backend,
            "chunk_config": dict(self.chunk_config),
            "embed_config": dict(self.embed_config),
            "warnings": list(self.warnings),
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class RetrieveResult:
    """Dense top-k retrieval for one query."""

    query: str
    k: int
    hits: tuple[Hit, ...]
    embedder_id: str
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "n_hits": len(self.hits),
            "hits": [h.to_dict() for h in self.hits],
            "embedder_id": self.embedder_id,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class RagEvalResult:
    """Retrieval quality metrics on gold qrels."""

    n_queries: int
    k: int
    recall_at_k: float
    mrr: float
    per_query: tuple[dict[str, Any], ...] = ()
    relevance_mode: str = "document"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "relevance_mode": self.relevance_mode,
            "per_query": list(self.per_query),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
