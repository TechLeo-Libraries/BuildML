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
    """Top-k retrieval for one query (dense, BM25, or hybrid)."""

    query: str
    k: int
    hits: tuple[Hit, ...]
    embedder_id: str
    mode: str = "dense"
    fusion: str | None = None
    filters: dict[str, Any] | None = None
    rerank: bool = False
    disclosures: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "n_hits": len(self.hits),
            "hits": [h.to_dict() for h in self.hits],
            "embedder_id": self.embedder_id,
            "mode": self.mode,
            "fusion": self.fusion,
            "filters": None if self.filters is None else dict(self.filters),
            "rerank": self.rerank,
            "disclosures": list(self.disclosures),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RagEvalResult:
    """Retrieval quality metrics on gold qrels."""

    n_queries: int
    k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float = 0.0
    hit_rate_at_k: float = 0.0
    per_query: tuple[dict[str, Any], ...] = ()
    relevance_mode: str = "document"
    retrieve_mode: str = "dense"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "relevance_mode": self.relevance_mode,
            "retrieve_mode": self.retrieve_mode,
            "per_query": list(self.per_query),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RagGenerateEvalResult:
    """Generation-quality metrics over grounded answers (cheap heuristics)."""

    n_queries: int
    mean_faithfulness: float
    mean_answer_relevance: float
    citation_coverage: float
    per_query: tuple[dict[str, Any], ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevance": self.mean_answer_relevance,
            "citation_coverage": self.citation_coverage,
            "per_query": list(self.per_query),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ConfigCompareResult:
    """Side-by-side retrieval metrics for multiple configs."""

    rows: tuple[dict[str, Any], ...]
    k: int
    relevance_mode: str
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": list(self.rows),
            "k": self.k,
            "relevance_mode": self.relevance_mode,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class Citation:
    """One grounded citation tied to a retrieved chunk."""

    source_id: int
    chunk_id: str
    doc_id: str
    score: float
    text: str
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "text": self.text,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class FaithfulnessReport:
    """Cheap grounding / faithfulness heuristics for a generated answer."""

    citation_marker_coverage: float
    cited_source_ids: tuple[int, ...]
    missing_source_ids: tuple[int, ...]
    answer_context_token_overlap: float
    grounded: bool
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """Scalar in [0, 1] combining citation coverage and token overlap."""
        return float(
            0.5 * self.citation_marker_coverage + 0.5 * self.answer_context_token_overlap
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "citation_marker_coverage": self.citation_marker_coverage,
            "cited_source_ids": list(self.cited_source_ids),
            "missing_source_ids": list(self.missing_source_ids),
            "answer_context_token_overlap": self.answer_context_token_overlap,
            "grounded": self.grounded,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
        }


@dataclass(slots=True)
class GenerateResult:
    """Grounded generation answer with citations and retrieve provenance."""

    query: str
    answer: str
    citations: tuple[Citation, ...]
    retrieve_result: RetrieveResult | None = None
    provider_model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    prompt_context: str = ""
    disclosures: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    faithfulness: FaithfulnessReport | None = None

    @property
    def n_citations(self) -> int:
        return len(self.citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "n_citations": self.n_citations,
            "citations": [c.to_dict() for c in self.citations],
            "retrieve": None
            if self.retrieve_result is None
            else self.retrieve_result.to_dict(),
            "provider_model": self.provider_model,
            "usage": dict(self.usage),
            "prompt_context_chars": len(self.prompt_context),
            "disclosures": list(self.disclosures),
            "config": dict(self.config),
            "faithfulness": None if self.faithfulness is None else self.faithfulness.to_dict(),
        }
