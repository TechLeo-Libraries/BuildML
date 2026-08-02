"""Typed configuration for the RAG retrieve path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_EMBED_DIM = 384
DEFAULT_TOP_K = 5
HASHING_EMBEDDER_ID = "buildml.hashing_embed.v1"
DEFAULT_STORE_BACKEND = "numpy_cosine"
DEFAULT_RETRIEVE_MODE: Literal["dense", "bm25", "hybrid"] = "dense"
DEFAULT_FUSION: Literal["rrf", "weighted"] = "rrf"
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_RERANK_CANDIDATES = 20

RetrieveMode = Literal["dense", "bm25", "hybrid"]
FusionMethod = Literal["rrf", "weighted"]
RelevanceMode = Literal["document", "chunk"]


@dataclass(slots=True)
class ChunkConfig:
    """Fixed-size character chunking with overlap."""

    size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChunkConfig:
        return cls(
            size=int(payload.get("size", DEFAULT_CHUNK_SIZE)),
            overlap=int(payload.get("overlap", DEFAULT_CHUNK_OVERLAP)),
        )


@dataclass(slots=True)
class EmbedConfig:
    """Embedding backend knobs recorded in the RAG bundle."""

    embedder_id: str = HASHING_EMBEDDER_ID
    dim: int = DEFAULT_EMBED_DIM
    backend: Literal["hashing", "sentence-transformers", "callable"] = "hashing"
    model_name: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbedConfig:
        return cls(
            embedder_id=str(payload.get("embedder_id") or HASHING_EMBEDDER_ID),
            dim=int(payload.get("dim") or DEFAULT_EMBED_DIM),
            backend=payload.get("backend") or "hashing",
            model_name=payload.get("model_name"),
            device=payload.get("device"),
        )


@dataclass(slots=True)
class IndexConfig:
    """Vector index construction knobs."""

    store_backend: str = DEFAULT_STORE_BACKEND

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexConfig:
        return cls(store_backend=str(payload.get("store_backend") or DEFAULT_STORE_BACKEND))


@dataclass(slots=True)
class RetrieveConfig:
    """Retrieval knobs for dense, BM25, and hybrid modes.

    Defaults
    --------
    - ``mode="dense"`` — cosine top-k over the NumPy store (M1 behavior).
    - ``fusion="rrf"`` with ``rrf_k=60`` when ``mode="hybrid"``.
    - ``rerank=False`` — no cross-encoder pass unless explicitly requested.
    - ``filters=None`` — no metadata equality filter.
    """

    k: int = DEFAULT_TOP_K
    mode: RetrieveMode = DEFAULT_RETRIEVE_MODE
    fusion: FusionMethod = DEFAULT_FUSION
    rrf_k: int = DEFAULT_RRF_K
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    filters: dict[str, Any] | None = None
    rerank: bool | str = False
    rerank_model: str | None = None
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filters"] = None if self.filters is None else dict(self.filters)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrieveConfig:
        filters = payload.get("filters")
        return cls(
            k=int(payload.get("k") or DEFAULT_TOP_K),
            mode=payload.get("mode") or DEFAULT_RETRIEVE_MODE,
            fusion=payload.get("fusion") or DEFAULT_FUSION,
            rrf_k=int(payload.get("rrf_k") or DEFAULT_RRF_K),
            dense_weight=float(payload.get("dense_weight", DEFAULT_DENSE_WEIGHT)),
            bm25_k1=float(payload.get("bm25_k1", DEFAULT_BM25_K1)),
            bm25_b=float(payload.get("bm25_b", DEFAULT_BM25_B)),
            filters=None if filters is None else dict(filters),
            rerank=payload.get("rerank", False),
            rerank_model=payload.get("rerank_model"),
            rerank_candidates=int(
                payload.get("rerank_candidates") or DEFAULT_RERANK_CANDIDATES
            ),
        )


@dataclass(slots=True)
class EvalConfig:
    """Retrieval evaluation knobs."""

    k: int = DEFAULT_TOP_K
    relevance_mode: RelevanceMode = "document"
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "relevance_mode": self.relevance_mode,
            "retrieve": self.retrieve.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalConfig:
        retrieve_payload = payload.get("retrieve") or {}
        return cls(
            k=int(payload.get("k") or DEFAULT_TOP_K),
            relevance_mode=payload.get("relevance_mode") or "document",
            retrieve=RetrieveConfig.from_dict(retrieve_payload),
        )


DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_GENERATE_TEMPERATURE = 0.0


@dataclass(slots=True)
class GenerateConfig:
    """Grounded generation knobs for :func:`buildml.rag.generate.generate_grounded`."""

    k: int = DEFAULT_TOP_K
    max_tokens: int | None = None
    temperature: float = DEFAULT_GENERATE_TEMPERATURE
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    system_template: str | None = None
    user_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GenerateConfig:
        return cls(
            k=int(payload.get("k") or DEFAULT_TOP_K),
            max_tokens=payload.get("max_tokens"),
            temperature=float(payload.get("temperature", DEFAULT_GENERATE_TEMPERATURE)),
            max_context_chars=int(
                payload.get("max_context_chars") or DEFAULT_MAX_CONTEXT_CHARS
            ),
            system_template=payload.get("system_template"),
            user_template=payload.get("user_template"),
        )
