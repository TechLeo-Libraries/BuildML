"""Typed configuration for the RAG retrieve path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_EMBED_DIM = 384
DEFAULT_TOP_K = 5
HASHING_EMBEDDER_ID = "buildml.hashing_embed.v1"
DEFAULT_STORE_BACKEND = "numpy_cosine"


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbedConfig:
        return cls(
            embedder_id=str(payload.get("embedder_id") or HASHING_EMBEDDER_ID),
            dim=int(payload.get("dim") or DEFAULT_EMBED_DIM),
            backend=payload.get("backend") or "hashing",
            model_name=payload.get("model_name"),
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
    """Dense retrieval knobs."""

    k: int = DEFAULT_TOP_K

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrieveConfig:
        return cls(k=int(payload.get("k") or DEFAULT_TOP_K))
