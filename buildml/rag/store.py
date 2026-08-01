"""Vector store protocol and NumPy cosine default."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rag.hybrid import match_metadata_filters
from buildml.rag.results import Chunk, Hit


class VectorStore(Protocol):
    """Minimal dense store: build, query, expose embeddings/chunks."""

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    dim: int

    def query(
        self,
        vector: np.ndarray,
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]: ...


@dataclass
class NumpyCosineStore:
    """In-process dense store using L2-normalized cosine via matmul."""

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    dim: int
    backend: str = "numpy_cosine"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        chunks: tuple[Chunk, ...] | list[Chunk],
        embeddings: np.ndarray,
    ) -> NumpyCosineStore:
        chunk_tuple = tuple(chunks)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValidationError(f"embeddings must be 2-D; got shape {matrix.shape}")
        if matrix.shape[0] != len(chunk_tuple):
            raise ValidationError(
                f"embeddings rows ({matrix.shape[0]}) != n_chunks ({len(chunk_tuple)})"
            )
        # Ensure unit rows for cosine via dot product.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        matrix = matrix / norms
        return cls(chunks=chunk_tuple, embeddings=matrix, dim=int(matrix.shape[1]))

    def query(
        self,
        vector: np.ndarray,
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        if k <= 0:
            raise ValidationError(f"k must be positive; got {k}")
        if self.embeddings.shape[0] == 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValidationError(
                f"Query dim {q.shape[0]} does not match index dim {self.dim}"
            )
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        scores = self.embeddings @ q
        if filters:
            mask = np.array(
                [match_metadata_filters(c.metadata, filters) for c in self.chunks],
                dtype=bool,
            )
            if not bool(mask.any()):
                return []
            eligible = np.flatnonzero(mask)
            eligible_scores = scores[eligible]
            top_k = min(k, eligible_scores.shape[0])
            local = np.argpartition(-eligible_scores, top_k - 1)[:top_k]
            local = local[np.argsort(-eligible_scores[local], kind="stable")]
            idx = eligible[local]
        else:
            top_k = min(k, scores.shape[0])
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            idx = idx[np.argsort(-scores[idx], kind="stable")]
        hits: list[Hit] = []
        for rank, i in enumerate(idx, start=1):
            chunk = self.chunks[int(i)]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(scores[int(i)]),
                    text=chunk.text,
                    rank=rank,
                    metadata=dict(chunk.metadata),
                )
            )
        return hits

    def without_ids(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> NumpyCosineStore:
        """Return a store with matching chunk/doc ids removed (no re-embed)."""
        drop_chunks = set(chunk_ids or ())
        drop_docs = set(doc_ids or ())
        keep_idx = [
            i
            for i, c in enumerate(self.chunks)
            if c.chunk_id not in drop_chunks and c.doc_id not in drop_docs
        ]
        if len(keep_idx) == len(self.chunks):
            return self
        if not keep_idx:
            empty = np.zeros((0, self.dim), dtype=np.float32)
            return NumpyCosineStore(chunks=(), embeddings=empty, dim=self.dim)
        kept_chunks = tuple(self.chunks[i] for i in keep_idx)
        kept_emb = self.embeddings[np.asarray(keep_idx, dtype=np.int64)]
        return NumpyCosineStore(chunks=kept_chunks, embeddings=kept_emb, dim=self.dim)
