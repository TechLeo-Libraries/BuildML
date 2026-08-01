"""Vector store protocol and NumPy cosine default."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rag.results import Chunk, Hit


class VectorStore(Protocol):
    """Minimal dense store: build, query, expose embeddings/chunks."""

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    dim: int

    def query(self, vector: np.ndarray, *, k: int) -> list[Hit]: ...


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

    def query(self, vector: np.ndarray, *, k: int) -> list[Hit]:
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
        top_k = min(k, scores.shape[0])
        # argpartition then sort the shortlist for stable ranking
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
