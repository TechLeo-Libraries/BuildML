"""Build a dense vector index from a corpus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rag.chunk import chunk_documents
from buildml.rag.corpus import refuse_eval_only_index
from buildml.rag.embed import Embedder, EmbedFn, resolve_embedder
from buildml.rag.results import Chunk, ChunkResult, CorpusHandle, IndexResult
from buildml.rag.store import NumpyCosineStore
from buildml.rag.types import HASHING_EMBEDDER_ID, ChunkConfig, EmbedConfig, IndexConfig


class RagIndex:
    """In-memory RAG index: chunks + embeddings + store + configs."""

    def __init__(
        self,
        *,
        store: NumpyCosineStore,
        embedder: Any,
        embed_config: EmbedConfig,
        chunk_config: ChunkConfig,
        index_config: IndexConfig,
        n_documents: int,
        warnings: tuple[str, ...] = (),
        disclosures: tuple[str, ...] = (),
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.embed_config = embed_config
        self.chunk_config = chunk_config
        self.index_config = index_config
        self.n_documents = n_documents
        self.warnings = warnings
        self.disclosures = disclosures

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self.store.chunks

    @property
    def embeddings(self) -> np.ndarray:
        return self.store.embeddings

    def to_index_result(self) -> IndexResult:
        return IndexResult(
            n_chunks=len(self.store.chunks),
            n_documents=self.n_documents,
            embedder_id=self.embed_config.embedder_id,
            dim=self.embed_config.dim,
            store_backend=self.index_config.store_backend,
            chunk_config=self.chunk_config.to_dict(),
            embed_config=self.embed_config.to_dict(),
            warnings=self.warnings,
            disclosures=self.disclosures,
        )


def build_index(
    corpus: CorpusHandle,
    *,
    chunk_config: ChunkConfig | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder: Embedder | EmbedFn | str | None = None,
    chunks: ChunkResult | Sequence[Chunk] | None = None,
) -> RagIndex:
    """Chunk (optional), embed, and build the default NumPy cosine index.

    Raises :class:`~buildml.core.errors.LeakageError` when the corpus contains
    any ``eval_only`` documents.
    """
    refuse_eval_only_index(corpus)
    cfg = chunk_config or ChunkConfig()
    if chunk_size is not None or chunk_overlap is not None:
        cfg = ChunkConfig(
            size=cfg.size if chunk_size is None else chunk_size,
            overlap=cfg.overlap if chunk_overlap is None else chunk_overlap,
        )
    if chunks is None:
        chunk_result = chunk_documents(corpus, config=cfg)
        chunk_list = list(chunk_result.chunks)
        cfg = ChunkConfig.from_dict(chunk_result.config)
    elif isinstance(chunks, ChunkResult):
        chunk_list = list(chunks.chunks)
        cfg = ChunkConfig.from_dict(chunks.config)
    else:
        chunk_list = list(chunks)
    if not chunk_list:
        raise ValidationError("Cannot build an index from zero chunks.")

    resolved, embed_cfg = resolve_embedder(embedder)
    texts = [c.text for c in chunk_list]
    matrix = resolved.encode(texts)
    store = NumpyCosineStore.build(chunk_list, matrix)
    # Align recorded dim with actual matrix.
    embed_cfg = EmbedConfig(
        embedder_id=embed_cfg.embedder_id,
        dim=int(matrix.shape[1]),
        backend=embed_cfg.backend,
        model_name=embed_cfg.model_name,
    )
    index_cfg = IndexConfig()
    warnings: list[str] = []
    disclosures = [
        f"embedder_id={embed_cfg.embedder_id}",
        f"dim={embed_cfg.dim}",
        f"store_backend={index_cfg.store_backend}",
        f"n_chunks={len(chunk_list)}",
        f"n_documents={corpus.n_documents}",
    ]
    if embed_cfg.embedder_id == HASHING_EMBEDDER_ID:
        disclosures.append(
            "Default hashing embedder is lexical/hashed, not a semantic sentence model."
        )
    return RagIndex(
        store=store,
        embedder=resolved,
        embed_config=embed_cfg,
        chunk_config=cfg,
        index_config=index_cfg,
        n_documents=corpus.n_documents,
        warnings=tuple(warnings),
        disclosures=tuple(disclosures),
    )
