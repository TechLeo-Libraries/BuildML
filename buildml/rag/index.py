"""Build and update a dense vector index from a corpus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rag.chunk import chunk_documents
from buildml.rag.corpus import corpus_from_documents, refuse_eval_only_index
from buildml.rag.embed import Embedder, EmbedFn, resolve_embedder
from buildml.rag.results import Chunk, ChunkResult, CorpusHandle, Document, IndexResult
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

    def _refresh_doc_count(self) -> None:
        self.n_documents = len({c.doc_id for c in self.store.chunks})

    def _refresh_disclosures(self, *, note: str | None = None) -> None:
        notes = [
            f"embedder_id={self.embed_config.embedder_id}",
            f"dim={self.embed_config.dim}",
            f"store_backend={self.index_config.store_backend}",
            f"n_chunks={len(self.store.chunks)}",
            f"n_documents={self.n_documents}",
        ]
        if self.embed_config.embedder_id == HASHING_EMBEDDER_ID:
            notes.append(
                "Default hashing embedder is lexical/hashed, not a semantic sentence model."
            )
        if note:
            notes.append(note)
        self.disclosures = tuple(notes)

    def delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> IndexResult:
        """Remove chunks by ``chunk_id`` and/or parent ``doc_id`` without full rebuild.

        Dense rows for survivors are retained; embeddings for deleted rows are dropped.
        """
        if not chunk_ids and not doc_ids:
            raise ValidationError("rag delete requires chunk_ids and/or doc_ids.")
        before = len(self.store.chunks)
        self.store = self.store.without_ids(chunk_ids=chunk_ids, doc_ids=doc_ids)
        removed = before - len(self.store.chunks)
        self._refresh_doc_count()
        self._refresh_disclosures(note=f"deleted_chunks={removed}")
        return self.to_index_result()

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> IndexResult:
        """Insert or replace chunks by ``chunk_id``, re-embedding only new/changed rows.

        Existing embeddings for untouched chunk ids are preserved. Replaced ids are
        re-encoded with the index embedder.
        """
        incoming = list(chunks)
        if not incoming:
            raise ValidationError("upsert_chunks requires at least one chunk.")
        by_id = {c.chunk_id: (i, c) for i, c in enumerate(self.store.chunks)}
        emb = (
            np.asarray(self.store.embeddings, dtype=np.float32)
            if self.store.embeddings.size
            else np.zeros((0, self.embed_config.dim), dtype=np.float32)
        )
        kept_chunks: list[Chunk] = list(self.store.chunks)
        kept_emb_rows: list[np.ndarray] = (
            [emb[i] for i in range(emb.shape[0])] if emb.shape[0] else []
        )

        to_encode: list[Chunk] = []
        replace_ids: set[str] = set()
        for chunk in incoming:
            if chunk.chunk_id in by_id:
                replace_ids.add(chunk.chunk_id)
            to_encode.append(chunk)

        if replace_ids:
            kept_chunks = []
            next_emb_rows: list[np.ndarray] = []
            for c, row in zip(self.store.chunks, kept_emb_rows, strict=True):
                if c.chunk_id in replace_ids:
                    continue
                kept_chunks.append(c)
                next_emb_rows.append(row)
            kept_emb_rows = next_emb_rows

        matrix_new = self.embedder.encode([c.text for c in to_encode])
        if matrix_new.shape[1] != self.embed_config.dim:
            raise ValidationError(
                f"Upsert embed dim {matrix_new.shape[1]} != index dim {self.embed_config.dim}."
            )
        # L2-normalize new rows to match store convention.
        norms = np.linalg.norm(matrix_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        matrix_new = matrix_new / norms

        final_chunks = kept_chunks + to_encode
        if kept_emb_rows:
            final_emb = np.vstack([np.stack(kept_emb_rows, axis=0), matrix_new])
        else:
            final_emb = matrix_new
        self.store = NumpyCosineStore(
            chunks=tuple(final_chunks),
            embeddings=np.asarray(final_emb, dtype=np.float32),
            dim=self.embed_config.dim,
        )
        self._refresh_doc_count()
        self._refresh_disclosures(
            note=f"upserted_chunks={len(to_encode)}; replaced={len(replace_ids)}"
        )
        return self.to_index_result()

    def upsert_documents(
        self,
        documents: Sequence[Document | Mapping[str, Any] | str],
        *,
        chunk: bool = True,
    ) -> IndexResult:
        """Chunk (optional) and upsert documents into this index."""
        corpus = corpus_from_documents(documents)
        refuse_eval_only_index(corpus)
        if chunk:
            chunked = chunk_documents(corpus, config=self.chunk_config)
            return self.upsert_chunks(chunked.chunks)
        # Treat each document as a single chunk when chunk=False.
        synthetic = [
            Chunk(
                chunk_id=f"{d.doc_id}::c0",
                doc_id=d.doc_id,
                text=d.text,
                start_char=0,
                end_char=len(d.text),
                metadata=dict(d.metadata),
            )
            for d in corpus.documents
        ]
        return self.upsert_chunks(synthetic)


def build_index(
    corpus: CorpusHandle,
    *,
    chunk_config: ChunkConfig | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder: Embedder | EmbedFn | str | None = "auto",
    chunks: ChunkResult | Sequence[Chunk] | None = None,
    device: str | None = None,
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

    resolved, embed_cfg = resolve_embedder(embedder, device=device)
    texts = [c.text for c in chunk_list]
    matrix = resolved.encode(texts)
    store = NumpyCosineStore.build(chunk_list, matrix)
    # Align recorded dim with actual matrix.
    embed_cfg = EmbedConfig(
        embedder_id=embed_cfg.embedder_id,
        dim=int(matrix.shape[1]),
        backend=embed_cfg.backend,
        model_name=embed_cfg.model_name,
        device=embed_cfg.device or device,
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
    if embed_cfg.device:
        disclosures.append(f"embed_device={embed_cfg.device}")
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
