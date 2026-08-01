"""Deterministic document chunking."""

from __future__ import annotations

from collections.abc import Sequence

from buildml.core.errors import ValidationError
from buildml.rag.results import Chunk, ChunkResult, CorpusHandle, Document
from buildml.rag.types import ChunkConfig


def _chunk_text(text: str, *, size: int, overlap: int) -> list[tuple[int, int, str]]:
    if size <= 0:
        raise ValidationError(f"chunk size must be positive; got {size}")
    if overlap < 0:
        raise ValidationError(f"chunk overlap must be >= 0; got {overlap}")
    if overlap >= size:
        raise ValidationError(
            f"chunk overlap ({overlap}) must be smaller than size ({size})"
        )
    if not text:
        return [(0, 0, "")]
    step = size - overlap
    spans: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        spans.append((start, end, text[start:end]))
        if end >= n:
            break
        start += step
    return spans


def chunk_documents(
    documents: Sequence[Document] | CorpusHandle,
    *,
    config: ChunkConfig | None = None,
    size: int | None = None,
    overlap: int | None = None,
) -> ChunkResult:
    """Split documents into overlapping character chunks with stable ids.

    Chunk ids are ``{doc_id}::c{ordinal}`` so resume/update paths stay deterministic
    for a fixed corpus and config.
    """
    if isinstance(documents, CorpusHandle):
        docs = list(documents.documents)
    else:
        docs = list(documents)
    if not docs:
        raise ValidationError("No documents to chunk.")
    cfg = config or ChunkConfig()
    if size is not None:
        cfg = ChunkConfig(size=size, overlap=cfg.overlap if overlap is None else overlap)
    elif overlap is not None:
        cfg = ChunkConfig(size=cfg.size, overlap=overlap)

    chunks: list[Chunk] = []
    for doc in docs:
        spans = _chunk_text(doc.text, size=cfg.size, overlap=cfg.overlap)
        for ordinal, (start, end, piece) in enumerate(spans):
            chunk_id = f"{doc.doc_id}::c{ordinal}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=piece,
                    start_char=start,
                    end_char=end,
                    metadata=dict(doc.metadata),
                )
            )
    return ChunkResult(
        chunks=tuple(chunks),
        config=cfg.to_dict(),
        n_documents=len(docs),
    )
