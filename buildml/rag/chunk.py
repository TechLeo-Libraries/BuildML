"""Deterministic document chunking (fixed and recursive strategies)."""

from __future__ import annotations

from collections.abc import Sequence

from buildml.core.errors import ValidationError
from buildml.rag.results import Chunk, ChunkResult, CorpusHandle, Document
from buildml.rag.types import ChunkConfig, ChunkStrategy


def _chunk_text_fixed(text: str, *, size: int, overlap: int) -> list[tuple[int, int, str]]:
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


def _merge_splits(splits: list[str], *, size: int, overlap: int) -> list[str]:
    """Merge small splits up to ``size`` with ``overlap`` carry."""
    if not splits:
        return []
    merged: list[str] = []
    current = ""
    for piece in splits:
        if not piece:
            continue
        candidate = piece if not current else current + piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            merged.append(current)
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + piece
            else:
                current = piece
        else:
            merged.append(piece[:size])
            current = piece[size:]
            while len(current) > size:
                merged.append(current[:size])
                current = current[size - overlap :] if overlap else current[size:]
    if current:
        merged.append(current)
    return merged


def _split_text_recursive(
    text: str,
    separators: Sequence[str],
    *,
    size: int,
) -> list[str]:
    """LangChain-style recursive split: try coarse separators first."""
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep = separators[0]
    rest = list(separators[1:])
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    out: list[str] = []
    carry = ""
    for i, part in enumerate(parts):
        segment = part if i == 0 else sep + part
        candidate = carry + segment
        if len(candidate) <= size:
            carry = candidate
            continue
        if carry:
            out.extend(_split_text_recursive(carry, rest, size=size))
            carry = segment
        elif len(segment) <= size:
            carry = segment
        else:
            out.extend(_split_text_recursive(segment, rest, size=size))
            carry = ""
    if carry:
        out.extend(_split_text_recursive(carry, rest, size=size))
    return out


def _chunk_text_recursive(
    text: str,
    *,
    size: int,
    overlap: int,
    separators: Sequence[str],
) -> list[tuple[int, int, str]]:
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
    raw_splits = _split_text_recursive(text, separators, size=size)
    pieces = _merge_splits(raw_splits, size=size, overlap=overlap)
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        spans.append((start, end, piece))
        cursor = max(0, end - overlap) if overlap else end
    return spans or [(0, len(text), text)]


def _chunk_text(
    text: str,
    *,
    size: int,
    overlap: int,
    strategy: ChunkStrategy,
    separators: Sequence[str],
) -> list[tuple[int, int, str]]:
    if strategy == "recursive":
        return _chunk_text_recursive(
            text,
            size=size,
            overlap=overlap,
            separators=separators,
        )
    return _chunk_text_fixed(text, size=size, overlap=overlap)


def chunk_documents(
    documents: Sequence[Document] | CorpusHandle,
    *,
    config: ChunkConfig | None = None,
    size: int | None = None,
    overlap: int | None = None,
    strategy: ChunkStrategy | None = None,
) -> ChunkResult:
    """Split documents into overlapping chunks with stable ids.

    Strategies
    ----------
    - ``fixed`` (default): sliding character windows.
    - ``recursive``: separator-aware splits (paragraph → line → sentence → word).

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
        cfg = ChunkConfig(
            size=size,
            overlap=cfg.overlap if overlap is None else overlap,
            strategy=cfg.strategy if strategy is None else strategy,
            separators=cfg.separators,
        )
    elif overlap is not None:
        cfg = ChunkConfig(
            size=cfg.size,
            overlap=overlap,
            strategy=cfg.strategy if strategy is None else strategy,
            separators=cfg.separators,
        )
    elif strategy is not None:
        cfg = ChunkConfig(
            size=cfg.size,
            overlap=cfg.overlap,
            strategy=strategy,
            separators=cfg.separators,
        )

    chunks: list[Chunk] = []
    for doc in docs:
        spans = _chunk_text(
            doc.text,
            size=cfg.size,
            overlap=cfg.overlap,
            strategy=cfg.strategy,
            separators=cfg.separators,
        )
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
