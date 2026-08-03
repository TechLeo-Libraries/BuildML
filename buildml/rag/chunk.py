"""Cut documents into passages, the same way every time.

Chunking decides what a retrievable unit is, and therefore sets a ceiling on
retrieval quality that no amount of tuning downstream can lift. If the sentence
answering a question is split across two chunks, neither chunk answers it.

Two strategies. **Fixed** slides a window of ``size`` characters across the text
at regular steps: predictable, uniform, and indifferent to whether it cuts
through the middle of a word. **Recursive** tries a list of separators from
coarsest to finest and cuts at the largest boundary that fits, so paragraphs
stay whole where they can and only fall back to arbitrary cuts when a single
paragraph is oversized. Recursive is the better default for prose.

Both overlap adjacent chunks, which is the insurance against cutting through the
one sentence that mattered.

Chunk IDs are ``{doc_id}::c{ordinal}``, derived from position rather than
content, so the same corpus and config always produce the same identifiers. That
is what lets an index be rebuilt and still match stored references.

See Also
--------
buildml.rag.types.ChunkConfig : The settings, and how to choose them.
buildml.rag.index : What consumes chunks.
"""

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
    """Cut every document into overlapping passages.

    The step between ingest and indexing, and the one most worth checking before
    going further. Inspect the resulting chunk count against the document count:
    if they are close, the chunk size exceeds your documents and retrieval is
    effectively working at document level.

    Parameters
    ----------
    documents:
        A corpus handle, or a sequence of documents.
    config:
        Chunking settings. Defaults are a reasonable starting point, not a
        tuned choice for your corpus.
    size:
        Chunk length in characters. Overrides ``config``.
    overlap:
        Characters shared between neighbours. Overrides ``config``.
    strategy:
        ``'fixed'`` or ``'recursive'``. Overrides ``config``.

    Returns
    -------
    ChunkResult
        The chunks, with the settings that produced them.

    Raises
    ------
    ValidationError
        If there are no documents, the size is not positive, the overlap is
        negative, or the overlap is not smaller than the size. The last is
        refused because a step of zero or less would never advance.

    Notes
    -----
    **Chunk IDs are positional.** Editing a document changes the content of
    every chunk after the edit while their IDs stay the same, so a partial
    re-index against stored IDs will silently mismatch. Rebuild after edits.

    **Roles are not filtered here.** Passing a mixed corpus chunks the eval-only
    documents too; the indexing path is where they are excluded.

    **The overrides are applied one at a time.** Passing ``size`` and
    ``strategy`` together with a ``config`` follows a precedence chain: pass a
    complete :class:`~buildml.rag.types.ChunkConfig` when setting several.

    **Metadata is copied to every chunk**, so a document with large metadata
    multiplies it by its chunk count.

    Examples
    --------
    Chunk with sentence-aware boundaries::

        result = chunk_documents(corpus, size=1024, overlap=128, strategy="recursive")
        print(result.n_chunks / result.n_documents)

    See Also
    --------
    buildml.rag.types.ChunkConfig : What the settings mean.
    buildml.rag.index.build_index : The next step.
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
