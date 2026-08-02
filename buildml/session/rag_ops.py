"""Thin Session facades over buildml.rag (no new RAG depth)."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def rag_ingest_corpus(
    session,
    source: str | Path | Sequence[Any] | None = None,
    *,
    text_column: str | None = None,
    id_column: str | None = None,
    glob: str = "*.txt",
    encoding: str = "utf-8",
    role: Literal['index', 'eval_only'] = "index",
) -> Session:
    """Load a text corpus for the RAG path (requires ``buildml[rag]``).

    Provide a file/directory ``source``, an in-memory document sequence, or
    ``text_column`` to bridge the current Session frame. Never silently
    indexes every column.

    Delegates to :mod:`buildml.rag.corpus`. Distinct from classical ingest.
    """
    from buildml.rag.corpus import corpus_from_documents, corpus_from_frame, load_text_corpus
    from buildml.rag.extras import require_rag_stack

    require_rag_stack(feature="RAG corpus ingest")
    if text_column is not None:
        if session._dataset is None:
            raise ValidationError(
                "text_column requires an attached dataset. Call Session.ingest(...) first or pass source= documents/path."
            )
        corpus = corpus_from_frame(
            session._dataset.frame,
            text_column=text_column,
            id_column=id_column,
            role=role,
            source=f"session[{text_column}]",
        )
    elif source is None:
        raise ValidationError(
            "rag_ingest_corpus requires source= (path or documents) or text_column=."
        )
    elif isinstance(source, (str, Path)):
        corpus = load_text_corpus(source, glob=glob, encoding=encoding, role=role)
    else:
        corpus = corpus_from_documents(source, source="memory", default_role=role)
    session._rag_corpus = corpus
    session._rag_chunks = None
    session._rag_index = None
    session._rag_index_result = None
    session._rag_retrieve_result = None
    session._rag_eval_result = None
    session._record(
        "rag_ingest_corpus",
        {"source": corpus.source, "role": role, "text_column": text_column, "id_column": id_column},
        result_summary=corpus.to_dict(),
    )
    return session


def rag_chunk(session, *, size: int = 512, overlap: int = 64) -> Session:
    """Chunk the active RAG corpus with size + overlap (requires ``buildml[rag]``)."""
    from buildml.rag.chunk import chunk_documents
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.types import ChunkConfig

    require_rag_stack(feature="RAG chunking")
    if session._rag_corpus is None:
        raise ValidationError("No RAG corpus. Call rag_ingest_corpus(...) first.")
    result = chunk_documents(session._rag_corpus, config=ChunkConfig(size=size, overlap=overlap))
    session._rag_chunks = result
    session._record(
        "rag_chunk", {"size": size, "overlap": overlap}, result_summary=result.to_dict()
    )
    return session


def rag_embed_and_index(
    session,
    *,
    embedder: Any | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    device: str | None = None,
) -> Session:
    """Embed chunks and build the default NumPy cosine index (requires ``buildml[rag]``).

    Refuses corpora that contain ``eval_only`` documents (:class:`LeakageError`).
    Default embedder is ``buildml.hashing_embed.v1`` (lexical/hashed, not semantic).
    ``device`` applies to sentence-transformer backends; hashing stays CPU-only.
    """
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.index import build_index
    from buildml.rag.results import ChunkResult

    require_rag_stack(feature="RAG embed and index")
    if session._rag_corpus is None:
        raise ValidationError("No RAG corpus. Call rag_ingest_corpus(...) first.")
    index = build_index(
        session._rag_corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedder=embedder,
        chunks=session._rag_chunks,
        device=device,
    )
    session._rag_index = index
    session._rag_index_result = index.to_index_result()
    session._rag_chunks = ChunkResult(
        chunks=index.chunks, config=index.chunk_config.to_dict(), n_documents=index.n_documents
    )
    session._record(
        "rag_embed_and_index",
        {
            "embedder_id": index.embed_config.embedder_id,
            "dim": index.embed_config.dim,
            "store_backend": index.index_config.store_backend,
            "device": index.embed_config.device,
        },
        result_summary=session._rag_index_result.to_dict(),
        warnings=tuple(index.warnings),
    )
    return session


def rag_retrieve(
    session,
    query: str,
    *,
    k: int = 5,
    mode: str | None = None,
    fusion: str | None = None,
    filters: dict[str, Any] | None = None,
    rerank: bool | str | None = None,
    config: Any | None = None,
) -> Any:
    """Retrieve ranked chunks (dense / BM25 / hybrid) against the active RAG index.

    Defaults: ``mode="dense"``, no metadata filters, ``rerank=False``. Hybrid
    defaults to RRF (``rrf_k=60``). Cross-encoder rerank requires ``buildml[rag]``.
    """
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.retrieve import retrieve
    from buildml.rag.types import RetrieveConfig

    require_rag_stack(feature="RAG retrieve")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    cfg = config if config is not None else RetrieveConfig(k=k, mode=mode or "dense")
    result = retrieve(
        session._rag_index,
        query,
        k=k,
        config=cfg,
        mode=mode,
        filters=filters,
        rerank=rerank,
        fusion=fusion,
    )
    session._rag_retrieve_result = result
    session._record(
        "rag_retrieve",
        {
            "query": query,
            "k": k,
            "mode": result.mode,
            "fusion": result.fusion,
            "rerank": result.rerank,
            "filters": result.filters,
        },
        result_summary=result.to_dict(),
    )
    return result


def rag_evaluate(
    session,
    qrels: Any,
    *,
    k: int = 5,
    relevance_mode: str = "document",
    mode: str | None = None,
    retrieve_config: Any | None = None,
) -> Any:
    """Score retrieval with gold qrels (recall@k, MRR, nDCG@k, hit-rate@k).

    ``relevance_mode="document"`` (default) scores parent ``doc_id`` hits;
    ``"chunk"`` scores ``chunk_id`` labels. Requires ``buildml[rag]``.
    """
    from buildml.rag.evaluate import evaluate_retrieval
    from buildml.rag.extras import require_rag_stack

    require_rag_stack(feature="RAG evaluate")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    result = evaluate_retrieval(
        session._rag_index,
        qrels,
        k=k,
        relevance_mode=relevance_mode,
        retrieve_config=retrieve_config,
        mode=mode,
    )
    session._rag_eval_result = result
    session._record(
        "rag_evaluate",
        {
            "k": k,
            "n_queries": result.n_queries,
            "relevance_mode": result.relevance_mode,
            "retrieve_mode": result.retrieve_mode,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def rag_upsert(
    session,
    documents: Sequence[Any] | None = None,
    *,
    chunks: Sequence[Any] | None = None,
    chunk: bool = True,
) -> Session:
    """Upsert documents or chunks into the active RAG index without a full rebuild.

    Replaces existing ``chunk_id`` rows and re-embeds only new/changed text.
    """
    from buildml.rag.extras import require_rag_stack

    require_rag_stack(feature="RAG upsert")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if chunks is not None:
        result = session._rag_index.upsert_chunks(chunks)
    elif documents is not None:
        result = session._rag_index.upsert_documents(documents, chunk=chunk)
    else:
        raise ValidationError("rag_upsert requires documents= or chunks=.")
    session._rag_index_result = result
    from buildml.rag.results import ChunkResult

    session._rag_chunks = ChunkResult(
        chunks=session._rag_index.chunks,
        config=session._rag_index.chunk_config.to_dict(),
        n_documents=session._rag_index.n_documents,
    )
    session._record(
        "rag_upsert",
        {"n_chunks": result.n_chunks, "n_documents": result.n_documents, "chunk": chunk},
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return session


def rag_delete(
    session, *, chunk_ids: Sequence[str] | None = None, doc_ids: Sequence[str] | None = None
) -> Session:
    """Delete chunks by id and/or parent document id from the active RAG index."""
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.results import ChunkResult

    require_rag_stack(feature="RAG delete")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    result = session._rag_index.delete(chunk_ids=chunk_ids, doc_ids=doc_ids)
    session._rag_index_result = result
    session._rag_chunks = ChunkResult(
        chunks=session._rag_index.chunks,
        config=session._rag_index.chunk_config.to_dict(),
        n_documents=session._rag_index.n_documents,
    )
    session._record(
        "rag_delete",
        {
            "chunk_ids": list(chunk_ids or ()),
            "doc_ids": list(doc_ids or ()),
            "n_chunks": result.n_chunks,
        },
        result_summary=result.to_dict(),
    )
    return session


def save_rag_bundle(session, path: str | Path) -> Path:
    """Persist the active RAG index as ``buildml.rag_bundle.v1``.

    Distinct from Session checkpoints and Torch trainer bundles.
    See :data:`buildml.rag.checkpoint.CHECKPOINT_BOUNDARY`.
    """
    from buildml.rag.checkpoint import save_rag_bundle
    from buildml.rag.extras import require_rag_stack

    require_rag_stack(feature="RAG bundle save")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    destination = save_rag_bundle(path, session._rag_index, eval_result=session._rag_eval_result)
    session._record("save_rag_bundle", {"path": str(destination)})
    return destination


def load_rag_bundle(session, path: str | Path) -> Session:
    """Load a RAG bundle into this Session (requires ``buildml[rag]``)."""
    from buildml.rag.checkpoint import load_rag_bundle
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.results import ChunkResult

    require_rag_stack(feature="RAG bundle load")
    index = load_rag_bundle(path)
    session._rag_index = index
    session._rag_index_result = index.to_index_result()
    session._rag_chunks = ChunkResult(
        chunks=index.chunks, config=index.chunk_config.to_dict(), n_documents=index.n_documents
    )
    session._record(
        "load_rag_bundle", {"path": str(path)}, result_summary=session._rag_index_result.to_dict()
    )
    return session
