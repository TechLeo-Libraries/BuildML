"""Thin Session facades over buildml.rag (no new RAG depth)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.session._imports import (
    ValidationError,
)


def rag_ingest_corpus(
    session,
    source: str | Path | Sequence[Any] | None = None,
    *,
    text_column: str | None = None,
    id_column: str | None = None,
    glob: str = "*.txt",
    encoding: str = "utf-8",
    role: Literal['index', 'eval_only'] = "index",
) -> "Session":
    """Load a text corpus for the RAG path (requires ``buildml[rag]``).

    Provide a file/directory ``source``, an in-memory document sequence, or
    ``text_column`` to bridge the current Session frame. Never silently
    indexes every column. Delegates to :mod:`buildml.rag.corpus`. Distinct
    from classical ingest.

    Parameters
    ----------
    session:
        Active Session to attach the corpus to.
    source:
        Optional path, directory, or in-memory document sequence.
    text_column:
        Optional Session frame column to ingest as documents.
    id_column:
        Optional document id column when using ``text_column``.
    glob:
        Glob pattern when ``source`` is a directory (``*.txt`` by default).
    encoding:
        Text encoding for file-based sources.
    role:
        Corpus role (``index`` or ``eval_only``).

    Returns
    -------
    Session
        ``session`` with RAG corpus attached for chaining.

    Raises
    ------
    ValidationError
        When inputs are missing or ``text_column`` is used without a dataset.
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
    session._rag_generate_result = None
    session._record(
        "rag_ingest_corpus",
        {"source": corpus.source, "role": role, "text_column": text_column, "id_column": id_column},
        result_summary=corpus.to_dict(),
    )
    return cast("Session", session)
def rag_chunk(
    session,
    *,
    size: int = 512,
    overlap: int = 64,
    strategy: str = "fixed",
) -> "Session":
    """Chunk the active RAG corpus (fixed or recursive strategy).

    ``strategy="recursive"`` splits on paragraph/line/sentence boundaries before
    applying size/overlap (LangChain/LlamaIndex parity). Requires ``buildml[rag]``.
    Delegates to :func:`buildml.rag.chunk.chunk_documents`.

    Parameters
    ----------
    session:
        Active Session with a corpus from :func:`rag_ingest_corpus`.
    size:
        Target chunk size in characters or tokens.
    overlap:
        Overlap between consecutive chunks.
    strategy:
        Chunking strategy (``fixed`` or ``recursive``).

    Returns
    -------
    Session
        ``session`` with chunk result attached for chaining.

    Raises
    ------
    ValidationError
        When no RAG corpus exists on the Session.
    """
    from buildml.rag.chunk import chunk_documents
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.types import ChunkConfig

    require_rag_stack(feature="RAG chunking")
    if session._rag_corpus is None:
        raise ValidationError("No RAG corpus. Call rag_ingest_corpus(...) first.")
    result = chunk_documents(
        session._rag_corpus,
        config=ChunkConfig(size=size, overlap=overlap, strategy=strategy),  # type: ignore[arg-type]
    )
    session._rag_chunks = result
    session._record(
        "rag_chunk",
        {"size": size, "overlap": overlap, "strategy": strategy},
        result_summary=result.to_dict(),
    )
    return cast("Session", session)
def rag_embed_and_index(
    session,
    *,
    embedder: Any | None = "auto",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunk_strategy: str | None = None,
    device: str | None = None,
) -> "Session":
    """Embed chunks and build the default NumPy cosine index (requires ``buildml[rag]``).

    Refuses corpora that contain ``eval_only`` documents (:class:`LeakageError`).
    Default embedder is ``auto``: sentence-transformers when ``buildml[rag]`` is
    installed, else explicit hashing fallback with disclosure.
    Pass ``embedder="hashing"`` for deterministic CI / lexical-only paths.
    ``device`` applies to sentence-transformer backends; hashing stays CPU-only.
    Delegates to :func:`buildml.rag.index.build_index`.

    Parameters
    ----------
    session:
        Active Session with a corpus from :func:`rag_ingest_corpus`.
    embedder:
        Embedder id or ``auto`` / ``hashing`` sentinel.
    chunk_size:
        Optional chunk size override before indexing.
    chunk_overlap:
        Optional chunk overlap override before indexing.
    chunk_strategy:
        Optional chunk strategy override before indexing.
    device:
        Optional device for sentence-transformer embedders.

    Returns
    -------
    Session
        ``session`` with RAG index attached for chaining.

    Raises
    ------
    ValidationError
        When no RAG corpus exists on the Session.
    """
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.index import build_index
    from buildml.rag.results import ChunkResult

    require_rag_stack(feature="RAG embed and index")
    if session._rag_corpus is None:
        raise ValidationError("No RAG corpus. Call rag_ingest_corpus(...) first.")
    from buildml.rag.types import ChunkConfig

    chunk_cfg = None
    if chunk_strategy is not None:
        chunk_cfg = ChunkConfig(
            size=chunk_size or 512,
            overlap=chunk_overlap or 64,
            strategy=chunk_strategy,  # type: ignore[arg-type]
        )
    index = build_index(
        session._rag_corpus,
        chunk_config=chunk_cfg,
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
    return cast("Session", session)
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

    Defaults: ``mode="hybrid"`` (BM25 + dense RRF) when ``buildml[rag]`` is installed,
    else ``mode="dense"``. Metadata filters and cross-encoder rerank are opt-in.
    Delegates to :func:`buildml.rag.retrieve.retrieve`.

    Parameters
    ----------
    session:
        Active Session with a RAG index from :func:`rag_embed_and_index`.
    query:
        Natural-language query string.
    k:
        Number of chunks to retrieve.
    mode:
        Optional retrieve mode override (``dense``, ``bm25``, ``hybrid``).
    fusion:
        Optional fusion strategy for hybrid retrieval.
    filters:
        Optional metadata filters applied before ranking.
    rerank:
        Optional reranker toggle or model identifier.
    config:
        Optional full :class:`~buildml.rag.types.RetrieveConfig` override.

    Returns
    -------
    RetrieveResult
        Ranked chunks, scores, and retrieve provenance.

    Raises
    ------
    ValidationError
        When no RAG index exists on the Session.
    """
    from buildml.rag.defaults import default_retrieve_config
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.retrieve import retrieve
    from buildml.rag.types import RetrieveConfig

    require_rag_stack(feature="RAG retrieve")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    cfg = config if config is not None else default_retrieve_config(k=k)
    if mode is not None:
        cfg = RetrieveConfig(
            k=int(k),
            mode=mode,  # type: ignore[arg-type]
            fusion=cfg.fusion,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            filters=cfg.filters,
            rerank=cfg.rerank if rerank is None else rerank,
            rerank_model=cfg.rerank_model,
            rerank_candidates=cfg.rerank_candidates,
        )
    result = retrieve(
        session._rag_index,
        query,
        k=k,
        config=cfg,
        mode=cast(Any, mode),
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


def rag_generate(
    session,
    query: str,
    *,
    k: int = 5,
    provider: Any | None = None,
    mode: str | None = None,
    fusion: str | None = None,
    filters: dict[str, Any] | None = None,
    rerank: bool | str | None = None,
    retrieve_config: Any | None = None,
    config: Any | None = None,
    use_last_retrieve: bool = False,
) -> Any:
    """Retrieve (unless reusing the last retrieve) and generate a grounded answer.

    Requires an active RAG index and a chat provider. When ``provider`` is
    omitted, reuses ``Session.ai_configure``'s provider. For offline CI, pass
    :class:`buildml.rag.generate.EchoGroundedProvider` or a
    :class:`buildml.ai.provider.MockProvider`. Delegates to
    :func:`buildml.rag.generate.generate_grounded`.

    Parameters
    ----------
    session:
        Active Session with a RAG index from :func:`rag_embed_and_index`.
    query:
        Natural-language question to answer.
    k:
        Number of chunks to retrieve for grounding.
    provider:
        Optional chat provider; uses Session AI provider when omitted.
    mode:
        Optional retrieve mode passed through to retrieval.
    fusion:
        Optional fusion strategy for hybrid retrieval.
    filters:
        Optional metadata filters for retrieval.
    rerank:
        Optional reranker toggle or model identifier.
    retrieve_config:
        Optional retrieve configuration override.
    config:
        Optional :class:`~buildml.rag.types.GenerateConfig` override.
    use_last_retrieve:
        When True, reuse the prior :func:`rag_retrieve` result.

    Returns
    -------
    GenerateResult
        Answer text, citations (source ids / chunk / doc), and retrieve provenance.

    Raises
    ------
    ValidationError
        When no RAG index or provider is available, or reuse is requested
        without a prior retrieve result.
    """
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.generate import generate_grounded
    from buildml.rag.types import GenerateConfig

    require_rag_stack(feature="RAG generate")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    resolved_provider = provider
    if resolved_provider is None:
        resolved_provider = getattr(session, "_ai_provider", None)
    if resolved_provider is None:
        raise ValidationError(
            "rag_generate requires a chat provider. Pass provider=... or call "
            "ai_configure(...) first to reuse the Session AI provider."
        )
    retrieve_result = None
    if use_last_retrieve:
        retrieve_result = session._rag_retrieve_result
        if retrieve_result is None:
            raise ValidationError(
                "use_last_retrieve=True requires a prior rag_retrieve(...) result."
            )
    cfg = config if config is not None else GenerateConfig(k=k)
    result = generate_grounded(
        session._rag_index,
        query,
        resolved_provider,
        k=k,
        retrieve_config=retrieve_config,
        mode=mode,
        filters=filters,
        rerank=rerank,
        fusion=fusion,
        config=cfg,
        retrieve_result=retrieve_result,
    )
    if result.retrieve_result is not None:
        session._rag_retrieve_result = result.retrieve_result
    session._rag_generate_result = result
    session._record(
        "rag_generate",
        {
            "query": query,
            "k": k,
            "n_citations": result.n_citations,
            "provider_model": result.provider_model,
            "use_last_retrieve": use_last_retrieve,
        },
        result_summary=result.to_dict(),
        warnings=(),
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
    Delegates to :func:`buildml.rag.evaluate.evaluate_retrieval`.

    Parameters
    ----------
    session:
        Active Session with a RAG index from :func:`rag_embed_and_index`.
    qrels:
        Gold relevance judgments mapping queries to relevant ids.
    k:
        Cutoff k for retrieval metrics.
    relevance_mode:
        Whether qrels label documents or chunks.
    mode:
        Optional retrieve mode override for evaluation queries.
    retrieve_config:
        Optional retrieve configuration override.

    Returns
    -------
    RagEvalResult
        Aggregate retrieval metrics and per-query summaries.

    Raises
    ------
    ValidationError
        When no RAG index exists on the Session.
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
        relevance_mode=cast(Any, relevance_mode),
        retrieve_config=retrieve_config,
        mode=cast(Any, mode),
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
) -> "Session":
    """Upsert documents or chunks into the active RAG index without a full rebuild.

    Replaces existing ``chunk_id`` rows and re-embeds only new/changed text.
    Delegates to the active index object's upsert methods.

    Parameters
    ----------
    session:
        Active Session with a RAG index from :func:`rag_embed_and_index`.
    documents:
        Optional new or updated documents to upsert.
    chunks:
        Optional pre-chunked rows to upsert (mutually exclusive with
        ``documents``).
    chunk:
        When True and ``documents`` is supplied, chunk before upserting.

    Returns
    -------
    Session
        ``session`` with updated index and chunk state attached.

    Raises
    ------
    ValidationError
        When no RAG index exists or neither ``documents`` nor ``chunks`` is
        supplied.
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
    return cast("Session", session)
def rag_delete(
    session, *, chunk_ids: Sequence[str] | None = None, doc_ids: Sequence[str] | None = None
) -> "Session":
    """Delete chunks by id and/or parent document id from the active RAG index.

    Removes matching rows from the in-memory index and refreshes Session chunk
    state without requiring a full rebuild.

    Parameters
    ----------
    session:
        Active Session with a RAG index from :func:`rag_embed_and_index`.
    chunk_ids:
        Optional chunk identifiers to delete.
    doc_ids:
        Optional parent document identifiers whose chunks should be deleted.

    Returns
    -------
    Session
        ``session`` with updated index and chunk state attached.

    Raises
    ------
    ValidationError
        When no RAG index exists on the Session.
    """
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
    return cast("Session", session)
def save_rag_bundle(session, path: str | Path) -> Path:
    """Persist the active RAG index as ``buildml.rag_bundle.v1``.

    Distinct from Session checkpoints and Torch trainer bundles.
    See :data:`buildml.rag.checkpoint.CHECKPOINT_BOUNDARY`.
    Delegates to :func:`buildml.rag.checkpoint.save_rag_bundle`.

    Parameters
    ----------
    session:
        Active Session with a RAG index from :func:`rag_embed_and_index`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no RAG index exists on the Session.
    """
    from buildml.rag.checkpoint import save_rag_bundle
    from buildml.rag.extras import require_rag_stack

    require_rag_stack(feature="RAG bundle save")
    if session._rag_index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    destination = save_rag_bundle(path, session._rag_index, eval_result=session._rag_eval_result)
    session._record("save_rag_bundle", {"path": str(destination)})
    return destination


def load_rag_bundle(session, path: str | Path) -> "Session":
    """Load a RAG bundle into this Session (requires ``buildml[rag]``).

    Delegates to :func:`buildml.rag.checkpoint.load_rag_bundle` and restores
    index, chunk, and index-result state on the Session. RAG bundles use JSONL
    and NumPy arrays: not joblib/pickle: so no ``trusted`` gate applies.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded RAG index.
    path:
        Path to a ``buildml.rag_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with RAG index attached for chaining.
    """
    from buildml.rag.checkpoint import load_rag_bundle as _load_rag_bundle
    from buildml.rag.extras import require_rag_stack
    from buildml.rag.results import ChunkResult

    require_rag_stack(feature="RAG bundle load")
    index = _load_rag_bundle(path)
    session._rag_index = index
    session._rag_index_result = index.to_index_result()
    session._rag_chunks = ChunkResult(
        chunks=index.chunks, config=index.chunk_config.to_dict(), n_documents=index.n_documents
    )
    session._record(
        "load_rag_bundle", {"path": str(path)}, result_summary=session._rag_index_result.to_dict()
    )
    return cast("Session", session)