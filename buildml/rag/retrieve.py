"""Find the passages most likely to answer a question.

The step everything downstream depends on. A language model cannot answer from a
passage it was never given, so retrieval quality is a hard ceiling on answer
quality — and when answers are poor, this is where to look first.

The pipeline is: pick a mode, pull candidates, optionally rerank, truncate to
``k``. Reranking is why the candidate pool is wider than ``k`` — a cross-encoder
that only ever saw the top five could not promote the passage that placed
twentieth, which is precisely the kind of rescue it is for.

Every result records what actually ran, including any fallback. A hybrid request
in an environment without the optional dependencies quietly becomes dense, and
that shows up in the result's ``mode`` rather than as an error.

See Also
--------
buildml.rag.types.RetrieveConfig : The settings.
buildml.rag.hybrid : Keyword search and fusion.
buildml.rag.rerank : The cross-encoder pass.
"""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.hybrid import BM25Index, filter_chunks, rrf_fuse, weighted_fuse
from buildml.rag.index import RagIndex
from buildml.rag.results import Hit, RetrieveResult
from buildml.rag.types import RetrieveConfig, RetrieveMode


def _candidate_k(cfg: RetrieveConfig, top_k: int) -> int:
    """How many candidates to pull before optional rerank."""
    if cfg.rerank:
        return max(int(cfg.rerank_candidates), top_k)
    return top_k


def _dense_hits(
    index: RagIndex,
    query: str,
    *,
    k: int,
    filters: dict[str, Any] | None,
) -> list[Hit]:
    vector = index.embedder.encode([query])[0]
    return index.store.query(vector, k=k, filters=filters)


def _bm25_hits(
    index: RagIndex,
    query: str,
    *,
    k: int,
    cfg: RetrieveConfig,
    filters: dict[str, Any] | None,
) -> list[Hit]:
    chunks = filter_chunks(index.chunks, filters)
    if not chunks:
        return []
    bm25 = BM25Index.build(chunks, k1=cfg.bm25_k1, b=cfg.bm25_b)
    return bm25.query(query, k=k)


def _hybrid_hits(
    index: RagIndex,
    query: str,
    *,
    k: int,
    cfg: RetrieveConfig,
    filters: dict[str, Any] | None,
) -> list[Hit]:
    # Pull a wider pool from each channel before fusion.
    pool = max(k * 4, k, int(cfg.rerank_candidates))
    dense = _dense_hits(index, query, k=pool, filters=filters)
    sparse = _bm25_hits(index, query, k=pool, cfg=cfg, filters=filters)
    if cfg.fusion == "weighted":
        return weighted_fuse(
            dense,
            sparse,
            k=k,
            dense_weight=cfg.dense_weight,
        )
    if cfg.fusion == "rrf":
        return rrf_fuse([dense, sparse], k=k, rrf_k=cfg.rrf_k)
    raise ValidationError(
        f"Unknown fusion {cfg.fusion!r}; expected 'rrf' or 'weighted'."
    )


def _apply_rerank(
    query: str,
    hits: list[Hit],
    *,
    k: int,
    cfg: RetrieveConfig,
) -> tuple[list[Hit], tuple[str, ...]]:
    if not cfg.rerank:
        return hits[:k], ()
    from buildml.rag.rerank import resolve_reranker

    reranker = resolve_reranker(cfg.rerank, model_name=cfg.rerank_model)
    if reranker is None:
        return hits[:k], ()
    reranked = reranker.rerank(query, hits, k=k)
    disclosures = (
        f"rerank=cross-encoder:{reranker.model_name}",
        f"rerank_candidates={len(hits)}",
    )
    return reranked, disclosures


def retrieve(
    index: RagIndex,
    query: str,
    *,
    k: int | None = None,
    config: RetrieveConfig | None = None,
    mode: RetrieveMode | None = None,
    filters: dict[str, Any] | None = None,
    rerank: bool | str | None = None,
    fusion: str | None = None,
) -> RetrieveResult:
    """Return the ``k`` passages most likely to answer the query.

    The main retrieval entry point. Defaults come from the environment — hybrid
    where the optional dependencies allow it, dense otherwise — and any argument
    given here overrides them.

    Parameters
    ----------
    index:
        The index to search.
    query:
        The question. Must be non-empty.
    k:
        How many passages to return.
    config:
        Full retrieval settings. Individual arguments override its fields.
    mode:
        ``'dense'``, ``'bm25'``, or ``'hybrid'``.
    filters:
        Metadata equality constraints, applied before scoring.
    rerank:
        Run a cross-encoder over the candidates.
    fusion:
        ``'rrf'`` or ``'weighted'``, for hybrid mode.

    Returns
    -------
    RetrieveResult
        The ranked passages, plus what mode actually ran and why.

    Raises
    ------
    ValidationError
        If there is no index, the query is empty, ``k`` is not positive, or the
        mode or fusion name is unrecognised.

    Notes
    -----
    **Something always comes back.** There is no relevance threshold, so a
    question the corpus cannot answer still produces ``k`` confidently ranked
    passages. Judging whether they are relevant is the caller's job.

    **Hybrid pulls a wider pool from each method before fusing**, so a passage
    ranked tenth by both can still surface — which is where fusion earns its
    keep.

    **Reranking is the largest single quality gain available**, and it costs a
    model forward pass per candidate. Raise ``rerank_candidates`` to give it more
    to work with.

    **Read ``result.mode``, not the request.** A hybrid request falls back to
    dense without the dependencies, and the fallback is recorded rather than
    raised.

    Examples
    --------
    Retrieve with reranking, restricted to one document version::

        result = retrieve(
            index, "how do I cancel?", k=5, rerank=True,
            filters={"version": "2024"},
        )
        print(result.mode, [h.doc_id for h in result.hits])

    See Also
    --------
    buildml.rag.results.RetrieveResult : What comes back.
    buildml.rag.generate.generate_grounded : Answering from these passages.
    buildml.rag.evaluate : Measuring whether this works.
    """
    if index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("query must be a non-empty string.")
    from buildml.rag.defaults import default_retrieve_config

    base = config or default_retrieve_config()
    resolved_mode = mode if mode is not None else base.mode
    cfg = RetrieveConfig(
        k=base.k if k is None else int(k),
        mode=resolved_mode,
        fusion=fusion or base.fusion,  # type: ignore[arg-type]
        rrf_k=base.rrf_k,
        dense_weight=base.dense_weight,
        bm25_k1=base.bm25_k1,
        bm25_b=base.bm25_b,
        filters=filters if filters is not None else base.filters,
        rerank=base.rerank if rerank is None else rerank,
        rerank_model=base.rerank_model,
        rerank_candidates=base.rerank_candidates,
    )
    top_k = int(cfg.k)
    if top_k <= 0:
        raise ValidationError(f"k must be positive; got {top_k}")
    if cfg.mode not in {"dense", "bm25", "hybrid"}:
        raise ValidationError(
            f"Unknown retrieve mode {cfg.mode!r}; expected dense, bm25, or hybrid."
        )

    cand_k = _candidate_k(cfg, top_k)
    if cfg.mode == "dense":
        hits = _dense_hits(index, query, k=cand_k, filters=cfg.filters)
    elif cfg.mode == "bm25":
        hits = _bm25_hits(index, query, k=cand_k, cfg=cfg, filters=cfg.filters)
    else:
        hits = _hybrid_hits(index, query, k=cand_k, cfg=cfg, filters=cfg.filters)

    hits, rerank_notes = _apply_rerank(query, hits, k=top_k, cfg=cfg)
    hits = hits[:top_k]
    # Re-number ranks after truncation.
    hits = [
        Hit(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            score=h.score,
            text=h.text,
            rank=i,
            metadata=dict(h.metadata),
        )
        for i, h in enumerate(hits, start=1)
    ]

    disclosures = list(index.disclosures)
    disclosures.append(f"retrieve_mode={cfg.mode}")
    if cfg.mode == "hybrid":
        disclosures.append(f"fusion={cfg.fusion}")
        if cfg.fusion == "rrf":
            disclosures.append(f"rrf_k={cfg.rrf_k}")
        else:
            disclosures.append(f"dense_weight={cfg.dense_weight}")
    if cfg.filters:
        disclosures.append(f"metadata_filters={sorted(cfg.filters)}")
    disclosures.extend(rerank_notes)
    if not cfg.rerank:
        disclosures.append("rerank=off")

    return RetrieveResult(
        query=query,
        k=top_k,
        hits=tuple(hits),
        embedder_id=index.embed_config.embedder_id,
        mode=cfg.mode,
        fusion=cfg.fusion if cfg.mode == "hybrid" else None,
        filters=None if cfg.filters is None else dict(cfg.filters),
        rerank=bool(cfg.rerank),
        disclosures=tuple(disclosures),
        config=cfg.to_dict(),
    )
