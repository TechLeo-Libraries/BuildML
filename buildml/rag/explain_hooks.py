"""Condense RAG results into the small payloads history and walkthroughs record.

A full :class:`~buildml.rag.results.RetrieveResult` carries the text of every
retrieved passage, and a Session that ran fifty retrievals would carry fifty
copies of it. History needs the shape of what happened — the mode, the counts,
the top document IDs — not the payload. These functions extract that shape.

Everything here is defensive by design. Each summariser accepts ``None``, a
result object, or a plain dict, and reads through ``.get`` so a missing field
becomes ``None`` rather than an exception. Explanation must never be the thing
that breaks a working session.

The status builder goes further and states what is *not* known: that no index is
attached, that no evaluation has been run, that a Session checkpoint does not
contain the vector index. Silence there gets read as reassurance, which is
exactly the wrong inference.

See Also
--------
buildml.rag.results : The result objects being summarised.
buildml.rag.catalog.rag_capability_matrix : What the install can do.
"""

from __future__ import annotations

from typing import Any


def index_result_summary(index_result: Any) -> dict[str, Any]:
    """Reduce an index result to the few facts worth keeping in history.

    Enough to answer later "what was indexed, and with what?" without storing
    chunks or vectors.

    Parameters
    ----------
    index_result:
        An :class:`~buildml.rag.results.IndexResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        ``n_chunks``, ``n_documents``, ``embedder_id``, ``dim``, and
        ``store_backend``. Empty when the input is ``None``; individual values
        are ``None`` when absent.

    Notes
    -----
    **The embedder ID is the important one.** An index built with hashing
    embeddings behaves quite differently from a semantic one, and this is where
    that gets recorded.
    """
    if index_result is None:
        return {}
    if hasattr(index_result, "to_dict"):
        payload = index_result.to_dict()
    else:
        payload = dict(index_result)
    return {
        "n_chunks": payload.get("n_chunks"),
        "n_documents": payload.get("n_documents"),
        "embedder_id": payload.get("embedder_id"),
        "dim": payload.get("dim"),
        "store_backend": payload.get("store_backend"),
    }


def retrieve_result_summary(retrieve_result: Any) -> dict[str, Any]:
    """Reduce a retrieval result to a history-sized record.

    Keeps the query, the settings that were actually used, and the top document
    IDs — enough to see later that a query returned the wrong sources, without
    storing their text.

    Parameters
    ----------
    retrieve_result:
        A :class:`~buildml.rag.results.RetrieveResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        ``query``, ``k``, ``n_hits``, ``top_doc_ids`` (at most five),
        ``embedder_id``, ``mode``, and ``rerank``. Empty when the input is
        ``None``.

    Notes
    -----
    **``mode`` is what ran, not what was requested**, so a hybrid request that
    fell back to dense is visible here.

    **Only the first five document IDs are kept**, which is enough to recognise
    a retrieval and bounded regardless of ``k``.
    """
    if retrieve_result is None:
        return {}
    if hasattr(retrieve_result, "to_dict"):
        payload = retrieve_result.to_dict()
    else:
        payload = dict(retrieve_result)
    hits = payload.get("hits") or []
    return {
        "query": payload.get("query"),
        "k": payload.get("k"),
        "n_hits": payload.get("n_hits", len(hits)),
        "top_doc_ids": [h.get("doc_id") for h in hits[:5]],
        "embedder_id": payload.get("embedder_id"),
        "mode": payload.get("mode"),
        "rerank": payload.get("rerank"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Reduce an evaluation result to its headline metrics.

    Drops the per-query breakdown, which is the bulk of the result and the part
    you read once while debugging rather than needing in history.

    Parameters
    ----------
    eval_result:
        A :class:`~buildml.rag.results.RagEvalResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        ``n_queries``, ``k``, ``recall_at_k``, ``mrr``, ``ndcg_at_k``,
        ``hit_rate_at_k``, ``relevance_mode``, and ``retrieve_mode``. Empty when
        the input is ``None``.

    Notes
    -----
    **The two modes are kept alongside the numbers on purpose.** Metrics from
    document-mode and chunk-mode evaluations are not comparable, and neither are
    metrics from different retrieval modes.
    """
    if eval_result is None:
        return {}
    if hasattr(eval_result, "to_dict"):
        payload = eval_result.to_dict()
    else:
        payload = dict(eval_result)
    return {
        "n_queries": payload.get("n_queries"),
        "k": payload.get("k"),
        "recall_at_k": payload.get("recall_at_k"),
        "mrr": payload.get("mrr"),
        "ndcg_at_k": payload.get("ndcg_at_k"),
        "hit_rate_at_k": payload.get("hit_rate_at_k"),
        "relevance_mode": payload.get("relevance_mode"),
        "retrieve_mode": payload.get("retrieve_mode"),
    }


def generate_result_summary(generate_result: Any) -> dict[str, Any]:
    """Reduce a generation result to a history-sized record.

    Records that an answer was produced, how long it was, which model produced
    it, and what it was allowed to cite — without storing the answer text.

    Parameters
    ----------
    generate_result:
        A :class:`~buildml.rag.results.GenerateResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        ``query``, ``n_citations``, ``provider_model``, ``citation_doc_ids`` (at
        most five), and ``answer_chars``. Empty when the input is ``None``.

    Notes
    -----
    **The answer itself is deliberately not kept**, only its length. Generated
    text can be long and may contain material that should not be retained in a
    session log.

    **``citation_doc_ids`` lists what was available to cite**, not what the
    answer cited. The faithfulness report on the full result has that.
    """
    if generate_result is None:
        return {}
    if hasattr(generate_result, "to_dict"):
        payload = generate_result.to_dict()
    else:
        payload = dict(generate_result)
    citations = payload.get("citations") or []
    return {
        "query": payload.get("query"),
        "n_citations": payload.get("n_citations", len(citations)),
        "provider_model": payload.get("provider_model"),
        "citation_doc_ids": [c.get("doc_id") for c in citations[:5]],
        "answer_chars": len(str(payload.get("answer") or "")),
    }


def rag_status(
    *,
    index_result: Any | None = None,
    eval_result: Any | None = None,
    retrieve_result: Any | None = None,
    generate_result: Any | None = None,
    corpus: Any | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe the state of a RAG workflow, including what is missing from it.

    Assembles a factual picture across all four stages for walkthroughs and
    status displays: what was indexed and with what, whether it has been
    evaluated, what the last retrieval and generation did.

    The absences are as informative as the presences, and are reported
    explicitly. An index with no evaluation attached says so. RAG operations in
    history with no live index says so. The fact that a Session checkpoint does
    not contain the vector index is stated every time, because that is the
    assumption people make and it costs them their index.

    Parameters
    ----------
    index_result:
        The current index result, or ``None`` if nothing is indexed.
    eval_result:
        The last evaluation, or ``None``.
    retrieve_result:
        The last retrieval, or ``None``.
    generate_result:
        The last generation, or ``None``.
    corpus:
        The corpus handle, or ``None``.
    history:
        Session history records, scanned for past RAG operations.

    Returns
    -------
    dict
        ``enabled`` (an index is attached), ``present`` (RAG has been used at
        all), ``disclosures`` (plain-language statements), and per-stage
        sections ``index``, ``eval``, ``retrieve``, ``generate``, ``corpus`` —
        each ``None`` when that stage has not run.

    Notes
    -----
    **``enabled`` and ``present`` differ, and the difference matters.** A
    session restored from a checkpoint has ``present=True`` from history but
    ``enabled=False``, because the index did not travel with the checkpoint and
    must be rebuilt or loaded from a bundle.

    **Nothing here is a readiness verdict.** It reports facts; whether they
    constitute a system worth deploying is a judgement this cannot make.

    See Also
    --------
    rag_status_for_session : The Session-facing wrapper.
    buildml.rag.checkpoint.save_rag_bundle : Persisting the index properly.
    """
    records = list(history or [])
    saw_rag = any(
        str(r.get("operation_id") or r.get("action")).startswith("rag_")
        or str(r.get("operation_id") or r.get("action"))
        in {"save_rag_bundle", "load_rag_bundle"}
        for r in records
    )
    if index_result is None:
        return {
            "enabled": False,
            "present": saw_rag or corpus is not None,
            "disclosures": (
                [
                    "RAG operations appear in Session history, but no live "
                    "rag_index_result is attached for index disclosure."
                ]
                if saw_rag
                else (
                    [
                        "A RAG corpus handle is attached, but no index has been built yet."
                    ]
                    if corpus is not None
                    else []
                )
            ),
            "index": None,
            "eval": None,
            "retrieve": None,
            "generate": None,
            "corpus": None
            if corpus is None
            else {
                "n_documents": getattr(corpus, "n_documents", None),
                "source": getattr(corpus, "source", None),
            },
        }

    index_payload = (
        index_result.to_dict() if hasattr(index_result, "to_dict") else dict(index_result)
    )
    embed_cfg = dict(index_payload.get("embed_config") or {})
    disclosures = [
        f"Indexed n_documents={index_payload.get('n_documents')}, "
        f"n_chunks={index_payload.get('n_chunks')}.",
        f"embedder_id={index_payload.get('embedder_id')}, "
        f"dim={index_payload.get('dim')}, "
        f"store_backend={index_payload.get('store_backend')}.",
        "Session checkpoints do not contain the vector index; use save_rag_bundle / "
        "load_rag_bundle for retrieval artifacts.",
    ]
    if embed_cfg.get("device"):
        disclosures.append(f"embed_device={embed_cfg.get('device')}")
    elif embed_cfg.get("backend") == "hashing":
        disclosures.append(
            "Hashing embedder is lexical/hashed, not a semantic sentence model."
        )
    elif embed_cfg.get("backend") == "sentence-transformers":
        disclosures.append(
            "Sentence-transformer embedder is the recommended semantic default when buildml[rag] is installed."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last eval: "
            f"relevance_mode={eval_payload.get('relevance_mode')}, "
            f"retrieve_mode={eval_payload.get('retrieve_mode')}, "
            f"k={eval_payload.get('k')}, "
            f"recall@k={eval_payload.get('recall_at_k')}, "
            f"mrr={eval_payload.get('mrr')}, "
            f"ndcg@k={eval_payload.get('ndcg_at_k')}."
        )
        disclosures.append(
            "Eval metrics measure ranking quality on the supplied qrels; they are not "
            "classification accuracy and do not by themselves prove grounded generate quality."
        )
    else:
        disclosures.append("No rag_evaluate result is attached on this Session.")

    retrieve_payload = None
    if retrieve_result is not None:
        retrieve_payload = (
            retrieve_result.to_dict()
            if hasattr(retrieve_result, "to_dict")
            else dict(retrieve_result)
        )
        disclosures.append(
            "Last retrieve: "
            f"mode={retrieve_payload.get('mode')}, "
            f"k={retrieve_payload.get('k')}, "
            f"rerank={retrieve_payload.get('rerank')}, "
            f"n_hits={retrieve_payload.get('n_hits')}."
        )

    generate_payload = None
    if generate_result is not None:
        generate_payload = (
            generate_result.to_dict()
            if hasattr(generate_result, "to_dict")
            else dict(generate_result)
        )
        disclosures.append(
            "Last generate: "
            f"n_citations={generate_payload.get('n_citations')}, "
            f"provider_model={generate_payload.get('provider_model')}."
        )
        disclosures.append(
            "Grounded answers cite retrieved chunks; verify claims against source text."
        )
    else:
        disclosures.append("No rag_generate result is attached on this Session.")

    corpus_payload = None
    if corpus is not None:
        corpus_payload = {
            "n_documents": getattr(corpus, "n_documents", None),
            "source": getattr(corpus, "source", None),
        }

    return {
        "enabled": True,
        "present": True,
        "disclosures": disclosures,
        "index": {
            "n_chunks": index_payload.get("n_chunks"),
            "n_documents": index_payload.get("n_documents"),
            "embedder_id": index_payload.get("embedder_id"),
            "dim": index_payload.get("dim"),
            "store_backend": index_payload.get("store_backend"),
            "embed_config": embed_cfg,
            "chunk_config": dict(index_payload.get("chunk_config") or {}),
        },
        "eval": eval_payload,
        "retrieve": None
        if retrieve_payload is None
        else {
            "query": retrieve_payload.get("query"),
            "k": retrieve_payload.get("k"),
            "mode": retrieve_payload.get("mode"),
            "rerank": retrieve_payload.get("rerank"),
            "n_hits": retrieve_payload.get("n_hits"),
        },
        "generate": None
        if generate_payload is None
        else {
            "query": generate_payload.get("query"),
            "n_citations": generate_payload.get("n_citations"),
            "provider_model": generate_payload.get("provider_model"),
        },
        "corpus": corpus_payload,
    }


def rag_status_for_session(session: Any) -> dict[str, Any]:
    """Read a Session's RAG state and describe it.

    Pulls the four result attributes, the corpus, and the history off a Session
    and hands them to :func:`rag_status`. Every read uses ``getattr`` with a
    default, so this works on a Session that has never touched RAG and on
    partially-constructed or mock objects.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`, or anything with the same
        attributes.

    Returns
    -------
    dict
        The status payload from :func:`rag_status`.

    See Also
    --------
    rag_status : The underlying builder and its return shape.
    """
    return rag_status(
        index_result=getattr(session, "rag_index_result", None),
        eval_result=getattr(session, "rag_eval_result", None),
        retrieve_result=getattr(session, "rag_retrieve_result", None),
        generate_result=getattr(session, "rag_generate_result", None),
        corpus=getattr(session, "_rag_corpus", None),
        history=list(getattr(session, "history", []) or []),
    )
