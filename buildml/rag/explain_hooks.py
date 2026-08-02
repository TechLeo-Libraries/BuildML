"""History / catalog / walkthrough helpers for RAG operations."""

from __future__ import annotations

from typing import Any


def index_result_summary(index_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``rag_embed_and_index`` history."""
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
    """Compact result_summary for ``rag_retrieve`` history."""
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
    """Compact result_summary for ``rag_evaluate`` history."""
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
    """Compact result_summary for ``rag_generate`` history."""
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
    """Factual walkthrough disclosure for RAG index / retrieve / generate.

    Does not imply Session checkpoint holds the vector index, and does not
    treat catalog availability as production readiness.
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
    """Build walkthrough ``rag_status`` from a Session."""
    return rag_status(
        index_result=getattr(session, "rag_index_result", None),
        eval_result=getattr(session, "rag_eval_result", None),
        retrieve_result=getattr(session, "rag_retrieve_result", None),
        generate_result=getattr(session, "rag_generate_result", None),
        corpus=getattr(session, "_rag_corpus", None),
        history=list(getattr(session, "history", []) or []),
    )
