"""History / catalog helpers for RAG operations."""

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
        "relevance_mode": payload.get("relevance_mode"),
    }
