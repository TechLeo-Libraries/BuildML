"""Retrieval evaluation metrics (recall@k, MRR)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.index import RagIndex
from buildml.rag.results import RagEvalResult
from buildml.rag.retrieve import retrieve


def _normalize_qrels(
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
) -> list[tuple[str, set[str]]]:
    """Normalize gold labels to ``(query, relevant_doc_ids)`` pairs."""
    pairs: list[tuple[str, set[str]]] = []
    if isinstance(qrels, Mapping):
        for query, docs in qrels.items():
            relevant = {str(d) for d in docs}
            if not relevant:
                raise ValidationError(f"qrels entry for {query!r} has no relevant docs.")
            pairs.append((str(query), relevant))
        return pairs
    for i, row in enumerate(qrels):
        if not isinstance(row, Mapping):
            raise ValidationError(f"qrels[{i}] must be a mapping.")
        query = row.get("query")
        if query is None:
            raise ValidationError(f"qrels[{i}] is missing 'query'.")
        if "relevant_doc_ids" in row:
            relevant = {str(d) for d in row["relevant_doc_ids"]}
        elif "relevant_docs" in row:
            relevant = {str(d) for d in row["relevant_docs"]}
        elif "doc_id" in row:
            relevant = {str(row["doc_id"])}
        else:
            raise ValidationError(
                f"qrels[{i}] needs relevant_doc_ids, relevant_docs, or doc_id."
            )
        if not relevant:
            raise ValidationError(f"qrels[{i}] has an empty relevance set.")
        pairs.append((str(query), relevant))
    if not pairs:
        raise ValidationError("qrels is empty.")
    return pairs


def evaluate_retrieval(
    index: RagIndex,
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    k: int = 5,
) -> RagEvalResult:
    """Compute document-level recall@k and MRR for gold query relevance labels.

    A chunk hit counts as a hit for its parent ``doc_id``. Metrics are not
    classification accuracy; disclosures state the relevance mode and ``k``.
    """
    if index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if k <= 0:
        raise ValidationError(f"k must be positive; got {k}")
    pairs = _normalize_qrels(qrels)
    recalls: list[float] = []
    rr_values: list[float] = []
    per_query: list[dict[str, Any]] = []
    for query, relevant in pairs:
        result = retrieve(index, query, k=k)
        ranked_docs = [h.doc_id for h in result.hits]
        hit_set = set(ranked_docs) & relevant
        recall = len(hit_set) / len(relevant)
        recalls.append(recall)
        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        rr_values.append(rr)
        per_query.append(
            {
                "query": query,
                "relevant_doc_ids": sorted(relevant),
                "retrieved_doc_ids": ranked_docs,
                "recall_at_k": recall,
                "rr": rr,
                "hit": bool(hit_set),
            }
        )
    n = len(pairs)
    return RagEvalResult(
        n_queries=n,
        k=k,
        recall_at_k=float(sum(recalls) / n),
        mrr=float(sum(rr_values) / n),
        per_query=tuple(per_query),
        relevance_mode="document",
        disclosures=(
            "relevance_mode=document (chunk hits count via parent doc_id)",
            f"k={k}",
            f"embedder_id={index.embed_config.embedder_id}",
            "recall@k is not classification accuracy.",
        ),
        warnings=(),
    )
