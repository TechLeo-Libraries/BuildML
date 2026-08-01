"""Retrieval evaluation metrics (recall@k, MRR, nDCG@k) and config compare."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.index import RagIndex, build_index
from buildml.rag.results import ConfigCompareResult, CorpusHandle, RagEvalResult
from buildml.rag.retrieve import retrieve
from buildml.rag.types import EvalConfig, RelevanceMode, RetrieveConfig


def _normalize_qrels(
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    relevance_mode: RelevanceMode,
) -> list[tuple[str, set[str]]]:
    """Normalize gold labels to ``(query, relevant_ids)`` pairs.

    In ``document`` mode ids are ``doc_id`` values. In ``chunk`` mode ids are
    ``chunk_id`` values (``relevant_chunk_ids`` / ``relevant_chunks``).
    """
    pairs: list[tuple[str, set[str]]] = []
    if isinstance(qrels, Mapping):
        for query, docs in qrels.items():
            relevant = {str(d) for d in docs}
            if not relevant:
                raise ValidationError(f"qrels entry for {query!r} has no relevant ids.")
            pairs.append((str(query), relevant))
        return pairs
    for i, row in enumerate(qrels):
        if not isinstance(row, Mapping):
            raise ValidationError(f"qrels[{i}] must be a mapping.")
        query = row.get("query")
        if query is None:
            raise ValidationError(f"qrels[{i}] is missing 'query'.")
        relevant: set[str]
        if relevance_mode == "chunk":
            if "relevant_chunk_ids" in row:
                relevant = {str(d) for d in row["relevant_chunk_ids"]}
            elif "relevant_chunks" in row:
                relevant = {str(d) for d in row["relevant_chunks"]}
            elif "chunk_id" in row:
                relevant = {str(row["chunk_id"])}
            else:
                raise ValidationError(
                    f"qrels[{i}] needs relevant_chunk_ids, relevant_chunks, or "
                    "chunk_id when relevance_mode='chunk'."
                )
        elif "relevant_doc_ids" in row:
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


def _dcg(relevances: Sequence[float]) -> float:
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return total


def _ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], *, k: int) -> float:
    """Binary nDCG@k over the ranked id list."""
    gains = [1.0 if item in relevant else 0.0 for item in ranked_ids[:k]]
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(relevant), k))
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def evaluate_retrieval(
    index: RagIndex,
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    k: int = 5,
    relevance_mode: RelevanceMode = "document",
    retrieve_config: RetrieveConfig | None = None,
    mode: str | None = None,
) -> RagEvalResult:
    """Compute recall@k, MRR, hit-rate@k, and nDCG@k on gold qrels.

    ``relevance_mode="document"`` (default): a chunk hit counts via parent
    ``doc_id``. ``relevance_mode="chunk"`` scores against ``chunk_id`` labels.
    Metrics are ranking quality, not classification accuracy.
    """
    if index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if k <= 0:
        raise ValidationError(f"k must be positive; got {k}")
    if relevance_mode not in {"document", "chunk"}:
        raise ValidationError(
            f"relevance_mode must be 'document' or 'chunk'; got {relevance_mode!r}"
        )
    pairs = _normalize_qrels(qrels, relevance_mode=relevance_mode)
    cfg = retrieve_config or RetrieveConfig(k=k)
    if mode is not None:
        cfg = RetrieveConfig(
            k=k,
            mode=mode,  # type: ignore[arg-type]
            fusion=cfg.fusion,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            filters=cfg.filters,
            rerank=cfg.rerank,
            rerank_model=cfg.rerank_model,
            rerank_candidates=cfg.rerank_candidates,
        )
    else:
        cfg = RetrieveConfig(
            k=k,
            mode=cfg.mode,
            fusion=cfg.fusion,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            filters=cfg.filters,
            rerank=cfg.rerank,
            rerank_model=cfg.rerank_model,
            rerank_candidates=cfg.rerank_candidates,
        )

    recalls: list[float] = []
    rr_values: list[float] = []
    ndcgs: list[float] = []
    hits_flags: list[float] = []
    per_query: list[dict[str, Any]] = []
    for query, relevant in pairs:
        result = retrieve(index, query, k=k, config=cfg)
        if relevance_mode == "chunk":
            ranked_ids = [h.chunk_id for h in result.hits]
        else:
            ranked_ids = [h.doc_id for h in result.hits]
        # Deduplicate doc ids for document-mode recall/MRR while keeping first rank.
        if relevance_mode == "document":
            seen: set[str] = set()
            unique_ranked: list[str] = []
            for doc_id in ranked_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                unique_ranked.append(doc_id)
            ranked_for_metrics = unique_ranked
        else:
            ranked_for_metrics = ranked_ids

        hit_set = set(ranked_for_metrics) & relevant
        recall = len(hit_set) / len(relevant)
        recalls.append(recall)
        rr = 0.0
        for rank, item_id in enumerate(ranked_for_metrics, start=1):
            if item_id in relevant:
                rr = 1.0 / rank
                break
        rr_values.append(rr)
        ndcg = _ndcg_at_k(ranked_for_metrics, relevant, k=k)
        ndcgs.append(ndcg)
        hit_flag = 1.0 if hit_set else 0.0
        hits_flags.append(hit_flag)
        per_query.append(
            {
                "query": query,
                "relevant_ids": sorted(relevant),
                "retrieved_ids": ranked_for_metrics,
                "recall_at_k": recall,
                "rr": rr,
                "ndcg_at_k": ndcg,
                "hit": bool(hit_set),
                # Backward-compatible aliases used by M1 tests/readers.
                "relevant_doc_ids": sorted(relevant) if relevance_mode == "document" else [],
                "retrieved_doc_ids": ranked_for_metrics if relevance_mode == "document" else [],
            }
        )
    n = len(pairs)
    return RagEvalResult(
        n_queries=n,
        k=k,
        recall_at_k=float(sum(recalls) / n),
        mrr=float(sum(rr_values) / n),
        ndcg_at_k=float(sum(ndcgs) / n),
        hit_rate_at_k=float(sum(hits_flags) / n),
        per_query=tuple(per_query),
        relevance_mode=relevance_mode,
        retrieve_mode=cfg.mode,
        disclosures=(
            f"relevance_mode={relevance_mode}",
            f"retrieve_mode={cfg.mode}",
            f"k={k}",
            f"embedder_id={index.embed_config.embedder_id}",
            "recall@k / MRR / nDCG@k are ranking metrics, not classification accuracy.",
        ),
        warnings=(),
    )


def compare_retrieval_configs(
    corpus: CorpusHandle,
    configs: Sequence[Mapping[str, Any] | EvalConfig],
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    k: int = 5,
    relevance_mode: RelevanceMode = "document",
) -> ConfigCompareResult:
    """Build an index per config row and compare retrieval metrics.

    Each config mapping may include ``name``, ``chunk_size``, ``chunk_overlap``,
    ``embedder``, ``retrieve`` (``RetrieveConfig`` or dict), and ``relevance_mode``.
    """
    if not configs:
        raise ValidationError("compare_retrieval_configs requires at least one config.")
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(configs):
        if isinstance(item, EvalConfig):
            name = f"config-{i}"
            chunk_size = None
            chunk_overlap = None
            embedder = None
            retrieve_cfg = item.retrieve
            mode = item.relevance_mode
            eval_k = item.k
        else:
            name = str(item.get("name") or f"config-{i}")
            chunk_size = item.get("chunk_size")
            chunk_overlap = item.get("chunk_overlap")
            embedder = item.get("embedder")
            retrieve_raw = item.get("retrieve")
            if isinstance(retrieve_raw, RetrieveConfig):
                retrieve_cfg = retrieve_raw
            elif isinstance(retrieve_raw, Mapping):
                retrieve_cfg = RetrieveConfig.from_dict(dict(retrieve_raw))
            else:
                retrieve_cfg = RetrieveConfig(k=k)
            mode = item.get("relevance_mode") or relevance_mode
            eval_k = int(item.get("k") or k)
        index = build_index(
            corpus,
            chunk_size=None if chunk_size is None else int(chunk_size),
            chunk_overlap=None if chunk_overlap is None else int(chunk_overlap),
            embedder=embedder,
        )
        metrics = evaluate_retrieval(
            index,
            qrels,
            k=eval_k,
            relevance_mode=mode,  # type: ignore[arg-type]
            retrieve_config=retrieve_cfg,
        )
        rows.append(
            {
                "name": name,
                "n_chunks": index.to_index_result().n_chunks,
                "embedder_id": index.embed_config.embedder_id,
                "dim": index.embed_config.dim,
                "chunk_config": index.chunk_config.to_dict(),
                "retrieve_mode": metrics.retrieve_mode,
                "relevance_mode": metrics.relevance_mode,
                "k": metrics.k,
                "recall_at_k": metrics.recall_at_k,
                "mrr": metrics.mrr,
                "ndcg_at_k": metrics.ndcg_at_k,
                "hit_rate_at_k": metrics.hit_rate_at_k,
            }
        )
    return ConfigCompareResult(
        rows=tuple(rows),
        k=k,
        relevance_mode=relevance_mode,
        disclosures=(
            f"Compared {len(rows)} retrieval config(s).",
            f"relevance_mode={relevance_mode}",
            f"k={k}",
            "Each row rebuilds its own index; scores are not shared across rows.",
        ),
    )
