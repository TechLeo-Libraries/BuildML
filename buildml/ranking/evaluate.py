"""Holdout ranking evaluation for tabular LTR (per-query metrics)."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.ranking.features import (
    average_precision_at_k,
    mean_metric_over_queries,
    mrr_at_k,
    ndcg_at_k_graded,
    partition_frame,
)
from buildml.ranking.rank import score_rows
from buildml.ranking.results import RankerEvalResult, RankerPlan
from buildml.ranking.types import RankerBackend


def evaluate_ranker(
    dataset: Dataset,
    plan: RankerPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "test",
    k: int = 10,
    backend: RankerBackend | None = None,
) -> RankerEvalResult:
    """Score ranking quality on a holdout partition.

    Protocol
    --------
    - Fit never sees holdout rows (caller / Session gate).
    - For each holdout query: score all candidate rows for that query,
      sort by descending score, then compute graded nDCG@K, MAP@K, and MRR@K.
    - MAP/MRR treat relevance > ``plan.relevance_threshold`` as relevant.
    - Metrics are macro-averaged over queries that have ≥1 relevant item
      (nDCG also requires a positive ideal DCG; queries with all-zero labels
      are skipped and disclosed).

    Honesty: tabular LTR metrics on labeled judgments — not RAG retrieve
    eval and not recommender known-item CF metrics.
    """
    if int(k) < 1:
        raise ValidationError("k must be >= 1.")

    holdout = partition_frame(dataset, split_plan, partition)
    n_holdout = int(len(holdout))
    disclosures = [
        "Holdout scored with frozen train ranker (no refit).",
        "Per-query ranking: score all judgment rows for the query, then cut@K.",
        f"Binary relevance for MAP/MRR uses grade > {plan.relevance_threshold}.",
        "Tabular LTR is not RAG nDCG/MRR on chunks; not recommender Precision@K CF.",
    ]
    warnings: list[str] = []
    if not plan.group_split_disclosed:
        warnings.append(
            "Query groups may overlap partitions; prefer group_split on "
            "query_column for leakage-safe LTR evaluation."
        )

    if holdout.empty:
        return RankerEvalResult(
            partition=str(partition),
            method=plan.method,
            k=int(k),
            n_queries_scored=0,
            n_holdout_rows=0,
            metrics={
                "ndcg_at_k": 0.0,
                "map_at_k": 0.0,
                "mrr_at_k": 0.0,
            },
            disclosures=tuple(disclosures + ["Empty holdout partition."]),
            warnings=tuple(warnings),
        )

    scores = score_rows(plan, holdout, backend=backend)
    frame = holdout.copy()
    frame["__score__"] = scores
    qcol = plan.query_column
    rel_col = plan.relevance_column
    threshold = float(plan.relevance_threshold)

    ndcgs: list[float] = []
    maps: list[float] = []
    mrrs: list[float] = []
    skipped_no_rel = 0

    for _, sub in frame.groupby(qcol, sort=False):
        rels = sub[rel_col].to_numpy(dtype=float)
        if not np.any(rels > threshold):
            skipped_no_rel += 1
            continue
        ordered = sub.sort_values("__score__", ascending=False)
        ranked_rels = ordered[rel_col].tolist()
        ndcgs.append(ndcg_at_k_graded(ranked_rels, int(k)))
        maps.append(
            average_precision_at_k(ranked_rels, int(k), threshold=threshold)
        )
        mrrs.append(mrr_at_k(ranked_rels, int(k), threshold=threshold))

    if skipped_no_rel:
        disclosures.append(
            f"Skipped {skipped_no_rel} query(ies) with no relevant items "
            f"(grade ≤ {threshold})."
        )

    if not ndcgs:
        return RankerEvalResult(
            partition=str(partition),
            method=plan.method,
            k=int(k),
            n_queries_scored=0,
            n_holdout_rows=n_holdout,
            metrics={
                "ndcg_at_k": 0.0,
                "map_at_k": 0.0,
                "mrr_at_k": 0.0,
            },
            disclosures=tuple(
                disclosures + ["No holdout queries with relevant items to score."]
            ),
            warnings=tuple(warnings),
        )

    return RankerEvalResult(
        partition=str(partition),
        method=plan.method,
        k=int(k),
        n_queries_scored=len(ndcgs),
        n_holdout_rows=n_holdout,
        metrics={
            "ndcg_at_k": mean_metric_over_queries(ndcgs),
            "map_at_k": mean_metric_over_queries(maps),
            "mrr_at_k": mean_metric_over_queries(mrrs),
        },
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
