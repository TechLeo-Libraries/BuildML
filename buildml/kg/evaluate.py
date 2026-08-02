"""Holdout link-prediction evaluation (filtered MRR / Hits@K)."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.kg.features import (
    build_triples,
    encode_triples,
    hits_at_k,
    mrr_from_ranks,
    partition_frame,
)
from buildml.kg.models import score_all_heads, score_all_tails
from buildml.kg.results import KgEvalResult, KgPlan


def _rank_of_true(
    scores: np.ndarray,
    true_id: int,
    *,
    true_set: frozenset[tuple[int, int, int]],
    head: int,
    relation: int,
    tail: int,
    mode: str,
) -> int:
    """1-indexed filtered rank of the true fill-in."""
    order = np.argsort(-scores)
    rank = 0
    for cand in order.tolist():
        if mode == "tail":
            triple = (head, relation, cand)
        else:
            triple = (cand, relation, tail)
        if cand == true_id:
            rank += 1
            return rank
        if triple in true_set and triple != (head, relation, tail):
            continue
        rank += 1
    return rank


def evaluate_kg(
    dataset: Dataset,
    plan: KgPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "test",
    k: int = 10,
) -> KgEvalResult:
    """Evaluate link prediction on a holdout partition.

    Protocol
    --------
    - Fit never sees holdout triples (Session / assert_fit_partition gate).
    - For each holdout triple whose head, relation, and tail are in the
      **train** vocab: rank the true tail among all train entities (and
      separately the true head), using the filtered setting (other known
      train true triples removed from the ranking).
    - Metrics: MRR and Hits@1 / Hits@3 / Hits@K, averaged over head+tail
      rankings (standard KG filtered ranking).

    Triples with OOV entities/relations are skipped and disclosed.
    """
    if int(k) < 1:
        raise ValidationError("k must be >= 1.")

    holdout = partition_frame(dataset, split_plan, partition)
    triples = build_triples(
        holdout,
        head_column=plan.head_column,
        relation_column=plan.relation_column,
        tail_column=plan.tail_column,
    )

    # Encode; track OOV skips
    n_raw = len(triples)
    try:
        heads_i, rels_i, tails_i = encode_triples(
            triples,
            head_column=plan.head_column,
            relation_column=plan.relation_column,
            tail_column=plan.tail_column,
            entity_index=plan.entity_index_,
            relation_index=plan.relation_index_,
        )
    except ValidationError:
        return KgEvalResult(
            partition=str(partition),
            method=plan.method,
            k=int(k),
            n_triples_scored=0,
            n_skipped_unknown=n_raw,
            metrics={
                "mrr": 0.0,
                "hits_at_1": 0.0,
                "hits_at_3": 0.0,
                f"hits_at_{int(k)}": 0.0,
            },
            disclosures=(
                "No holdout triples fully in train vocab; metrics are zeros.",
                "Known-item / known-relation protocol: OOV skipped.",
            ),
            warnings=("All holdout triples reference OOV entities or relations.",),
        )

    n_scored = int(len(heads_i))
    n_skipped = int(n_raw - n_scored)
    true_set = plan.true_triple_set_
    # Also filter other holdout triples so we do not penalize ranking a
    # different true holdout triple higher (filtered setting extension).
    holdout_set = frozenset(
        zip(heads_i.tolist(), rels_i.tolist(), tails_i.tolist(), strict=True)
    )
    filter_set = true_set | holdout_set

    ranks: list[int] = []
    norm = "l2" if plan.norm == "l2" else "l1"
    for h, r, t in zip(heads_i.tolist(), rels_i.tolist(), tails_i.tolist(), strict=True):
        # Tail prediction
        tail_scores = score_all_tails(
            plan.method,
            h,
            r,
            plan.entity_embeddings_,
            plan.relation_embeddings_,
            norm=norm,  # type: ignore[arg-type]
        )
        ranks.append(
            _rank_of_true(
                tail_scores,
                t,
                true_set=filter_set,
                head=h,
                relation=r,
                tail=t,
                mode="tail",
            )
        )
        # Head prediction
        head_scores = score_all_heads(
            plan.method,
            r,
            t,
            plan.entity_embeddings_,
            plan.relation_embeddings_,
            norm=norm,  # type: ignore[arg-type]
        )
        ranks.append(
            _rank_of_true(
                head_scores,
                h,
                true_set=filter_set,
                head=h,
                relation=r,
                tail=t,
                mode="head",
            )
        )

    metrics = {
        "mrr": mrr_from_ranks(ranks),
        "hits_at_1": hits_at_k(ranks, 1),
        "hits_at_3": hits_at_k(ranks, 3),
        f"hits_at_{int(k)}": hits_at_k(ranks, int(k)),
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
    }

    disclosures = [
        "Filtered ranking: train (+ holdout) true triples removed except the "
        "target fill-in.",
        "MRR / Hits averaged over head and tail prediction per holdout triple.",
        "Frozen train embeddings; holdout never used for fitting.",
        "Not Graph ML node-classify accuracy; not RAG retrieve metrics.",
    ]
    warnings: list[str] = []
    if n_skipped:
        warnings.append(
            f"{n_skipped} holdout triple(s) skipped (OOV entity or relation)."
        )

    return KgEvalResult(
        partition=str(partition),
        method=plan.method,
        k=int(k),
        n_triples_scored=n_scored,
        n_skipped_unknown=n_skipped,
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
