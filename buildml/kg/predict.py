"""Score triples and predict missing links from a frozen KgPlan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.kg.features import build_triples, partition_frame
from buildml.kg.models import (
    score_all_heads,
    score_all_relations,
    score_all_tails,
    score_triples_batch,
)
from buildml.kg.results import KgPlan, PredictLinksResult, ScoreTriplesResult
from buildml.kg.types import LinkPredictionMode


def _norm(plan: KgPlan) -> str:
    return "l2" if plan.norm == "l2" else "l1"


def score_triples(
    dataset: Dataset | None,
    plan: KgPlan,
    split_plan: SplitPlan | None = None,
    *,
    partition: str | None = None,
    triples: pd.DataFrame | Sequence[tuple[Any, Any, Any]] | None = None,
) -> ScoreTriplesResult:
    """Score explicit triples or all triples in a partition.

    Uses frozen train embeddings to score candidate triples; unknown entities
    or relations receive ``-inf`` scores and are counted in disclosures.

    Parameters
    ----------
    dataset:
        Session dataset when scoring a partition; ``None`` when ``triples`` supplied.
    plan:
        Train-fitted knowledge-graph plan.
    split_plan:
        Split plan for partition scoring.
    partition:
        ``train``, ``validation``, ``test``, or ``all`` when ``triples`` is ``None``.
    triples:
        Explicit triple frame or sequence instead of a partition scan.

    Returns
    -------
    ScoreTriplesResult
        Per-triple scores and OOV counts.

    Raises
    ------
    ValidationError
        When both ``triples`` and ``partition`` are supplied or inputs are invalid.
    """
    if triples is None and partition is None:
        partition = "test"
    if triples is not None and partition is not None:
        raise ValidationError("Pass either triples= or partition=, not both.")

    if triples is not None:
        if isinstance(triples, pd.DataFrame):
            frame = triples
            for col in (plan.head_column, plan.relation_column, plan.tail_column):
                if col not in frame.columns:
                    # Accept positional h/r/t column names
                    if list(frame.columns)[:3] == ["head", "relation", "tail"]:
                        frame = frame.rename(
                            columns={
                                "head": plan.head_column,
                                "relation": plan.relation_column,
                                "tail": plan.tail_column,
                            }
                        )
                    else:
                        raise ValidationError(
                            f"triples DataFrame must contain "
                            f"{plan.head_column!r}, {plan.relation_column!r}, "
                            f"{plan.tail_column!r} (or head/relation/tail)."
                        )
        else:
            arr = list(triples)
            frame = pd.DataFrame(
                arr, columns=[plan.head_column, plan.relation_column, plan.tail_column]
            )
    else:
        if dataset is None:
            raise ValidationError("dataset is required when scoring a partition.")
        frame = partition_frame(dataset, split_plan, str(partition))
        frame = build_triples(
            frame,
            head_column=plan.head_column,
            relation_column=plan.relation_column,
            tail_column=plan.tail_column,
        )

    heads_raw = frame[plan.head_column].tolist()
    rels_raw = frame[plan.relation_column].tolist()
    tails_raw = frame[plan.tail_column].tolist()
    n = len(heads_raw)
    scores = np.full(n, -np.inf, dtype=float)
    unknown_ent = 0
    unknown_rel = 0
    valid_idx: list[int] = []
    h_idx: list[int] = []
    r_idx: list[int] = []
    t_idx: list[int] = []
    for i, (h, r, t) in enumerate(zip(heads_raw, rels_raw, tails_raw, strict=True)):
        hi = plan.entity_index_.get(h)
        ti = plan.entity_index_.get(t)
        ri = plan.relation_index_.get(r)
        if hi is None or ti is None:
            unknown_ent += 1
            continue
        if ri is None:
            unknown_rel += 1
            continue
        valid_idx.append(i)
        h_idx.append(hi)
        r_idx.append(ri)
        t_idx.append(ti)
    if valid_idx:
        batch_scores = score_triples_batch(
            plan.method,
            np.asarray(h_idx, dtype=np.int64),
            np.asarray(r_idx, dtype=np.int64),
            np.asarray(t_idx, dtype=np.int64),
            plan.entity_embeddings_,
            plan.relation_embeddings_,
            norm=_norm(plan),  # type: ignore[arg-type]
        )
        for i, s in zip(valid_idx, batch_scores.tolist(), strict=True):
            scores[i] = float(s)

    warnings: list[str] = []
    if unknown_ent:
        warnings.append(
            f"{unknown_ent} triple(s) reference entities absent from train vocab."
        )
    if unknown_rel:
        warnings.append(
            f"{unknown_rel} triple(s) reference relations absent from train vocab."
        )

    return ScoreTriplesResult(
        method=plan.method,
        n_triples=n,
        scores=tuple(float(s) for s in scores.tolist()),
        heads=tuple(heads_raw),
        relations=tuple(rels_raw),
        tails=tuple(tails_raw),
        unknown_entities=unknown_ent,
        unknown_relations=unknown_rel,
        disclosures=(
            "Scores from frozen train embeddings; higher is better.",
            "OOV entities/relations scored as -inf.",
        ),
        warnings=tuple(warnings),
    )


def _top_k_filtered(
    scores: np.ndarray,
    *,
    k: int,
    true_set: frozenset[tuple[int, int, int]],
    head: int | None,
    relation: int | None,
    tail: int | None,
    mode: LinkPredictionMode,
    keep_true: tuple[int, int, int] | None,
) -> tuple[list[int], list[float]]:
    """Return top-k candidate ids after filtering other known true triples."""
    order = np.argsort(-scores)
    picked: list[int] = []
    picked_scores: list[float] = []
    for cand in order.tolist():
        if mode == "tail":
            assert head is not None and relation is not None
            triple = (head, relation, cand)
        elif mode == "head":
            assert relation is not None and tail is not None
            triple = (cand, relation, tail)
        else:
            assert head is not None and tail is not None
            triple = (head, cand, tail)
        if keep_true is not None and triple == keep_true:
            picked.append(cand)
            picked_scores.append(float(scores[cand]))
        elif triple in true_set:
            continue
        else:
            picked.append(cand)
            picked_scores.append(float(scores[cand]))
        if len(picked) >= k:
            break
    return picked, picked_scores


def predict_links(
    plan: KgPlan,
    *,
    mode: LinkPredictionMode = "tail",
    heads: Sequence[Any] | None = None,
    relations: Sequence[Any] | None = None,
    tails: Sequence[Any] | None = None,
    k: int = 10,
    filtered: bool = True,
) -> PredictLinksResult:
    """Predict missing link components for incomplete triples.

    Ranks candidate heads, tails, or relations using frozen embeddings and
    optionally applies filtered ranking against known train triples.

    Parameters
    ----------
    plan:
        Train-fitted knowledge-graph plan.
    mode:
        ``tail``, ``head``, or ``relation`` prediction task.
    heads, relations, tails:
        Query components; required fields depend on ``mode``.
    k:
        Number of top candidates to return per query.
    filtered:
        When True, remove other known train true triples from rankings.

    Returns
    -------
    PredictLinksResult
        Top-k predictions and scores for each query.

    Raises
    ------
    ValidationError
        When ``k`` is invalid, ``mode`` is unknown, or required query fields are missing.

    Notes
    -----
    Modes
    ^^^^^
    - ``tail``: given (h, r, ?), rank candidate tails.
    - ``head``: given (?, r, t), rank candidate heads.
    - ``relation``: given (h, ?, t), rank candidate relations.

    When ``filtered=True``, other train-known true triples are removed from
    the ranking (standard KG filtered protocol); the query's own true fill-in
    is not required here: this is a prediction API.
    """
    if int(k) < 1:
        raise ValidationError("k must be >= 1.")
    if mode not in {"tail", "head", "relation"}:
        raise ValidationError("mode must be 'tail', 'head', or 'relation'.")

    warnings: list[str] = []
    predictions: list[tuple[Any, ...]] = []
    score_rows: list[tuple[float, ...]] = []
    q_heads: list[Any] = []
    q_rels: list[Any] = []
    q_tails: list[Any] = []

    true_set = plan.true_triple_set_ if filtered else frozenset()

    if mode == "tail":
        if heads is None or relations is None:
            raise ValidationError("mode='tail' requires heads= and relations=.")
        if len(heads) != len(relations):
            raise ValidationError("heads and relations must have equal length.")
        for h, r in zip(list(heads), list(relations), strict=True):
            q_heads.append(h)
            q_rels.append(r)
            q_tails.append(None)
            hi = plan.entity_index_.get(h)
            ri = plan.relation_index_.get(r)
            if hi is None or ri is None:
                warnings.append(f"Skipping OOV query head/relation: ({h!r}, {r!r}).")
                predictions.append(())
                score_rows.append(())
                continue
            scores = score_all_tails(
                plan.method,
                hi,
                ri,
                plan.entity_embeddings_,
                plan.relation_embeddings_,
                norm=_norm(plan),  # type: ignore[arg-type]
            )
            ids, sc = _top_k_filtered(
                scores,
                k=k,
                true_set=true_set,
                head=hi,
                relation=ri,
                tail=None,
                mode="tail",
                keep_true=None,
            )
            predictions.append(tuple(plan.entity_ids[i] for i in ids))
            score_rows.append(tuple(sc))

    elif mode == "head":
        if tails is None or relations is None:
            raise ValidationError("mode='head' requires tails= and relations=.")
        if len(tails) != len(relations):
            raise ValidationError("tails and relations must have equal length.")
        for t, r in zip(list(tails), list(relations), strict=True):
            q_heads.append(None)
            q_rels.append(r)
            q_tails.append(t)
            ti = plan.entity_index_.get(t)
            ri = plan.relation_index_.get(r)
            if ti is None or ri is None:
                warnings.append(f"Skipping OOV query relation/tail: ({r!r}, {t!r}).")
                predictions.append(())
                score_rows.append(())
                continue
            scores = score_all_heads(
                plan.method,
                ri,
                ti,
                plan.entity_embeddings_,
                plan.relation_embeddings_,
                norm=_norm(plan),  # type: ignore[arg-type]
            )
            ids, sc = _top_k_filtered(
                scores,
                k=k,
                true_set=true_set,
                head=None,
                relation=ri,
                tail=ti,
                mode="head",
                keep_true=None,
            )
            predictions.append(tuple(plan.entity_ids[i] for i in ids))
            score_rows.append(tuple(sc))

    else:  # relation
        if heads is None or tails is None:
            raise ValidationError("mode='relation' requires heads= and tails=.")
        if len(heads) != len(tails):
            raise ValidationError("heads and tails must have equal length.")
        for h, t in zip(list(heads), list(tails), strict=True):
            q_heads.append(h)
            q_rels.append(None)
            q_tails.append(t)
            hi = plan.entity_index_.get(h)
            ti = plan.entity_index_.get(t)
            if hi is None or ti is None:
                warnings.append(f"Skipping OOV query head/tail: ({h!r}, {t!r}).")
                predictions.append(())
                score_rows.append(())
                continue
            scores = score_all_relations(
                plan.method,
                hi,
                ti,
                plan.entity_embeddings_,
                plan.relation_embeddings_,
                norm=_norm(plan),  # type: ignore[arg-type]
            )
            ids, sc = _top_k_filtered(
                scores,
                k=k,
                true_set=true_set,
                head=hi,
                relation=None,
                tail=ti,
                mode="relation",
                keep_true=None,
            )
            predictions.append(tuple(plan.relation_ids[i] for i in ids))
            score_rows.append(tuple(sc))

    return PredictLinksResult(
        mode=mode,
        method=plan.method,
        k=int(k),
        n_queries=len(predictions),
        predictions=tuple(predictions),
        scores=tuple(score_rows),
        query_heads=tuple(q_heads),
        query_relations=tuple(q_rels),
        query_tails=tuple(q_tails),
        filtered=filtered,
        disclosures=(
            f"Link prediction mode={mode!r} over train entity/relation catalog.",
            "Filtered=True removes other train-known true triples from candidates."
            if filtered
            else "Filtered=False ranks raw scores against the full catalog.",
            "Not Graph ML node classification; not RAG retrieve.",
        ),
        warnings=tuple(warnings),
    )
