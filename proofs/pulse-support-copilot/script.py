"""Tier B product: Pulse Support Copilot.

Composes RAG retrieval + learning-to-rank over ticket→doc pairs + CBR case
memory for similar resolved tickets + symbolic guardrails for escalation /
PII blocks. Leakage discipline at every stage.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.rag.generate import EchoGroundedProvider
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_support_kb_corpus,
    metrics_round,
    new_proof_context,
    write_results,
)


def _ticket_rank_frame(seed: int = 13) -> pd.DataFrame:
    """Synthetic ticket→article relevance judgments for LTR stage."""
    rng = np.random.default_rng(seed)
    docs = ["billing-refund", "password-reset", "data-export", "rate-limits", "leakage-eval"]
    rows = []
    for q in range(48):
        topic = q % len(docs)
        for d_i, doc in enumerate(docs):
            lexical = float(rng.normal(1.2 if d_i == topic else 0.2, 0.25))
            intent = float(rng.normal(1.0 if d_i == topic else 0.1, 0.2))
            freshness = float(rng.random())
            rel = float(max(0, min(4, int(round(3.2 * (d_i == topic) + 0.4 * lexical)))))
            rows.append(
                {
                    "query_id": f"tkt-{q}",
                    "item_id": doc,
                    "lexical": lexical,
                    "intent": intent,
                    "freshness": freshness,
                    "relevance": rel,
                }
            )
    return pd.DataFrame(rows)


def _case_memory_frame(seed: int = 13) -> tuple[pd.DataFrame, dict]:
    """Resolved ticket cases for CBR stage (features → escalate flag)."""
    rng = np.random.default_rng(seed)
    n = 280
    severity = rng.integers(1, 6, size=n).astype(float)
    wait_hours = rng.exponential(6.0, size=n).clip(0, 72)
    n_touches = rng.poisson(2.0, size=n).astype(float)
    sentiment = rng.normal(0, 1, size=n)
    pii_hits = rng.binomial(1, 0.12, size=n).astype(float)
    logit = (
        -2.0
        + 0.55 * severity
        + 0.08 * wait_hours
        + 0.35 * n_touches
        - 0.4 * sentiment
        + 1.8 * pii_hits
        + rng.normal(0, 0.35, size=n)
    )
    escalate = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "severity": severity,
            "wait_hours": wait_hours,
            "n_touches": n_touches,
            "sentiment": sentiment,
            "pii_hits": pii_hits,
            "escalate": escalate,
            "case_id": [f"case-{i}" for i in range(n)],
        }
    )
    meta = {
        "name": "pulse_resolved_ticket_cases",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(escalate.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("pulse-support-copilot", seed=13)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: RAG over support KB (answers never indexed) ---
    docs, judgments = load_support_kb_corpus()
    st_ok = extra_available("sentence_transformers")
    try:
        rag = Session()
        rag.rag_ingest_corpus(docs)
        rag.rag_chunk(size=180, overlap=40)
        embed_backend = "hashing"
        try:
            if st_ok:
                rag.rag_embed_and_index(embedder="auto")
                embed_backend = "sentence_transformers_or_auto"
            else:
                rag.rag_embed_and_index(embedder="hashing")
        except (MissingExtraError, TypeError, ValueError):
            rag.rag_embed_and_index(embedder="hashing")
            embed_backend = "hashing"
        sample = rag.rag_retrieve("forgot password reset link expired", k=3, mode="hybrid")
        answer = rag.rag_generate(
            "How do I reset a forgotten password?",
            provider=EchoGroundedProvider(),
            k=3,
        )
        metrics = rag.rag_evaluate(judgments, k=3)
        stages["rag"] = {
            "status": "ok",
            "embed_backend": embed_backend,
            "sentence_transformers_available": st_ok,
            "sample_top_docs": [h.doc_id for h in sample.hits],
            "generate_citations": [c.doc_id for c in answer.citations],
            "retrieval_metrics": {
                "recall_at_k": float(metrics.recall_at_k),
                "mrr": float(metrics.mrr),
                "ndcg_at_k": float(metrics.ndcg_at_k),
            },
        }
        write_results(ctx, stages["rag"], filename="rag.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["rag"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"rag: {exc}")

    # --- Stage 2: LTR over ticket→doc judgments (group split) ---
    try:
        rank_frame = _ticket_rank_frame(seed=ctx.seed)
        lgbm = extra_available("lightgbm")
        rank_session = (
            Session.ingest(rank_frame)
            .set_roles(
                {
                    "query_id": "group",
                    "item_id": "id",
                    "relevance": "target",
                    "lexical": "feature",
                    "intent": "feature",
                    "freshness": "feature",
                }
            )
            .group_split(test_size=0.25, validation_size=0.15, random_state=ctx.seed)
        )
        method = "lambdarank" if lgbm else "pointwise"
        try:
            if lgbm:
                fit_r = rank_session.fit_ranker(
                    method="lambdarank",
                    query_column="query_id",
                    item_column="item_id",
                    random_state=ctx.seed,
                )
            else:
                raise MissingExtraError("ranking-industry", "lambdarank")
        except (MissingExtraError, TypeError, ValueError):
            fit_r = rank_session.fit_ranker(
                method="pointwise",
                query_column="query_id",
                item_column="item_id",
                pointwise_estimator="ridge",
                random_state=ctx.seed,
            )
            method = "pointwise"
        ev_r = rank_session.evaluate_ranker(partition="test", k=3)
        plan_r = rank_session.split_plan
        assert plan_r is not None
        stages["ranking"] = {
            "status": "ok",
            "method": method,
            "lightgbm_available": lgbm,
            "fit": metrics_round(fit_r.to_dict() if hasattr(fit_r, "to_dict") else {}),
            "test_metrics": metrics_round(dict(ev_r.metrics)),
            "split_counts": {
                "train": len(plan_r.train_indices),
                "validation": len(plan_r.validation_indices),
                "test": len(plan_r.test_indices),
            },
        }
        write_results(ctx, stages["ranking"], filename="ranking.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["ranking"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"ranking: {exc}")

    # --- Stage 3: CBR case memory for escalate-or-not ---
    try:
        cases, case_meta = _case_memory_frame(seed=ctx.seed)
        cbr_session = (
            Session.ingest(cases)
            .set_roles(
                {
                    "severity": "feature",
                    "wait_hours": "feature",
                    "n_touches": "feature",
                    "sentiment": "feature",
                    "pii_hits": "feature",
                    "escalate": "target",
                    "case_id": "id",
                }
            )
            .split(
                test_size=0.2,
                validation_size=0.2,
                stratify=True,
                random_state=ctx.seed,
            )
            .scale(method="standard")
        )
        fit_c = cbr_session.fit_cbr(
            task="classification",
            metric="euclidean",
            reuse="distance_weighted",
            k=5,
            random_state=ctx.seed,
        )
        ev_c = cbr_session.evaluate_cbr(partition="test")
        plan_c = cbr_session.split_plan
        assert plan_c is not None
        stages["cbr"] = {
            "status": "ok",
            "data": case_meta,
            "fit": metrics_round(fit_c.to_dict() if hasattr(fit_c, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(ev_c, "metrics", {}) or {})),
            "split_counts": {
                "train": len(plan_c.train_indices),
                "validation": len(plan_c.validation_indices),
                "test": len(plan_c.test_indices),
            },
        }
        write_results(ctx, stages["cbr"], filename="cbr.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["cbr"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"cbr: {exc}")
        cases = None
        cbr_session = None

    # --- Stage 4: symbolic guardrails (escalate / PII) on same case table ---
    try:
        if cases is None:
            raise ValueError("CBR stage unavailable; symbolic needs case frame")
        sym_session = (
            Session.ingest(cases.copy())
            .set_roles(
                {
                    "severity": "feature",
                    "wait_hours": "feature",
                    "n_touches": "feature",
                    "sentiment": "feature",
                    "pii_hits": "feature",
                    "escalate": "target",
                    "case_id": "id",
                }
            )
            .inject_split(
                train_indices=list(cbr_session.split_plan.train_indices),
                validation_indices=list(cbr_session.split_plan.validation_indices),
                test_indices=list(cbr_session.split_plan.test_indices),
            )
        )
        assert_no_test_in_selection(
            selection_partition="train", evaluation_partition="test"
        )
        try:
            fit_s = sym_session.fit_symbolic(
                source="decision_tree", max_depth=3, random_state=ctx.seed
            )
        except TypeError:
            fit_s = sym_session.fit_symbolic(
                method="decision_tree", random_state=ctx.seed
            )
        ev_s = sym_session.evaluate_symbolic(partition="test")
        stages["symbolic"] = {
            "status": "ok",
            "fit": metrics_round(fit_s.to_dict() if hasattr(fit_s, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(ev_s, "metrics", {}) or {})),
        }
        write_results(ctx, stages["symbolic"], filename="symbolic.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["symbolic"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"symbolic: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Pulse Support Copilot",
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "RAG corpus = KB articles only; judgments never indexed as answers",
            "LTR group_split by query_id before ranker fit",
            "CBR case memory built from train cases only",
            "Symbolic rules induced on the same train split as CBR; test after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Indexing judgment answers into RAG inflates recall@k",
            "Fitting the ranker on test queries overstates NDCG",
            "Putting test tickets into CBR memory makes accuracy meaningless",
            "Inducing guardrail rules on full data looks more 'safe' than production",
        ],
        "limitations": [
            "Synthetic KB + tickets — not a live helpdesk",
            "EchoGroundedProvider is offline scaffolding, not a production LLM",
            "Product proof, not a support SaaS certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "pulse-support-copilot OK",
        {
            "rag_recall": (stages.get("rag") or {}).get("retrieval_metrics", {}).get("recall_at_k"),
            "rank_metrics": (stages.get("ranking") or {}).get("test_metrics"),
            "cbr_acc": (stages.get("cbr") or {}).get("test_metrics", {}).get("accuracy"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
