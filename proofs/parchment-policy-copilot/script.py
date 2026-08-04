"""Tier B product: Parchment Policy Copilot.

Composes RAG retrieval/generate + learning-to-rank over policy queries +
CBR case memory for similar prior escalations.
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
    extra_available,
    metrics_round,
    new_proof_context,
    write_results,
)


def _policy_corpus() -> tuple[list[dict], dict[str, list[str]]]:
    docs = [
        {
            "doc_id": "leave-policy",
            "text": (
                "Employees accrue 15 days of paid time off per year. Requests must be "
                "submitted in the HR portal at least 5 business days in advance. "
                "Unused PTO may roll over up to 5 days into the next calendar year."
            ),
            "metadata": {"topic": "leave"},
        },
        {
            "doc_id": "expense-policy",
            "text": (
                "Business expenses under $75 do not require pre-approval. Receipts are "
                "required for all claims over $25. Submit expense reports within 30 days "
                "of the purchase date via the finance portal."
            ),
            "metadata": {"topic": "finance"},
        },
        {
            "doc_id": "remote-work",
            "text": (
                "Eligible roles may work remotely up to 3 days per week with manager "
                "approval. Core collaboration hours are 10:00–15:00 local time. "
                "Company equipment must be returned within 10 days of role change."
            ),
            "metadata": {"topic": "workplace"},
        },
        {
            "doc_id": "security-access",
            "text": (
                "Production systems require MFA. Access reviews run quarterly. "
                "Shared passwords are prohibited. Report suspected phishing to "
                "security@example.com within 24 hours."
            ),
            "metadata": {"topic": "security"},
        },
        {
            "doc_id": "travel-policy",
            "text": (
                "Domestic flights under 4 hours should be economy class. Hotel stays "
                "require a negotiated rate when available. Travel advances need "
                "finance approval 7 days before departure."
            ),
            "metadata": {"topic": "travel"},
        },
        {
            "doc_id": "escalation-matrix",
            "text": (
                "Policy exceptions escalate to People Ops for leave, Finance for "
                "expenses, and Security for access. Dual approval is required when "
                "exceptions exceed two policy dimensions."
            ),
            "metadata": {"topic": "escalation"},
        },
    ]
    judgments = {
        "How many PTO days do employees accrue?": ["leave-policy"],
        "When are receipts required for expenses?": ["expense-policy"],
        "What are core collaboration hours for remote work?": ["remote-work"],
        "How do I report phishing?": ["security-access"],
        "What class of flight for short domestic trips?": ["travel-policy"],
        "Who approves multi-dimension policy exceptions?": ["escalation-matrix"],
    }
    return docs, judgments


def _rank_frame(seed: int = 53) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    docs = [
        "leave-policy",
        "expense-policy",
        "remote-work",
        "security-access",
        "travel-policy",
        "escalation-matrix",
    ]
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
                    "query_id": f"pol-q-{q}",
                    "item_id": doc,
                    "lexical": lexical,
                    "intent": intent,
                    "freshness": freshness,
                    "relevance": rel,
                }
            )
    return pd.DataFrame(rows)


def _case_memory(n: int = 300, seed: int = 53) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    severity = rng.integers(1, 6, size=n).astype(float)
    wait_hours = rng.exponential(5.0, size=n).clip(0, 72)
    n_policy_hits = rng.poisson(1.5, size=n).astype(float)
    ambiguity = rng.beta(2, 3, size=n)
    logit = (
        -1.8
        + 0.5 * severity
        + 0.07 * wait_hours
        + 0.4 * n_policy_hits
        + 1.2 * ambiguity
        + rng.normal(0, 0.3, size=n)
    )
    escalate = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "severity": severity,
            "wait_hours": wait_hours,
            "n_policy_hits": n_policy_hits,
            "ambiguity": ambiguity,
            "escalate": escalate,
            "case_id": [f"case-{i}" for i in range(n)],
        }
    )
    meta = {
        "name": "parchment_policy_cases",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(escalate.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("parchment-policy-copilot", seed=53)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: RAG ---
    docs, judgments = _policy_corpus()
    try:
        session = Session()
        session.rag.ingest_corpus(docs)
        session.rag.chunk(size=180, overlap=40)
        embed_backend = "hashing"
        try:
            if extra_available("sentence_transformers"):
                session.rag.embed_and_index(embedder="auto")
                embed_backend = "sentence_transformers_or_auto"
            else:
                session.rag.embed_and_index(embedder="hashing")
        except (MissingExtraError, TypeError, ValueError):
            session.rag.embed_and_index(embedder="hashing")
            embed_backend = "hashing"
        sample = session.rag.retrieve(
            "How many paid time off days accrue each year?", k=3, mode="hybrid"
        )
        answer = session.rag.generate(
            "What is the PTO accrual policy?",
            provider=EchoGroundedProvider(),
            k=3,
        )
        metrics = session.rag.evaluate(judgments, k=3)
        stages["rag"] = {
            "status": "ok",
            "embed_backend": embed_backend,
            "sample_top_docs": [h.doc_id for h in sample.hits],
            "generate": {
                "answer": answer.answer,
                "citations": [c.doc_id for c in answer.citations],
            },
            "retrieval_metrics": {
                "recall_at_k": float(metrics.recall_at_k),
                "mrr": float(metrics.mrr),
                "ndcg_at_k": float(metrics.ndcg_at_k),
            },
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["rag"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"rag: {exc}")
    write_results(ctx, stages["rag"], filename="rag.json")

    # --- Stage 2: ranking ---
    rank_frame = _rank_frame(seed=ctx.seed)
    try:
        ltr = (
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
        method = "lambdarank" if extra_available("lightgbm") else "pointwise"
        try:
            if method == "lambdarank":
                rk_fit = ltr.ranking.fit(
                    method="lambdarank",
                    query_column="query_id",
                    item_column="item_id",
                    random_state=ctx.seed,
                )
            else:
                raise MissingExtraError("ranking-industry", "lambdarank")
        except (MissingExtraError, TypeError, ValueError):
            rk_fit = ltr.ranking.fit(
                method="pointwise",
                query_column="query_id",
                item_column="item_id",
                pointwise_estimator="ridge",
                random_state=ctx.seed,
            )
            method = "pointwise"
        rk_val = ltr.ranking.evaluate(partition="validation", k=5)
        rk_test = ltr.ranking.evaluate(partition="test", k=5)
        stages["ranking"] = {
            "status": "ok",
            "method": method,
            "fit": metrics_round(rk_fit.to_dict() if hasattr(rk_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(rk_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(rk_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["ranking"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"ranking: {exc}")
    write_results(ctx, stages["ranking"], filename="ranking.json")

    # --- Stage 3: CBR case memory ---
    cases, case_meta = _case_memory(seed=ctx.seed)
    try:
        session = (
            Session.ingest(cases)
            .set_roles(
                {
                    "severity": "feature",
                    "wait_hours": "feature",
                    "n_policy_hits": "feature",
                    "ambiguity": "feature",
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
        plan = session.split_plan
        assert plan is not None
        split_counts = {
            "train": len(plan.train_indices),
            "validation": len(plan.validation_indices),
            "test": len(plan.test_indices),
        }
        c_fit = session.cbr.fit(
            task="classification",
            metric="euclidean",
            reuse="distance_weighted",
            k=5,
            random_state=ctx.seed,
        )
        c_val = session.cbr.evaluate(partition="validation")
        c_test = session.cbr.evaluate(partition="test")
        stages["cbr"] = {
            "status": "ok",
            "data": case_meta,
            "fit": metrics_round(c_fit.to_dict() if hasattr(c_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(c_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(c_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["cbr"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"cbr: {exc}")
        plan = None
        split_counts = {}
    write_results(ctx, stages["cbr"], filename="cbr.json")

    summary = {
        "status": "completed",
        "product": "Parchment Policy Copilot",
        "data": {
            "rag_docs": len(docs),
            "rank_rows": int(len(rank_frame)),
            "cases": case_meta,
        },
        "split": {
            "kind": getattr(plan, "kind", None),
            "counts": split_counts,
            "stratify": True,
        },
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "RAG corpus contains policy articles only — judgments never indexed",
            "LTR group_split on query_id before fit",
            "CBR case memory built from train only",
            "Test retrieval / session.ranking.rank / CBR metrics after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Indexing labeled answers into the corpus turns RAG eval into a lookup",
            "Query leakage in LTR inflates nDCG on held-out policy questions",
            "CBR memory that includes test cases is not a fair retrieve-and-reuse bench",
        ],
        "limitations": [
            "Tiny handbook corpus; Echo generate is offline scaffolding",
            "CBR ≠ RAG; synthetic escalation cases",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "parchment-policy-copilot OK",
        {
            "rag_recall": stages.get("rag", {}).get("retrieval_metrics", {}).get(
                "recall_at_k"
            ),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
