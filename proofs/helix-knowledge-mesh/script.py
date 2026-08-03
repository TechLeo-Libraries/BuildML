"""Tier B product: Helix Knowledge Mesh.

Composes knowledge-graph link prediction + RAG retrieval/generate + symbolic
guardrails for grounded policy answers. Leakage discipline at every stage.
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


def _enterprise_kg(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"team{i}" for i in range(12)]
    systems = [f"sys{i}" for i in range(10)]
    policies = [f"pol{i}" for i in range(8)]
    owners = [f"own{i}" for i in range(6)]
    triples = []
    for i, t in enumerate(teams):
        triples.append((t, "owns", systems[i % len(systems)]))
        triples.append((systems[i % len(systems)], "governed_by", policies[i % len(policies)]))
        triples.append((owners[i % len(owners)], "stewards", policies[i % len(policies)]))
        triples.append((t, "reports_to", owners[i % len(owners)]))
    for _ in range(30):
        a, b = rng.choice(systems, size=2, replace=False)
        triples.append((str(a), "depends_on", str(b)))
    return (
        pd.DataFrame(triples, columns=["head", "relation", "tail"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _mesh_corpus() -> tuple[list[dict], dict[str, list[str]]]:
    docs = [
        {
            "doc_id": "access-review",
            "text": (
                "Access reviews run quarterly for production systems. Owners must "
                "attest entitlements within 10 business days. Orphaned accounts are "
                "disabled after 14 days of non-response."
            ),
            "metadata": {"topic": "access"},
        },
        {
            "doc_id": "data-retention",
            "text": (
                "Customer logs are retained for 90 days online and 365 days in cold "
                "storage. Legal holds override retention timers. Purge jobs run nightly."
            ),
            "metadata": {"topic": "retention"},
        },
        {
            "doc_id": "change-freeze",
            "text": (
                "A change freeze applies during peak sales windows. Emergency changes "
                "require dual approval from on-call and a steward. Post-mortems are due "
                "within 48 hours."
            ),
            "metadata": {"topic": "change"},
        },
        {
            "doc_id": "vendor-risk",
            "text": (
                "Critical vendors require annual SOC2 review. Data processing agreements "
                "must list subprocessors. Residual risk above medium escalates to the "
                "security committee."
            ),
            "metadata": {"topic": "vendor"},
        },
        {
            "doc_id": "incident-severity",
            "text": (
                "Severity-1 incidents page the incident commander within 5 minutes. "
                "Customer-facing outages are Sev-1 by default. Status updates every "
                "30 minutes until mitigated."
            ),
            "metadata": {"topic": "incident"},
        },
        {
            "doc_id": "model-governance",
            "text": (
                "Production models require a model card, schema contract, and leakage "
                "checklist. Retrains need validation metrics vs the frozen holdout. "
                "Shadow deploy for 7 days before cutover."
            ),
            "metadata": {"topic": "ml"},
        },
    ]
    judgments = {
        "How often are access reviews run?": ["access-review"],
        "How long are customer logs retained online?": ["data-retention"],
        "Who must approve emergency changes in a freeze?": ["change-freeze"],
        "What review do critical vendors need annually?": ["vendor-risk"],
        "When do Sev-1 incidents page the commander?": ["incident-severity"],
        "What artifacts are required for production models?": ["model-governance"],
    }
    return docs, judgments


def _guardrail_frame(n: int = 420, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    risk = rng.beta(2, 4, size=n)
    pii_hits = rng.binomial(1, 0.15, size=n).astype(float)
    ungrounded = rng.binomial(1, 0.2, size=n).astype(float)
    severity = rng.integers(1, 6, size=n).astype(float)
    block = (
        ((risk > 0.55) & (ungrounded == 1))
        | ((pii_hits == 1) & (severity >= 3))
    ).astype(int)
    frame = pd.DataFrame(
        {
            "risk_score": risk,
            "pii_hits": pii_hits,
            "ungrounded": ungrounded,
            "severity": severity,
            "block_answer": block,
        }
    )
    meta = {
        "name": "helix_answer_guardrails",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(block.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("helix-knowledge-mesh", seed=42)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: KG link prediction ---
    kg_frame = _enterprise_kg(seed=ctx.seed)
    try:
        kg_session = (
            Session.ingest(kg_frame)
            .set_roles({"head": "id", "relation": "id", "tail": "id"})
            .split(test_size=0.2, validation_size=0.1, random_state=ctx.seed)
        )
        kg_fit = kg_session.fit_kg(
            method="transe",
            head_column="head",
            relation_column="relation",
            tail_column="tail",
            embedding_dim=32,
            epochs=40,
            batch_size=64,
            learning_rate=0.05,
            neg_ratio=2,
            random_state=ctx.seed,
        )
        kg_val = kg_session.evaluate_kg(partition="validation")
        kg_test = kg_session.evaluate_kg(partition="test")
        stages["kg"] = {
            "status": "ok",
            "n_triples": int(len(kg_frame)),
            "fit": metrics_round(kg_fit.to_dict() if hasattr(kg_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(kg_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(kg_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["kg"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"kg: {exc}")
    write_results(ctx, stages["kg"], filename="kg.json")

    # --- Stage 2: RAG over mesh handbook ---
    docs, judgments = _mesh_corpus()
    try:
        rag = Session()
        rag.rag_ingest_corpus(docs)
        rag.rag_chunk(size=180, overlap=40)
        embed_backend = "hashing"
        try:
            if extra_available("sentence_transformers"):
                rag.rag_embed_and_index(embedder="auto")
                embed_backend = "sentence_transformers_or_auto"
            else:
                rag.rag_embed_and_index(embedder="hashing")
        except (MissingExtraError, TypeError, ValueError):
            rag.rag_embed_and_index(embedder="hashing")
            embed_backend = "hashing"
        sample = rag.rag_retrieve(
            "How often are access reviews run for production systems?",
            k=3,
            mode="hybrid",
        )
        answer = rag.rag_generate(
            "What is the access review cadence?",
            provider=EchoGroundedProvider(),
            k=3,
        )
        metrics = rag.rag_evaluate(judgments, k=3)
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

    # --- Stage 3: symbolic answer guardrails ---
    guard_frame, guard_meta = _guardrail_frame(seed=ctx.seed)
    try:
        sym_session = (
            Session.ingest(guard_frame)
            .set_roles(
                {
                    "risk_score": "feature",
                    "pii_hits": "feature",
                    "ungrounded": "feature",
                    "severity": "feature",
                    "block_answer": "target",
                }
            )
            .split(
                test_size=0.2,
                validation_size=0.2,
                stratify=True,
                random_state=ctx.seed,
            )
        )
        plan = sym_session.split_plan
        assert plan is not None
        split_counts = {
            "train": len(plan.train_indices),
            "validation": len(plan.validation_indices),
            "test": len(plan.test_indices),
        }
        sym = sym_session.fit_symbolic(
            source="decision_tree",
            max_depth=3,
            random_state=ctx.seed,
        )
        sym_val = sym_session.evaluate_symbolic(partition="validation")
        sym_test = sym_session.evaluate_symbolic(partition="test")
        stages["symbolic"] = {
            "status": "ok",
            "data": guard_meta,
            "fit": metrics_round(sym.to_dict() if hasattr(sym, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(sym_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(sym_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["symbolic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"symbolic: {exc}")
        plan = None
        split_counts = {}
    write_results(ctx, stages["symbolic"], filename="symbolic.json")

    summary = {
        "status": "completed",
        "product": "Helix Knowledge Mesh",
        "data": {
            "kg_triples": int(len(kg_frame)),
            "rag_docs": len(docs),
            "guardrails": guard_meta,
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
            "KG triple split before TransE fit",
            "RAG corpus contains policy articles only — judgments never indexed",
            "Symbolic guardrails fit on train; test after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Training TransE on all triples makes link metrics meaningless",
            "Indexing labeled answers into the corpus turns RAG eval into a lookup",
            "Inducing guardrail rules on the full table overstates compliance",
        ],
        "limitations": [
            "Synthetic enterprise mesh — not a licensed CMDB / handbook extract",
            "EchoGroundedProvider is offline scaffolding, not a production LLM",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "helix-knowledge-mesh OK",
        {
            "rag_recall": stages.get("rag", {}).get("retrieval_metrics", {}).get(
                "recall_at_k"
            ),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
