"""Tier A proof: policy handbook RAG (hashing default; HF when present)."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

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
            "doc_id": "code-of-conduct",
            "text": (
                "Harassment and discrimination are not tolerated. Report concerns to "
                "People Ops or the anonymous ethics hotline. Retaliation against "
                "good-faith reporters is prohibited."
            ),
            "metadata": {"topic": "conduct"},
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
    ]
    judgments = {
        "How many PTO days do employees accrue?": ["leave-policy"],
        "When are receipts required for expenses?": ["expense-policy"],
        "What are core collaboration hours for remote work?": ["remote-work"],
        "How do I report phishing?": ["security-access"],
        "Where do I report harassment?": ["code-of-conduct"],
        "What class of flight for short domestic trips?": ["travel-policy"],
    }
    return docs, judgments


def main() -> None:
    ctx = new_proof_context("policy-handbook-rag", seed=112)
    docs, judgments = _policy_corpus()
    st_ok = extra_available("sentence_transformers")

    session = Session()
    session.rag_ingest_corpus(docs)
    session.rag_chunk(size=180, overlap=40)

    embed_backend = "hashing"
    try:
        if st_ok:
            session.rag_embed_and_index(embedder="auto")
            embed_backend = "sentence_transformers_or_auto"
        else:
            session.rag_embed_and_index(embedder="hashing")
            embed_backend = "hashing"
    except (MissingExtraError, TypeError, ValueError):
        session.rag_embed_and_index(embedder="hashing")
        embed_backend = "hashing"

    sample = session.rag_retrieve(
        "How many paid time off days accrue each year?",
        k=3,
        mode="hybrid",
    )
    answer = session.rag_generate(
        "What is the PTO accrual policy?",
        provider=EchoGroundedProvider(),
        k=3,
    )
    metrics = session.rag_evaluate(judgments, k=3)
    try:
        bundle = session.save_rag_bundle(ctx.artifacts_dir / "rag_bundle")
        bundle_path = str(bundle)
    except Exception as exc:  # noqa: BLE001
        bundle_path = f"unavailable: {type(exc).__name__}: {exc}"

    write_results(
        ctx,
        {
            "status": "completed",
            "data": {
                "name": "policy_handbook_corpus",
                "license": "synthetic/public-domain (generated in-repo)",
                "n_docs": len(docs),
                "n_queries": len(judgments),
                "notes": "Policy handbook snippets; judgments never indexed as answers.",
            },
            "embed_backend": embed_backend,
            "sentence_transformers_available": st_ok,
            "sample_retrieve": {
                "top_doc_ids": [h.doc_id for h in sample.hits],
                "mode": "hybrid",
            },
            "generate": {
                "answer": answer.answer,
                "citations": [c.doc_id for c in answer.citations],
                "faithfulness": metrics_round(
                    answer.faithfulness.to_dict()
                    if getattr(answer, "faithfulness", None) is not None
                    and hasattr(answer.faithfulness, "to_dict")
                    else {}
                ),
            },
            "retrieval_metrics": {
                "recall_at_k": float(metrics.recall_at_k),
                "mrr": float(metrics.mrr),
                "ndcg_at_k": float(metrics.ndcg_at_k),
            },
            "bundle_path": bundle_path,
            "leakage_controls": [
                "Corpus contains policy articles only — not labeled answers",
                "Judgments used solely in rag_evaluate (not indexed)",
                "EchoGroundedProvider for offline generate (no live LLM required)",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn TF-IDF cosine twin on the same "
                    "corpus; run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Tiny handbook corpus; hashing embeddings are lexical, not semantic SOTA",
                "Echo generate is faithfulness scaffolding, not a production LLM",
            ],
        },
    )
    print(
        "policy-handbook-rag OK",
        {
            "recall_at_k": metrics.recall_at_k,
            "mrr": metrics.mrr,
            "backend": embed_backend,
        },
    )


if __name__ == "__main__":
    main()
