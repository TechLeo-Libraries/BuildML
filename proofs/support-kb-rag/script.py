"""Tier A proof: support knowledge-base RAG (hashing default; HF when present)."""

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
    load_support_kb_corpus,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("support-kb-rag", seed=5)
    docs, judgments = load_support_kb_corpus()
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

    # Retrieval hygiene: judgments are queries → doc_ids; answers are NOT indexed.
    sample = session.rag_retrieve(
        "evaluation contamination indexed answers",
        k=3,
        mode="hybrid",
    )
    answer = session.rag_generate(
        "What causes evaluation contamination?",
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
                "name": "support_kb_corpus",
                "license": "synthetic/public-domain (generated in-repo)",
                "n_docs": len(docs),
                "n_queries": len(judgments),
                "notes": "Support KB snippets; judgments never indexed as answers.",
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
                "Corpus contains knowledge articles only — not labeled answers",
                "Judgments used solely in rag_evaluate (not indexed)",
                "EchoGroundedProvider for offline generate (no live LLM required)",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn TF-IDF / BM25-style retrieval twin on "
                    "the same corpus; run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Tiny corpus; hashing embeddings are lexical, not semantic SOTA",
                "Echo generate is faithfulness scaffolding, not a production LLM",
            ],
        },
    )
    print(
        "support-kb-rag OK",
        {
            "recall_at_k": metrics.recall_at_k,
            "mrr": metrics.mrr,
            "backend": embed_backend,
        },
    )


if __name__ == "__main__":
    main()
