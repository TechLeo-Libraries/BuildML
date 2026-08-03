"""Tier C: sklearn TF-IDF cosine twin for policy-handbook-rag."""

from __future__ import annotations

import math
import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
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
        },
        {
            "doc_id": "expense-policy",
            "text": (
                "Business expenses under $75 do not require pre-approval. Receipts are "
                "required for all claims over $25. Submit expense reports within 30 days "
                "of the purchase date via the finance portal."
            ),
        },
        {
            "doc_id": "remote-work",
            "text": (
                "Eligible roles may work remotely up to 3 days per week with manager "
                "approval. Core collaboration hours are 10:00–15:00 local time. "
                "Company equipment must be returned within 10 days of role change."
            ),
        },
        {
            "doc_id": "security-access",
            "text": (
                "Production systems require MFA. Access reviews run quarterly. "
                "Shared passwords are prohibited. Report suspected phishing to "
                "security@example.com within 24 hours."
            ),
        },
        {
            "doc_id": "code-of-conduct",
            "text": (
                "Harassment and discrimination are not tolerated. Report concerns to "
                "People Ops or the anonymous ethics hotline. Retaliation against "
                "good-faith reporters is prohibited."
            ),
        },
        {
            "doc_id": "travel-policy",
            "text": (
                "Domestic flights under 4 hours should be economy class. Hotel stays "
                "require a negotiated rate when available. Travel advances need "
                "finance approval 7 days before departure."
            ),
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


def _ndcg_at_k(rels: list[int], k: int) -> float:
    gains = [(2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k])]
    dcg = sum(gains)
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def main() -> None:
    ctx = new_proof_context("policy-handbook-rag", seed=112)
    docs, judgments = _policy_corpus()
    corpus_ids = [d["doc_id"] for d in docs]
    corpus_text = [d["text"] for d in docs]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    doc_mat = vectorizer.fit_transform(corpus_text)

    recalls, mrrs, ndcgs = [], [], []
    k = 3
    for query, relevant in judgments.items():
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, doc_mat).ravel()
        order = np.argsort(-sims)[:k]
        ranked_ids = [corpus_ids[i] for i in order]
        hits = [1 if d in relevant else 0 for d in ranked_ids]
        recalls.append(float(any(hits)))
        rr = 0.0
        for rank, d in enumerate(ranked_ids, start=1):
            if d in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        ndcgs.append(_ndcg_at_k(hits, k))

    industry_metrics = metrics_round(
        {
            "recall_at_k": float(np.mean(recalls)),
            "mrr": float(np.mean(mrrs)),
            "ndcg_at_k": float(np.mean(ndcgs)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml_raw.get("retrieval_metrics", {})))

    write_comparison(
        ctx,
        buildml={
            "backend": f"buildml.rag/{bml_raw.get('embed_backend', 'hashing')}",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.TfidfVectorizer + cosine",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Corpus = policy articles only (no judgment answers indexed)",
                "Judgments used solely for offline retrieval metrics",
                "Same corpus + judgment set as BuildML path",
            ],
        },
        split_counts={"n_docs": len(docs), "n_queries": len(judgments)},
        delta_keys=("recall_at_k", "mrr", "ndcg_at_k"),
    )
    print("policy-handbook-rag Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
