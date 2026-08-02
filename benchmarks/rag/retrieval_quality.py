"""RAG retrieval quality benchmark: hashing vs ST vs hybrid+rerank."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from buildml.rag.corpus import corpus_from_documents
from buildml.rag.embed import HashingEmbedder, resolve_embedder
from buildml.rag.evaluate import evaluate_retrieval
from buildml.rag.index import build_index
from buildml.rag.results import Document
from buildml.rag.types import RetrieveConfig

# Small in-repo corpus for CI — no external download required for hashing path.
_BENCH_DOCS = (
    Document(
        doc_id="ml",
        text=(
            "Supervised learning fits models on labeled examples. "
            "Hold out a test partition for unbiased estimates."
        ),
        metadata={"topic": "ml"},
    ),
    Document(
        doc_id="rag",
        text=(
            "Retrieval augmented generation indexes a corpus, retrieves chunks, "
            "and generates grounded answers with citations."
        ),
        metadata={"topic": "rag"},
    ),
    Document(
        doc_id="hygiene",
        text=(
            "Evaluation contamination happens when labeled answers are indexed "
            "into the retrieval corpus."
        ),
        metadata={"topic": "hygiene"},
    ),
    Document(
        doc_id="rust",
        text="Rust is a systems programming language focused on safety and concurrency.",
        metadata={"topic": "lang"},
    ),
    Document(
        doc_id="python",
        text="Python is a popular language for data science and machine learning workflows.",
        metadata={"topic": "lang"},
    ),
)

_QRELS = {
    "evaluation contamination indexed corpus": ["hygiene"],
    "supervised learning labeled examples": ["ml"],
    "retrieval augmented generation citations": ["rag"],
    "systems programming language safety": ["rust"],
    "data science machine learning python": ["python"],
}

# Metric floors for the hashing baseline on this tiny corpus (CI gate).
_HASH_RECALL_FLOOR = 0.4
_HASH_MRR_FLOOR = 0.3


def _rag_usable() -> bool:
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        from buildml.rag.extras import rag_available

        return rag_available()
    except Exception:
        return False


def _run_config(name: str, *, embedder: object, retrieve: RetrieveConfig) -> dict:
    corpus = corpus_from_documents(list(_BENCH_DOCS))
    index = build_index(
        corpus,
        embedder=embedder,
        chunk_size=256,
        chunk_overlap=32,
    )
    metrics = evaluate_retrieval(
        index,
        _QRELS,
        k=retrieve.k,
        retrieve_config=retrieve,
    )
    return {
        "name": name,
        "embedder_id": index.embed_config.embedder_id,
        "retrieve_mode": metrics.retrieve_mode,
        "recall_at_k": metrics.recall_at_k,
        "mrr": metrics.mrr,
        "ndcg_at_k": metrics.ndcg_at_k,
        "hit_rate_at_k": metrics.hit_rate_at_k,
        "n_chunks": index.to_index_result().n_chunks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML RAG retrieval quality benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/rag/results/retrieval_quality.json"),
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    rows.append(
        _run_config(
            "hashing-dense",
            embedder=HashingEmbedder(n_features=256),
            retrieve=RetrieveConfig(k=args.k, mode="dense"),
        )
    )
    rows.append(
        _run_config(
            "hashing-hybrid",
            embedder=HashingEmbedder(n_features=256),
            retrieve=RetrieveConfig(k=args.k, mode="hybrid", fusion="rrf"),
        )
    )

    if _rag_usable():
        try:
            st_embedder, _ = resolve_embedder("sentence-transformers")
            rows.append(
                _run_config(
                    "st-dense",
                    embedder=st_embedder,
                    retrieve=RetrieveConfig(k=args.k, mode="dense"),
                )
            )
            rows.append(
                _run_config(
                    "st-hybrid",
                    embedder=st_embedder,
                    retrieve=RetrieveConfig(k=args.k, mode="hybrid", fusion="rrf"),
                )
            )
            rows.append(
                _run_config(
                    "st-hybrid-rerank",
                    embedder=st_embedder,
                    retrieve=RetrieveConfig(
                        k=args.k,
                        mode="hybrid",
                        fusion="rrf",
                        rerank=True,
                        rerank_candidates=max(12, args.k * 4),
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 — optional model download
            rows.append({"name": "st-skipped", "error": str(exc)})
    else:
        rows.append({"name": "st-skipped", "reason": "buildml[rag] not installed"})

    hashing_row = next(r for r in rows if r["name"] == "hashing-hybrid")
    passed = (
        hashing_row.get("recall_at_k", 0.0) >= _HASH_RECALL_FLOOR
        and hashing_row.get("mrr", 0.0) >= _HASH_MRR_FLOOR
    )

    payload = {
        "benchmark": "rag_retrieval_quality",
        "k": args.k,
        "passed": passed,
        "floors": {"hashing_hybrid_recall_at_k": _HASH_RECALL_FLOOR, "hashing_hybrid_mrr": _HASH_MRR_FLOOR},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
