"""BEIR SciFact document retrieval against official test qrels and a TF-IDF baseline."""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from benchmarks.public_data import public_file
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.embed import HashingEmbedder
from buildml.rag.index import build_index
from buildml.rag.results import Document
from buildml.rag.retrieve import retrieve
from buildml.rag.types import RetrieveConfig

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def ranking_metrics(ranking: list[str], relevant: dict[str, int], k: int) -> dict[str, float]:
    """Calculate graded document nDCG and binary relevance recall/MRR after deduplication."""
    ranking = list(dict.fromkeys(ranking))[:k]
    positive = {doc: grade for doc, grade in relevant.items() if grade > 0}
    gains = [(2 ** positive.get(doc, 0) - 1) / np.log2(i + 2) for i, doc in enumerate(ranking)]
    ideal = sorted(positive.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / np.log2(i + 2) for i, grade in enumerate(ideal))
    return {"ndcg": float(sum(gains) / idcg) if idcg else 0.0,
            "recall": len(set(ranking) & set(positive)) / len(positive) if positive else 0.0,
            "mrr": next((1 / (i + 1) for i, doc in enumerate(ranking) if doc in positive), 0.0)}


def main() -> int:
    """Evaluate held-out claims without indexing claims or relevance judgements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/public-data/scifact.zip"))
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmarks/beir-scifact.json"))
    args = parser.parse_args()
    if args.queries < 1 or args.k < 1:
        raise ValueError("--queries and --k must be positive")
    source = public_file(URL, args.cache, download=args.download)
    with zipfile.ZipFile(args.cache) as archive:
        corpus = [json.loads(line) for line in archive.read("scifact/corpus.jsonl").splitlines()]
        queries = {str(row["_id"]): row["text"] for row in
                   (json.loads(line) for line in archive.read("scifact/queries.jsonl").splitlines())}
        qrels: dict[str, dict[str, int]] = {}
        for row in csv.DictReader(io.StringIO(archive.read("scifact/qrels/test.tsv").decode()), delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    selected = sorted(qrels)[:args.queries]
    documents = [Document(doc_id=str(d["_id"]), text=d["title"] + "\n" + d["text"], metadata={}) for d in corpus]
    # One chunk per abstract: rankings and official qrels share document IDs.
    index = build_index(corpus_from_documents(documents), embedder=HashingEmbedder(n_features=2048),
                        chunk_size=max(len(d.text) for d in documents) + 1, chunk_overlap=0)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform([d.text for d in documents])
    query_vectors = vectorizer.transform([queries[q] for q in selected])
    rows = []
    for mode in ("tfidf_baseline", "bm25", "dense", "hybrid"):
        for position, query_id in enumerate(selected):
            started = time.perf_counter()
            if mode == "tfidf_baseline":
                similarity = (matrix @ query_vectors[position].T).toarray().ravel()
                order = np.argsort(-similarity, kind="stable")[:args.k]
                ranking = [documents[i].doc_id for i in order]
                actual_mode = mode
            else:
                result = retrieve(index, queries[query_id], config=RetrieveConfig(mode=mode, k=args.k))
                ranking = [hit.doc_id for hit in result.hits]
                actual_mode = result.mode
                if actual_mode != mode:
                    raise AssertionError(f"Requested {mode}, got fallback {actual_mode}")
            rows.append({"query": query_id, "method": mode, "actual_mode": actual_mode,
                         "seconds": time.perf_counter() - started,
                         **ranking_metrics(ranking, qrels[query_id], args.k)})
    summary = {mode: {metric: float(np.mean([r[metric] for r in rows if r["method"] == mode]))
                      for metric in ("ndcg", "recall", "mrr", "seconds")}
               for mode in sorted({r["method"] for r in rows})}
    payload = {"dataset": "BEIR SciFact", "source": source, "n_documents": len(documents),
               "n_queries": len(selected), "available_test_queries": len(qrels), "k": args.k,
               "protocol": "Official test qrels; full corpus; one chunk/document; no queries or qrels indexed",
               "summary": summary, "per_query": rows,
               "limitations": ["Hashing embeddings are lexical, not a trained semantic model.",
                               "Single domain; these measurements do not establish universal superiority.",
                               "No hyperparameter selection on test qrels."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
