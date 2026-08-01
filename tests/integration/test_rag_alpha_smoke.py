"""RAG alpha-gate smoke: ingest through hybrid retrieve, eval, upsert, bundle."""

from __future__ import annotations

from pathlib import Path

from buildml import Session


def test_rag_alpha_gate_smoke(tmp_path: Path) -> None:
    docs = [
        {
            "doc_id": "ml",
            "text": (
                "Supervised learning fits a model on labeled examples. "
                "Hold out a test partition for final estimates."
            ),
            "metadata": {"topic": "classical"},
        },
        {
            "doc_id": "rag",
            "text": (
                "Retrieval indexes a corpus, retrieves relevant chunks, "
                "and optionally generates grounded answers later."
            ),
            "metadata": {"topic": "retrieval"},
        },
        {
            "doc_id": "leak",
            "text": (
                "Evaluation contamination happens when labeled answers are "
                "indexed into the retrieval corpus."
            ),
            "metadata": {"topic": "hygiene"},
        },
    ]

    session = Session()
    session.rag_ingest_corpus(docs)
    session.rag_chunk(size=160, overlap=32)
    session.rag_embed_and_index()
    assert session.rag_index_result is not None
    assert session.rag_index_result.embedder_id == "buildml.hashing_embed.v1"
    assert session.rag_index_result.n_chunks >= 3

    dense = session.rag_retrieve("retrieval corpus contamination indexed answers", k=3)
    assert dense.mode == "dense"
    assert len(dense.hits) == 3
    assert dense.hits[0].doc_id in {"ml", "rag", "leak"}

    hybrid = session.rag_retrieve(
        "retrieval corpus contamination indexed answers",
        k=3,
        mode="hybrid",
    )
    assert hybrid.mode == "hybrid"
    assert hybrid.fusion == "rrf"
    assert len(hybrid.hits) == 3

    metrics = session.rag_evaluate(
        {
            "retrieval corpus contamination": ["leak"],
            "supervised learning hold out test": ["ml"],
        },
        k=3,
    )
    assert metrics.n_queries == 2
    assert 0.0 <= metrics.recall_at_k <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
    assert 0.0 <= metrics.ndcg_at_k <= 1.0
    assert 0.0 <= metrics.hit_rate_at_k <= 1.0
    assert session.rag_eval_result is not None

    before_chunks = session.rag_index_result.n_chunks
    session.rag_upsert(
        [{"doc_id": "extra", "text": "Temporary chunk for upsert and delete smoke."}]
    )
    assert session.rag_index_result.n_chunks >= before_chunks
    session.rag_delete(doc_ids=["extra"])
    assert session.rag_index_result.n_chunks == before_chunks

    bundle = session.save_rag_bundle(tmp_path / "rag_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "chunks.jsonl").is_file()
    assert (bundle / "embeddings.npy").is_file()

    restored = Session().load_rag_bundle(bundle)
    assert restored.rag_index_result is not None
    assert restored.rag_index_result.embedder_id == "buildml.hashing_embed.v1"
    again = restored.rag_retrieve("retrieval corpus contamination indexed answers", k=3)
    assert again.hits[0].doc_id == dense.hits[0].doc_id
    assert again.hits[0].chunk_id == dense.hits[0].chunk_id

    before = session.explain("rag_retrieve", moment="before")
    assert before.operation == "rag_retrieve"
    assert before.prerequisite_status.get("rag-index") is True
