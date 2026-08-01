"""Integration smoke for the RAG Session vertical slice."""

from __future__ import annotations

from buildml import Session


def test_session_rag_vertical_slice(tmp_path) -> None:
    docs = [
        {
            "doc_id": "ml",
            "text": (
                "Supervised learning fits a model on labeled examples. "
                "Hold out a test partition for final estimates."
            ),
        },
        {
            "doc_id": "rag",
            "text": (
                "Retrieval augmented generation indexes a corpus, retrieves "
                "relevant chunks, and optionally generates grounded answers."
            ),
        },
        {
            "doc_id": "leak",
            "text": (
                "Evaluation contamination happens when labeled answers are "
                "indexed into the retrieval corpus."
            ),
        },
    ]
    session = Session()
    session.rag_ingest_corpus(docs)
    session.rag_chunk(size=160, overlap=32)
    session.rag_embed_and_index()  # default hashing embedder
    assert session.rag_index_result is not None
    assert session.rag_index_result.embedder_id == "buildml.hashing_embed.v1"

    retrieved = session.rag_retrieve("retrieval corpus contamination indexed answers", k=3)
    assert len(retrieved.hits) == 3
    assert retrieved.hits[0].doc_id in {"ml", "rag", "leak"}

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

    bundle = session.save_rag_bundle(tmp_path / "rag_bundle")
    restored = Session().load_rag_bundle(bundle)
    again = restored.rag_retrieve("retrieval corpus contamination indexed answers", k=3)
    assert again.hits[0].doc_id == retrieved.hits[0].doc_id

    before = session.explain("rag_retrieve", moment="before")
    assert before.operation == "rag_retrieve"
    assert before.prerequisite_status.get("rag-index") is True
