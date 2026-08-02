"""Mirror of guides/rag-deep.md — hashing embed retrieve + eval + echo generate."""

from __future__ import annotations

from buildml import Session
from buildml.rag.generate import EchoGroundedProvider


def main() -> None:
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
                "Retrieval indexes a corpus, retrieves relevant chunks, "
                "and optionally generates grounded answers later."
            ),
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

    hybrid = session.rag_retrieve(
        "corpus contamination indexed answers",
        k=3,
        mode="hybrid",
    )
    print("top hit:", hybrid.hits[0].doc_id)

    answer = session.rag_generate(
        "What causes evaluation contamination?",
        provider=EchoGroundedProvider(),
        k=3,
    )
    print("answer:", answer.answer)
    print("citations:", [c.doc_id for c in answer.citations])
    if answer.faithfulness is not None:
        print(
            "faithfulness grounded:",
            answer.faithfulness.grounded,
            "overlap:",
            answer.faithfulness.answer_context_token_overlap,
        )

    metrics = session.rag_evaluate(
        {
            "corpus contamination indexed answers": ["leak"],
            "supervised learning hold out test": ["ml"],
        },
        k=3,
    )
    print(
        "recall@k",
        metrics.recall_at_k,
        "mrr",
        metrics.mrr,
        "ndcg",
        metrics.ndcg_at_k,
    )


if __name__ == "__main__":
    main()
