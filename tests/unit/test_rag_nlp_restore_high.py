"""HIGH-depth RAG / NLP bundle restore roundtrips (facade + library paths)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.nlp.checkpoint import load_nlp_bundle, save_nlp_bundle
from buildml.rag.checkpoint import load_rag_bundle, save_rag_bundle
from buildml.rag.embed import HashingEmbedder
from buildml.rag.index import build_index
from buildml.rag.retrieve import retrieve


def test_rag_facade_bundle_restore_roundtrip(tmp_path: Path) -> None:
    session = Session()
    session.rag.ingest_corpus(
        [
            {"doc_id": "a", "text": "alpha retrieval document about cats"},
            {"doc_id": "b", "text": "beta retrieval document about dogs"},
            {"doc_id": "c", "text": "gamma retrieval document about birds"},
        ]
    )
    session.rag.chunk(size=128, overlap=0)
    session.rag.embed_and_index(embedder="hashing")
    before = session.rag.retrieve("dogs document", k=2)
    path = session.rag.save_bundle(tmp_path / "rag_facade")
    other = Session()
    other.rag.load_bundle(path)
    assert other.rag.index_result is not None
    assert other.rag.index_result.n_chunks == session.rag.index_result.n_chunks
    after = other.rag.retrieve("dogs document", k=2)
    assert after.hits[0].doc_id == before.hits[0].doc_id


def test_rag_library_restore_preserves_retrieval(tmp_path: Path) -> None:
    from buildml.rag.corpus import corpus_from_documents

    corpus = corpus_from_documents(
        [
            {"doc_id": "cats", "text": "cats purr and sleep on warm windowsills"},
            {"doc_id": "dogs", "text": "dogs bark and chase balls in the park"},
            {"doc_id": "birds", "text": "birds sing and build nests in trees"},
        ]
    )
    index = build_index(
        corpus,
        embedder=HashingEmbedder(n_features=64),
        chunk_size=200,
        chunk_overlap=0,
    )
    hits = retrieve(index, "dogs bark park", k=2)
    path = save_rag_bundle(tmp_path / "rag_lib", index)
    restored = load_rag_bundle(path)
    assert len(restored.chunks) == len(index.chunks)
    again = retrieve(restored, "dogs bark park", k=2)
    assert again.hits[0].doc_id == hits.hits[0].doc_id
    assert np.allclose(restored.embeddings, index.embeddings)


def test_nlp_facade_bundle_restore_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    texts = [
        "card declined payment failed",
        "shipping delayed package late",
        "refund request for order",
        "login password reset help",
        "card charged twice billing",
        "delivery missing item",
        "payment error at checkout",
        "track shipment status",
    ] * 6
    labels = (["billing", "shipping", "billing", "account"] * 12)[: len(texts)]
    # Shuffle lightly while keeping determinism.
    order = rng.permutation(len(texts))
    frame = pd.DataFrame(
        {"text": [texts[i] for i in order], "label": [labels[i] for i in order]}
    )
    session = (
        Session.ingest(frame)
        .set_roles({"text": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    session.nlp.fit_classifier(text_column="text")
    preds_before = session.nlp.predict(partition="test")
    path = session.nlp.save_bundle(tmp_path / "nlp_facade")
    other = Session()
    other.nlp.load_bundle(path, trusted=True)
    assert other.nlp.text_plan is not None
    # Library-level reload also works.
    text_plan, _topic = load_nlp_bundle(path, trusted=True)
    assert text_plan is not None
    from buildml.nlp.predict import predict_documents

    sample = frame["text"].iloc[:3].tolist()
    labels, _proba = predict_documents(text_plan, sample)
    assert len(labels) == 3
    assert preds_before is not None


def test_nlp_library_save_load_roundtrip(tmp_path: Path) -> None:
    from buildml.nlp.fit import fit_text_classifier
    from buildml.nlp.predict import predict_documents

    frame = pd.DataFrame(
        {
            "text": [
                "great product loved it",
                "terrible experience refund",
                "amazing quality recommend",
                "awful support never again",
                "excellent value purchase",
                "poor packaging damaged",
            ]
            * 4,
            "y": [1, 0, 1, 0, 1, 0] * 4,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"text": "feature", "y": "target"})
        .split(test_size=0.3, stratify=True, random_state=0)
    )
    plan, _fit = fit_text_classifier(
        session.dataset, session.split_plan, text_column="text"
    )
    path = save_nlp_bundle(tmp_path / "nlp_lib", plan)
    text_plan, topic_plan = load_nlp_bundle(path, trusted=True)
    assert topic_plan is None
    assert text_plan is not None
    labels, _proba = predict_documents(text_plan, ["great product", "terrible refund"])
    assert len(labels) == 2
