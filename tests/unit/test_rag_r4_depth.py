"""Unit coverage for RAG R4 industry depth (defaults, recursive chunk, gen eval)."""

from __future__ import annotations

import importlib.util

import pytest

from buildml import Session
from buildml.rag.chunk import chunk_documents
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.defaults import default_embedder_spec, default_retrieve_mode
from buildml.rag.embed import HashingEmbedder
from buildml.rag.evaluate import evaluate_generation, evaluate_retrieval
from buildml.rag.extras import rag_available
from buildml.rag.index import build_index
from buildml.rag.results import Document
from buildml.rag.retrieve import retrieve
from buildml.rag.types import RetrieveConfig

_RAG_SPEC = importlib.util.find_spec("sentence_transformers") is not None


def _rag_usable() -> bool:
    if not _RAG_SPEC:
        return False
    try:
        return rag_available()
    except Exception:
        return False


def _toy_corpus():
    return corpus_from_documents(
        [
            Document(
                doc_id="para",
                text="First paragraph about cats.\n\nSecond paragraph about dogs.\n\nThird about birds.",
            ),
            Document(doc_id="rust", text="Rust systems programming language."),
        ]
    )


def test_default_embedder_spec_is_auto() -> None:
    assert default_embedder_spec() == "auto"


def test_default_retrieve_mode_tracks_rag_extra() -> None:
    expected = "hybrid" if _rag_usable() else "dense"
    assert default_retrieve_mode() == expected


def test_recursive_chunking_produces_stable_ids() -> None:
    corpus = _toy_corpus()
    fixed = chunk_documents(corpus, strategy="fixed", size=40, overlap=8)
    recursive = chunk_documents(corpus, strategy="recursive", size=40, overlap=8)
    assert fixed.n_chunks >= 1
    assert recursive.n_chunks >= 1
    assert fixed.config["strategy"] == "fixed"
    assert recursive.config["strategy"] == "recursive"
    again = chunk_documents(corpus, strategy="recursive", size=40, overlap=8)
    assert [c.chunk_id for c in recursive.chunks] == [c.chunk_id for c in again.chunks]


def test_auto_embedder_hashing_when_rag_missing() -> None:
    if _rag_usable():
        pytest.skip("rag installed; test explicit hashing path instead")
    corpus = _toy_corpus()
    index = build_index(corpus, embedder="auto", chunk_size=120, chunk_overlap=0)
    assert index.embed_config.backend == "hashing"


@pytest.mark.skipif(not _rag_usable(), reason="buildml[rag] unavailable")
def test_auto_embedder_sentence_transformers_when_rag_present() -> None:
    corpus = _toy_corpus()
    try:
        index = build_index(corpus, embedder="auto", chunk_size=120, chunk_overlap=0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"sentence-transformers runtime unavailable: {exc}")
    assert index.embed_config.backend == "sentence-transformers"
    assert "sentence-transformers" in index.embed_config.embedder_id


def test_hybrid_default_when_rag_present_or_dense_otherwise() -> None:
    corpus = _toy_corpus()
    index = build_index(
        corpus,
        embedder=HashingEmbedder(n_features=128),
        chunk_size=200,
        chunk_overlap=0,
    )
    result = retrieve(index, "systems programming rust", k=2)
    expected_mode = "hybrid" if _rag_usable() else "dense"
    assert result.mode == expected_mode


def test_evaluate_generation_echo_provider() -> None:
    corpus = corpus_from_documents(
        [Document(doc_id="d", text="BuildML retrieval uses citations and grounded context.")]
    )
    index = build_index(corpus, embedder=HashingEmbedder(n_features=64), chunk_size=200, chunk_overlap=0)
    gen_eval = evaluate_generation(
        index,
        [{"query": "retrieval citations", "reference_answer": "citations grounded context"}],
        k=2,
        retrieve_config=RetrieveConfig(k=2, mode="dense"),
    )
    assert gen_eval.n_queries == 1
    assert 0.0 <= gen_eval.mean_faithfulness <= 1.0
    assert 0.0 <= gen_eval.mean_answer_relevance <= 1.0


def test_session_explicit_hashing_and_hybrid_override() -> None:
    session = Session()
    session.rag_ingest_corpus(
        [
            {"doc_id": "py", "text": "python data science language"},
            {"doc_id": "rs", "text": "rust systems programming language"},
        ]
    )
    session.rag_chunk(size=80, overlap=8, strategy="recursive")
    session.rag_embed_and_index(embedder="hashing", chunk_size=80, chunk_overlap=8)
    assert session.rag_index_result.embedder_id == "buildml.hashing_embed.v1"
    dense = session.rag_retrieve("systems rust", k=2, mode="dense")
    hybrid = session.rag_retrieve("systems rust", k=2, mode="hybrid")
    assert dense.mode == "dense"
    assert hybrid.mode == "hybrid"

    metrics = session.rag_evaluate({"systems rust": ["rs"]}, k=2, mode="hybrid")
    assert metrics.retrieve_mode == "hybrid"
