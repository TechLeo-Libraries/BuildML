"""Unit coverage for the RAG thin slice (skip-friendly)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.rag.chunk import chunk_documents
from buildml.rag.corpus import corpus_from_documents, corpus_from_frame, refuse_eval_only_index
from buildml.rag.embed import HashingEmbedder
from buildml.rag.index import build_index
from buildml.rag.results import Document

_RAG_SPEC = importlib.util.find_spec("sentence_transformers") is not None


def _rag_usable() -> bool:
    if not _RAG_SPEC:
        return False
    try:
        from buildml.rag.extras import rag_available

        return rag_available()
    except Exception:
        return False


def test_core_import_does_not_require_rag() -> None:
    import buildml
    import buildml.rag as rag

    assert hasattr(buildml, "Session")
    assert hasattr(rag, "rag_available")


def test_missing_rag_extra_message_for_sentence_transformers() -> None:
    if _rag_usable():
        pytest.skip("rag extras installed and importable in this environment")
    if _RAG_SPEC and not _rag_usable():
        pytest.skip("sentence-transformers present but not importable")
    session = Session().rag_ingest_corpus(["alpha beta gamma for retrieval"])
    with pytest.raises(MissingExtraError, match="buildml\\[rag\\]"):
        session.rag_embed_and_index(embedder="sentence-transformers")


def test_catalog_covers_rag_operations() -> None:
    for name in (
        "rag_ingest_corpus",
        "rag_chunk",
        "rag_embed_and_index",
        "rag_retrieve",
        "rag_generate",
        "rag_evaluate",
        "save_rag_bundle",
        "load_rag_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "rag-eval-contamination" in OPERATION_CATALOG["rag_embed_and_index"].concept_links
    assert "rag-retrieval-metrics" in OPERATION_CATALOG["rag_evaluate"].concept_links
    assert "rag-chunk-index-boundary" in OPERATION_CATALOG["save_rag_bundle"].concept_links


def test_chunk_ids_deterministic() -> None:
    corpus = corpus_from_documents(
        [
            Document(doc_id="a", text="hello world " * 40),
            Document(doc_id="b", text="other text " * 40),
        ]
    )
    first = chunk_documents(corpus, size=64, overlap=16)
    second = chunk_documents(corpus, size=64, overlap=16)
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert first.chunks[0].chunk_id.startswith("a::c")


def test_refuse_eval_only_index() -> None:
    corpus = corpus_from_documents(
        [
            Document(doc_id="good", text="index me", role="index"),
            Document(doc_id="secret", text="the answer is 42", role="eval_only"),
        ]
    )
    with pytest.raises(LeakageError, match="eval_only"):
        refuse_eval_only_index(corpus)
    with pytest.raises(LeakageError, match="eval_only"):
        build_index(corpus, embedder=HashingEmbedder(n_features=64))


def test_hashing_embed_index_retrieve_eval_and_bundle(tmp_path: Path) -> None:
    corpus = corpus_from_documents(
        [
            {"doc_id": "cats", "text": "cats purr and chase mice in the garden"},
            {"doc_id": "dogs", "text": "dogs bark and chase balls in the park"},
            {"doc_id": "birds", "text": "birds sing and build nests in trees"},
        ]
    )
    index = build_index(
        corpus,
        embedder=HashingEmbedder(n_features=128),
        chunk_size=200,
        chunk_overlap=0,
    )
    assert index.to_index_result().n_chunks >= 3
    assert index.embed_config.embedder_id == "buildml.hashing_embed.v1"

    from buildml.rag.checkpoint import BUNDLE_FORMAT, load_rag_bundle, save_rag_bundle
    from buildml.rag.evaluate import evaluate_retrieval
    from buildml.rag.retrieve import retrieve

    hits = retrieve(index, "dogs bark park", k=2)
    assert len(hits.hits) == 2
    assert hits.hits[0].doc_id in {"dogs", "cats", "birds"}

    eval_result = evaluate_retrieval(
        index,
        {"dogs bark and chase": ["dogs"], "birds sing nests": ["birds"]},
        k=2,
    )
    assert eval_result.n_queries == 2
    assert 0.0 <= eval_result.recall_at_k <= 1.0
    assert 0.0 <= eval_result.mrr <= 1.0
    assert eval_result.relevance_mode == "document"

    path = save_rag_bundle(tmp_path / "rag_bundle", index, eval_result=eval_result)
    assert (path / "meta.json").is_file()
    restored = load_rag_bundle(path)
    assert restored.embed_config.dim == index.embed_config.dim
    assert len(restored.chunks) == len(index.chunks)
    again = retrieve(restored, "dogs bark park", k=2)
    assert again.hits[0].doc_id == hits.hits[0].doc_id

    with pytest.raises(ValidationError, match=BUNDLE_FORMAT):
        bad = tmp_path / "bad_bundle"
        bad.mkdir()
        (bad / "meta.json").write_text('{"format": "buildml.torch_bundle.v1"}', encoding="utf-8")
        (bad / "chunks.jsonl").write_text("\n", encoding="utf-8")
        np.save(bad / "embeddings.npy", np.zeros((1, 8), dtype=np.float32))
        load_rag_bundle(bad)


def test_corpus_from_frame_requires_text_column() -> None:
    frame = pd.DataFrame({"body": ["alpha", "beta"], "id": ["1", "2"]})
    corpus = corpus_from_frame(frame, text_column="body", id_column="id")
    assert corpus.n_documents == 2
    assert corpus.documents[0].doc_id == "1"
    with pytest.raises(ValidationError, match="text_column"):
        corpus_from_frame(frame, text_column="missing")


def test_session_rag_vertical_slice(tmp_path: Path) -> None:
    session = Session()
    session.rag_ingest_corpus(
        [
            {"doc_id": "py", "text": "python is a programming language for data work"},
            {"doc_id": "rs", "text": "rust is a systems programming language"},
            {"doc_id": "go", "text": "go is a programming language for concurrent services"},
        ]
    )
    session.rag_chunk(size=128, overlap=16)
    session.rag_embed_and_index(embedder="hashing")
    assert session.rag_index_result is not None
    result = session.rag_retrieve("systems programming rust", k=2)
    assert len(result.hits) == 2
    metrics = session.rag_evaluate({"systems programming rust": ["rs"]}, k=2)
    assert metrics.recall_at_k >= 0.0
    path = session.save_rag_bundle(tmp_path / "bundle")
    other = Session()
    other.load_rag_bundle(path)
    assert other.rag_index_result is not None
    assert other.rag_index_result.n_chunks == session.rag_index_result.n_chunks
    before = session.explain("rag_retrieve", moment="before")
    assert before.operation == "rag_retrieve"
