"""Passage limits and source provenance survive indexing, updates and bundles."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.rag.checkpoint import load_rag_bundle, save_rag_bundle
from buildml.rag.chunk import chunk_documents
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.embed import CallableEmbedder
from buildml.rag.generate import hits_to_citations
from buildml.rag.index import build_index
from buildml.rag.results import Document
from buildml.rag.retrieve import retrieve
from buildml.rag.types import ChunkConfig

TEXTS = [
    "",
    "abcdefghij" * 9,
    "first paragraph.\n\nsecond paragraph.\n\nthird paragraph.\n" * 3,
    "alpha beta " * 12,
    "cafe\u0301 \U0001f331\n\n\u6771\u4eac\u306e\u99c5\u3002 " * 9,
    "\n\n alpha  beta. . gamma\n\n\n delta\n " * 4,
]
SEPARATORS = [("\n\n", "\n", ". ", " ", ""), ("|", " "), ("",), ()]


def _assert_spans(chunks, documents, size, overlap):
    """Check against source characters, without a second chunking algorithm."""
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.doc_id for c in chunks} == {d.doc_id for d in documents}
    for doc in documents:
        selected = [c for c in chunks if c.doc_id == doc.doc_id]
        assert selected[0].start_char == 0
        assert selected[-1].end_char == len(doc.text)
        rebuilt = ""
        for ordinal, chunk in enumerate(selected):
            assert chunk.chunk_id == f"{doc.doc_id}::c{ordinal}"
            assert chunk.metadata == doc.metadata
            assert 0 <= chunk.start_char <= chunk.end_char <= len(doc.text)
            assert chunk.text == doc.text[chunk.start_char : chunk.end_char]
            assert len(chunk.text) <= size
            if ordinal:
                previous = selected[ordinal - 1]
                assert chunk.start_char == previous.end_char - overlap
                assert chunk.end_char > previous.end_char
                rebuilt += chunk.text[overlap:]
            else:
                rebuilt = chunk.text
        assert rebuilt == doc.text


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
@pytest.mark.parametrize("separators", SEPARATORS)
@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("size,overlap", [(1, 0), (20, 0), (20, 5), (20, 19)])
def test_passages_preserve_every_source_character(text, size, overlap, separators, strategy):
    documents = [Document(doc_id="source", text=text, metadata={"revision": 3})]
    config = ChunkConfig(size=size, overlap=overlap, strategy=strategy, separators=separators)
    original = config.to_dict()
    result = chunk_documents(documents, config=config)
    _assert_spans(result.chunks, documents, size, overlap)
    assert result.config == original == config.to_dict()
    assert result == chunk_documents(documents, config=config)
    if strategy == "fixed" and text:
        expected = []
        for start in range(0, len(text), size - overlap):
            expected.append((start, min(start + size, len(text))))
            if expected[-1][1] == len(text):
                break
        assert [(c.start_char, c.end_char) for c in result.chunks] == expected


def _documents():
    return [
        Document("refund", "refund window thirty days.\n\n" * 5, {"version": 1}),
        Document("garden", "orchard apples pears.\n\n" * 4, {"version": 1}),
    ]


def _assert_retrieval(index, documents):
    """Independently normalise lexical counts and score every indexed passage."""
    vectorizer = HashingVectorizer(
        n_features=index.embed_config.dim, alternate_sign=False, norm=None, lowercase=True
    )
    texts = [c.text for c in index.chunks]
    counts = vectorizer.transform(texts + ["refund window"]).toarray().astype(np.float64)
    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    vectors = counts / np.maximum(norms, 1e-12)
    np.testing.assert_allclose(index.embeddings, vectors[:-1], atol=2e-7)
    expected = vectors[:-1] @ vectors[-1]
    result = retrieve(index, "refund window", k=len(index.chunks), mode="dense")
    by_id = {c.chunk_id: (i, c) for i, c in enumerate(index.chunks)}
    assert len(result.hits) == len(by_id)
    assert set(h.chunk_id for h in result.hits) == set(by_id)
    assert result.hits[0].score == pytest.approx(float(expected.max()), abs=2e-7)
    for hit, citation in zip(result.hits, hits_to_citations(result.hits), strict=True):
        i, chunk = by_id[hit.chunk_id]
        assert hit.score == pytest.approx(float(expected[i]), abs=2e-7)
        assert hit.text == citation.text == chunk.text
        assert hit.doc_id == citation.doc_id == chunk.doc_id
        assert hit.metadata == citation.metadata == chunk.metadata
    _assert_spans(index.chunks, documents, index.chunk_config.size, index.chunk_config.overlap)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
@pytest.mark.parametrize("separators", [SEPARATORS[0], ("|", "")])
@pytest.mark.parametrize("override", [{}, {"chunk_size": 24}, {"chunk_overlap": 0}])
@pytest.mark.parametrize("update_chunk", [True, False])
def test_index_configuration_and_source_spans_survive_document_replacement(
    tmp_path, strategy, separators, override, update_chunk
):
    documents = _documents()
    config = ChunkConfig(size=20, overlap=5, strategy=strategy, separators=separators)
    before_config = deepcopy(config.to_dict())
    index = build_index(
        corpus_from_documents(documents), embedder="hashing", chunk_config=config, **override
    )
    assert index.chunk_config.strategy == strategy
    assert index.chunk_config.separators == separators
    assert index.chunk_config.size == override.get("chunk_size", 20)
    assert index.chunk_config.overlap == override.get("chunk_overlap", 5)
    assert config.to_dict() == before_config
    _assert_retrieval(index, documents)
    bundle = save_rag_bundle(tmp_path / "original", index)
    snapshot = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    restored = load_rag_bundle(bundle)
    assert restored.chunk_config.to_dict() == index.chunk_config.to_dict()
    assert restored.chunks == index.chunks
    _assert_retrieval(restored, documents)
    kept = [
        (c, row.copy())
        for c, row in zip(restored.chunks, restored.embeddings, strict=True)
        if c.doc_id == "garden"
    ]
    replacement = Document("refund", "refund changed.", {"version": 2})
    restored.upsert_documents([replacement], chunk=update_chunk)
    assert restored.n_documents == 2
    assert sum(c.doc_id == "refund" for c in restored.chunks) == 1
    _assert_retrieval(restored, [documents[1], replacement])
    for chunk, row in kept:
        position = list(restored.chunks).index(chunk)
        np.testing.assert_array_equal(restored.embeddings[position], row)
    final = load_rag_bundle(save_rag_bundle(tmp_path / "updated", restored))
    _assert_retrieval(final, [documents[1], replacement])
    if update_chunk:
        longer = Document("refund", "refund extension forty five days.\n\n" * 8, {"version": 3})
        final.upsert_documents([longer])
        _assert_retrieval(final, [documents[1], longer])
    # The original on-disk snapshot and live index still describe the original corpus.
    assert load_rag_bundle(bundle).chunks == index.chunks
    assert {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()} == snapshot


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
@pytest.mark.parametrize("pre_chunk", [False, True])
def test_session_retrieval_and_bundle_keep_passage_limits(tmp_path, strategy, pre_chunk):
    documents = _documents()
    session = Session().rag.ingest_corpus(documents)
    if pre_chunk:
        session.rag.chunk(size=20, overlap=5, strategy=strategy)
        session.rag.embed_and_index(embedder="hashing")
    else:
        session.rag.embed_and_index(
            embedder="hashing", chunk_size=20, chunk_overlap=5, chunk_strategy=strategy
        )
    assert session.rag_index_result.chunk_config["strategy"] == strategy
    path = session.rag.save_bundle(tmp_path / "session")
    index = load_rag_bundle(path)
    _assert_retrieval(index, documents)
    other = Session().rag.load_bundle(path)
    replacement = Document("refund", "refund changed.", {"version": 2})
    other.rag.upsert([replacement])
    updated = load_rag_bundle(other.rag.save_bundle(tmp_path / "session-updated"))
    _assert_retrieval(updated, [documents[1], replacement])
    hits = other.rag.retrieve("refund window", k=len(updated.chunks), mode="dense").hits
    assert {h.chunk_id for h in hits} == {c.chunk_id for c in updated.chunks}
    assert session.rag_index_result.n_chunks == len(index.chunks)
    assert other.rag_index_result.n_chunks == len(updated.chunks)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
@pytest.mark.parametrize("size,overlap", [(0, 0), (20, -1), (20, 20)])
def test_invalid_limits_refuse_before_changing_document_metadata(strategy, size, overlap):
    document = Document("source", "refund window", {"version": 1})
    before = deepcopy(document.to_dict())
    with pytest.raises(ValidationError, match="size|overlap"):
        chunk_documents([document], size=size, overlap=overlap, strategy=strategy)
    assert document.to_dict() == before


@pytest.mark.parametrize("override", [{"size": 20}, {"overlap": 0}, {"strategy": "recursive"}])
def test_individual_chunk_overrides_preserve_other_settings(override):
    document = Document("source", TEXTS[2])
    config = ChunkConfig(size=25, overlap=4, strategy="fixed", separators=("\n\n", ""))
    before = config.to_dict()
    result = chunk_documents([document], config=config, **override)
    expected = dict(before, **override)
    assert result.config == expected
    assert config.to_dict() == before
    _assert_spans(result.chunks, [document], expected["size"], expected["overlap"])


def test_eval_only_replacement_leaves_existing_index_usable():
    documents = _documents()
    index = build_index(
        corpus_from_documents(documents), embedder="hashing", chunk_size=20, chunk_overlap=5
    )
    before_chunks, before_vectors = index.chunks, index.embeddings.copy()
    with pytest.raises(LeakageError, match="eval_only"):
        index.upsert_documents([Document("refund", "hidden answer", role="eval_only")])
    assert index.chunks == before_chunks
    np.testing.assert_array_equal(index.embeddings, before_vectors)
    _assert_retrieval(index, documents)


def test_actual_callable_embedding_failure_preserves_entire_old_document():
    def embed(texts):
        # A user-provided embedding function with an inconsistent response width.
        width = 3 if any("reject" in text for text in texts) else 2
        return np.asarray([[len(text), 1.0, 2.0][:width] for text in texts], dtype=np.float32)

    documents = _documents()
    index = build_index(
        corpus_from_documents(documents),
        embedder=CallableEmbedder(embed, dim=2),
        chunk_config=ChunkConfig(size=20, overlap=5, strategy="recursive"),
    )
    chunks, vectors = index.chunks, index.embeddings.copy()
    before = retrieve(index, "refund", k=2, mode="dense")
    with pytest.raises(ValidationError, match="dim|shape"):
        index.upsert_documents([Document("refund", "reject this revision")])
    assert index.chunks == chunks
    np.testing.assert_array_equal(index.embeddings, vectors)
    assert retrieve(index, "refund", k=2, mode="dense") == before
