"""Unit coverage for RAG M2 depth (hybrid, rerank gate, upsert, eval, status)."""

from __future__ import annotations

import importlib.util

import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.embed import HashingEmbedder
from buildml.rag.evaluate import compare_retrieval_configs, evaluate_retrieval
from buildml.rag.explain_hooks import rag_status_for_session
from buildml.rag.hybrid import BM25Index, rrf_fuse, weighted_fuse
from buildml.rag.index import build_index
from buildml.rag.results import Chunk, Document, Hit
from buildml.rag.retrieve import retrieve
from buildml.rag.types import RetrieveConfig

_RAG_SPEC = importlib.util.find_spec("sentence_transformers") is not None


def _rag_usable() -> bool:
    if not _RAG_SPEC:
        return False
    try:
        from buildml.rag.extras import rag_available

        return rag_available()
    except Exception:
        return False


def _toy_index():
    corpus = corpus_from_documents(
        [
            Document(
                doc_id="cats",
                text="cats purr and chase mice in the garden",
                metadata={"topic": "feline"},
            ),
            Document(
                doc_id="dogs",
                text="dogs bark and chase balls in the park",
                metadata={"topic": "canine"},
            ),
            Document(
                doc_id="birds",
                text="birds sing and build nests in trees",
                metadata={"topic": "avian"},
            ),
        ]
    )
    return build_index(
        corpus,
        embedder=HashingEmbedder(n_features=128),
        chunk_size=200,
        chunk_overlap=0,
    )


def test_catalog_covers_m2_ops() -> None:
    for name in ("rag_upsert", "rag_delete", "rag_retrieve", "rag_evaluate"):
        assert name in OPERATION_CATALOG
    retrieve_params = {p.name for p in OPERATION_CATALOG["rag_retrieve"].parameters}
    assert "mode" in retrieve_params
    assert "filters" in retrieve_params
    assert "rerank" in retrieve_params


def test_bm25_and_hybrid_retrieve() -> None:
    index = _toy_index()
    dense = retrieve(index, "dogs bark park", k=2, mode="dense")
    sparse = retrieve(index, "dogs bark park", k=2, mode="bm25")
    hybrid = retrieve(index, "dogs bark park", k=2, mode="hybrid")
    assert dense.mode == "dense"
    assert sparse.mode == "bm25"
    assert hybrid.mode == "hybrid"
    assert hybrid.fusion == "rrf"
    assert len(sparse.hits) == 2
    assert sparse.hits[0].doc_id == "dogs"

    weighted = retrieve(
        index,
        "dogs bark park",
        k=2,
        mode="hybrid",
        fusion="weighted",
        config=RetrieveConfig(k=2, mode="hybrid", fusion="weighted", dense_weight=0.7),
    )
    assert weighted.fusion == "weighted"
    assert any("dense_weight=" in d for d in weighted.disclosures)


def test_rrf_and_weighted_fusion_helpers() -> None:
    a = [
        Hit("c1", "d1", 0.9, "t1", 1),
        Hit("c2", "d2", 0.5, "t2", 2),
    ]
    b = [
        Hit("c2", "d2", 3.0, "t2", 1),
        Hit("c3", "d3", 1.0, "t3", 2),
    ]
    fused = rrf_fuse([a, b], k=2, rrf_k=60)
    assert len(fused) == 2
    assert fused[0].rank == 1
    blended = weighted_fuse(a, b, k=2, dense_weight=0.5)
    assert len(blended) == 2


def test_metadata_filters() -> None:
    index = _toy_index()
    hits = retrieve(index, "chase", k=5, filters={"topic": "canine"})
    assert hits.hits
    assert all(h.metadata.get("topic") == "canine" for h in hits.hits)
    empty = retrieve(index, "chase", k=5, filters={"topic": "missing"})
    assert empty.hits == ()


def test_upsert_and_delete() -> None:
    index = _toy_index()
    before = len(index.chunks)
    index.upsert_documents(
        [
            {
                "doc_id": "fish",
                "text": "fish swim in rivers and lakes",
                "metadata": {"topic": "aquatic"},
            }
        ]
    )
    assert len(index.chunks) == before + 1
    assert any(c.doc_id == "fish" for c in index.chunks)

    # Replace existing chunk id path
    old_id = index.chunks[0].chunk_id
    index.upsert_chunks(
        [
            Chunk(
                chunk_id=old_id,
                doc_id=index.chunks[0].doc_id,
                text="updated text about cats in the garden",
                start_char=0,
                end_char=10,
                metadata={"topic": "feline", "rev": 2},
            )
        ]
    )
    replaced = next(c for c in index.chunks if c.chunk_id == old_id)
    assert "updated text" in replaced.text

    index.delete(doc_ids=["fish"])
    assert all(c.doc_id != "fish" for c in index.chunks)
    with pytest.raises(ValidationError, match="chunk_ids and/or doc_ids"):
        index.delete()


def test_eval_chunk_mode_ndcg_and_compare() -> None:
    index = _toy_index()
    doc_eval = evaluate_retrieval(
        index,
        {"dogs bark and chase": ["dogs"], "birds sing nests": ["birds"]},
        k=2,
        relevance_mode="document",
    )
    assert 0.0 <= doc_eval.ndcg_at_k <= 1.0
    assert 0.0 <= doc_eval.hit_rate_at_k <= 1.0
    assert doc_eval.relevance_mode == "document"

    chunk_id = next(c.chunk_id for c in index.chunks if c.doc_id == "dogs")
    chunk_eval = evaluate_retrieval(
        index,
        [{"query": "dogs bark park", "relevant_chunk_ids": [chunk_id]}],
        k=2,
        relevance_mode="chunk",
    )
    assert chunk_eval.relevance_mode == "chunk"
    assert chunk_eval.n_queries == 1

    corpus = corpus_from_documents(
        [
            Document(doc_id="cats", text="cats purr and chase mice in the garden"),
            Document(doc_id="dogs", text="dogs bark and chase balls in the park"),
        ]
    )
    compare = compare_retrieval_configs(
        corpus,
        [
            {"name": "dense", "chunk_size": 200, "embedder": HashingEmbedder(n_features=64)},
            {
                "name": "hybrid",
                "chunk_size": 200,
                "embedder": HashingEmbedder(n_features=64),
                "retrieve": {"mode": "hybrid", "fusion": "rrf"},
            },
        ],
        {"dogs bark park": ["dogs"]},
        k=2,
    )
    assert len(compare.rows) == 2
    assert {row["name"] for row in compare.rows} == {"dense", "hybrid"}
    assert "recall_at_k" in compare.rows[0]


def test_rerank_missing_extra_raises() -> None:
    if _rag_usable():
        pytest.skip("rag extras installed; cannot assert MissingExtraError path")
    if _RAG_SPEC and not _rag_usable():
        pytest.skip("sentence-transformers present but not importable")
    index = _toy_index()
    with pytest.raises(MissingExtraError, match="buildml\\[rag\\]"):
        retrieve(index, "dogs bark park", k=2, rerank=True)


@pytest.mark.skipif(not _rag_usable(), reason="buildml[rag] / sentence-transformers unavailable")
def test_cross_encoder_rerank_optional() -> None:
    index = _toy_index()
    # Use a tiny model if available; skip on download/runtime failure.
    try:
        result = retrieve(
            index,
            "dogs bark park",
            k=2,
            mode="hybrid",
            rerank="cross-encoder",
            config=RetrieveConfig(k=2, mode="hybrid", rerank=True, rerank_candidates=4),
        )
    except Exception as exc:  # noqa: BLE001 — optional model download / runtime
        pytest.skip(f"cross-encoder unavailable at runtime: {exc}")
    assert result.rerank is True
    assert len(result.hits) == 2
    assert any("rerank=cross-encoder" in d for d in result.disclosures)


def test_session_upsert_delete_hybrid_and_walkthrough_status() -> None:
    session = Session()
    session.rag_ingest_corpus(
        [
            {
                "doc_id": "py",
                "text": "python is a programming language for data work",
                "metadata": {"lang": "python"},
            },
            {
                "doc_id": "rs",
                "text": "rust is a systems programming language",
                "metadata": {"lang": "rust"},
            },
            {
                "doc_id": "go",
                "text": "go is a programming language for concurrent services",
                "metadata": {"lang": "go"},
            },
        ]
    )
    session.rag_embed_and_index(chunk_size=128, chunk_overlap=0)
    hybrid = session.rag_retrieve("systems programming rust", k=2, mode="hybrid")
    assert hybrid.mode == "hybrid"
    filtered = session.rag_retrieve(
        "programming language",
        k=3,
        mode="bm25",
        filters={"lang": "rust"},
    )
    assert filtered.hits
    assert all(h.metadata.get("lang") == "rust" for h in filtered.hits)

    session.rag_upsert(
        [
            {
                "doc_id": "js",
                "text": "javascript runs in browsers and servers",
                "metadata": {"lang": "js"},
            }
        ]
    )
    assert any(c.doc_id == "js" for c in session._rag_index.chunks)
    session.rag_delete(doc_ids=["js"])
    assert all(c.doc_id != "js" for c in session._rag_index.chunks)

    metrics = session.rag_evaluate(
        {"systems programming rust": ["rs"]},
        k=2,
        mode="hybrid",
    )
    assert metrics.ndcg_at_k >= 0.0
    assert metrics.retrieve_mode == "hybrid"

    status = rag_status_for_session(session)
    assert status["enabled"] is True
    assert status["index"]["n_chunks"] >= 1
    assert status["eval"] is not None
    assert any("embedder_id=" in d for d in status["disclosures"])
    assert any("Session checkpoints do not contain" in d for d in status["disclosures"])

    report = session.walkthrough()
    payload = report.to_dict()
    assert payload["rag_status"]["enabled"] is True
    assert payload["rag_status"]["index"]["embedder_id"]


def test_bm25_index_empty_safe() -> None:
    empty = BM25Index.build([])
    assert empty.query("anything", k=3) == []
