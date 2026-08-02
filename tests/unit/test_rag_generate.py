"""Unit coverage for grounded RAG generate (offline / mock provider)."""

from __future__ import annotations

import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.rag.generate import EchoGroundedProvider, assemble_grounded_messages, hits_to_citations
from buildml.rag.results import Hit


def _indexed_session() -> Session:
    docs = [
        "BuildML RAG retrieves chunks then generates grounded answers with citations.",
        "Classical Session.fit trains sklearn estimators on tabular data.",
        "Torch fit_torch trains neural networks with optional early stopping.",
    ]
    return Session().rag_ingest_corpus(docs).rag_embed_and_index()


def test_catalog_covers_rag_generate() -> None:
    assert "rag_generate" in OPERATION_CATALOG
    assert OPERATION_CATALOG["rag_generate"].name == "rag_generate"


def test_hits_to_citations_and_prompt_assembly() -> None:
    hits = (
        Hit(chunk_id="a::c0", doc_id="a", score=0.9, text="alpha evidence", rank=1),
        Hit(chunk_id="b::c0", doc_id="b", score=0.7, text="beta evidence", rank=2),
    )
    citations = hits_to_citations(hits)
    assert citations[0].source_id == 1
    assert citations[0].doc_id == "a"
    messages, context = assemble_grounded_messages("What is RAG?", citations)
    assert "[source:1]" in context
    assert "alpha evidence" in context
    assert messages[0].role == "system"
    assert "What is RAG?" in messages[1].content


def test_rag_generate_missing_index() -> None:
    session = Session()
    with pytest.raises(ValidationError, match="No RAG index"):
        session.rag_generate("hello", provider=EchoGroundedProvider())


def test_rag_generate_missing_provider() -> None:
    session = _indexed_session()
    with pytest.raises(ValidationError, match="provider"):
        session.rag_generate("What does RAG do?")


def test_rag_generate_grounded_with_echo_provider() -> None:
    session = _indexed_session()
    result = session.rag_generate(
        "How does BuildML RAG generate answers?",
        provider=EchoGroundedProvider(),
        k=3,
    )
    assert result.n_citations >= 1
    assert "[source:" in result.answer
    assert session.rag_generate_result is result
    assert session.rag_retrieve_result is not None
    assert any(c.doc_id for c in result.citations)


def test_rag_generate_empty_retrieval_fails() -> None:
    from buildml.rag.generate import generate_from_retrieve
    from buildml.rag.results import RetrieveResult

    empty = RetrieveResult(
        query="q",
        k=5,
        hits=(),
        embedder_id="test",
    )
    with pytest.raises(ValidationError, match="zero hits|Cannot generate"):
        generate_from_retrieve(empty, EchoGroundedProvider())


def test_rag_generate_reuses_session_ai_provider() -> None:
    session = _indexed_session().ai_configure(provider="mock")
    result = session.rag_generate("What is classical fit?", k=2)
    assert result.answer
    assert result.n_citations >= 1


def test_rag_generate_use_last_retrieve() -> None:
    session = _indexed_session()
    retrieved = session.rag_retrieve("Torch neural networks", k=2)
    result = session.rag_generate(
        "Torch neural networks",
        provider=EchoGroundedProvider(),
        use_last_retrieve=True,
    )
    assert result.retrieve_result is not None
    assert result.retrieve_result.query == retrieved.query
    assert result.n_citations == len(retrieved.hits)
