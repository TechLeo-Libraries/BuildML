"""Local RAG consumers keep passages live and discard stale citations."""

import json

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.rag.generate import EchoGroundedProvider

PASSAGE = "SYNTHETIC_PASSAGE_40792"
ANSWER = "SYNTHETIC_ANSWER_73109"
QUERY = "ceramic calibration reference"


def _index(session):
    documents = [
        {"doc_id": "calibration", "text": f"Ceramic calibration reference {PASSAGE}."},
        {"doc_id": "inventory", "text": "Crates and ladders belong in the inventory."},
    ]
    return session.rag.ingest_corpus(documents).rag.embed_and_index(embedder="hashing")


def _run(session, mode):
    retrieved = session.rag.retrieve(QUERY, k=1, mode=mode)
    evaluated = session.rag.evaluate({QUERY: ["calibration"]}, k=1, mode=mode)
    generated = session.rag.generate(
        QUERY, provider=EchoGroundedProvider(prefix=ANSWER), use_last_retrieve=True
    )
    assert [hit.doc_id for hit in retrieved.hits] == ["calibration"]
    assert [citation.doc_id for citation in generated.citations] == ["calibration"]
    assert (evaluated.recall_at_k, evaluated.mrr, evaluated.ndcg_at_k) == (1.0, 1.0, 1.0)
    return retrieved, evaluated, generated


@pytest.mark.parametrize("mode", ("bm25", "dense", "hybrid"))
def test_history_and_checkpoint_keep_metrics_and_ids_without_raw_text(tmp_path, mode):
    session = _index(Session.ingest(pd.DataFrame({"unrelated": [1, 2]})))
    retrieved, evaluated, generated = _run(session, mode)
    assert PASSAGE in retrieved.hits[0].text and ANSWER in generated.answer
    assert PASSAGE in json.dumps(generated.to_dict()) and evaluated.per_query

    def check(history):
        records = {item["operation_id"]: item["result_summary"] for item in history}
        retrieve = records["rag_retrieve"]
        assert retrieve["top_doc_ids"] == ["calibration"] and retrieve["n_hits"] == 1
        assert retrieve["mode"] == retrieved.mode
        evaluation = records["rag_evaluate"]
        assert evaluation["recall_at_k"] == evaluation["mrr"] == 1.0
        assert evaluation["retrieve_mode"] == evaluated.retrieve_mode
        assert "per_query" not in evaluation
        generation = records["rag_generate"]
        assert generation["citation_doc_ids"] == ["calibration"]
        assert generation["answer_chars"] == len(generated.answer)
        assert generation["provider_model"] == generated.provider_model
        snapshot = json.dumps(history)
        assert PASSAGE not in snapshot and ANSWER not in snapshot

    check(session.history)
    directory = session.data.checkpoint_save(tmp_path / "checkpoint")
    check(json.loads((directory / "history.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("mutation", ("delete_doc", "delete_chunk", "upsert", "reindex", "load"))
def test_generation_needs_a_new_retrieval_after_index_changes(tmp_path, mutation):
    session = _index(Session())
    retrieved, _, _ = _run(session, "bm25")
    if mutation == "delete_doc":
        session.rag.delete(doc_ids=["calibration"])
    elif mutation == "delete_chunk":
        session.rag.delete(chunk_ids=[retrieved.hits[0].chunk_id])
    elif mutation == "upsert":
        session.rag.upsert(
            [{"doc_id": "calibration", "text": "Revised ceramic calibration reference."}]
        )
    elif mutation == "reindex":
        session.rag.embed_and_index(embedder="hashing")
    else:
        replacement = (
            Session()
            .rag.ingest_corpus(
                [
                    {"doc_id": "replacement", "text": "Replacement ceramic calibration reference."},
                ]
            )
            .rag.embed_and_index(embedder="hashing")
        )
        bundle = replacement.rag.save_bundle(tmp_path / "replacement")
        session.rag.load_bundle(bundle)

    assert session.rag_retrieve_result is None
    assert session.rag_eval_result is None
    assert session.rag_generate_result is None
    with pytest.raises(ValidationError, match="requires a prior rag_retrieve"):
        session.rag.generate(QUERY, provider=EchoGroundedProvider(), use_last_retrieve=True)

    expected = {"delete_doc": "inventory", "delete_chunk": "inventory", "load": "replacement"}.get(
        mutation, "calibration"
    )
    fresh = session.rag.retrieve(QUERY, k=1, mode="dense")
    result = session.rag.generate(QUERY, provider=EchoGroundedProvider(), use_last_retrieve=True)
    assert [hit.doc_id for hit in fresh.hits] == [expected]
    assert [citation.doc_id for citation in result.citations] == [expected]
    if mutation != "reindex":
        assert PASSAGE not in json.dumps(result.to_dict())


def test_deleting_unknown_documents_preserves_usable_retrieval():
    session = _index(Session())
    retrieved, evaluated, generated = _run(session, "bm25")
    session.rag.delete(doc_ids=["unknown-document"])
    assert session.rag_retrieve_result is retrieved
    assert session.rag_eval_result is evaluated
    assert session.rag_generate_result is generated
