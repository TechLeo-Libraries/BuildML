"""Real local retrieval evaluation across file inputs and supported label forms."""
import math

import pytest

from buildml.core.errors import ValidationError
from buildml.rag.corpus import load_text_corpus
from buildml.rag.evaluate import evaluate_retrieval
from buildml.rag.index import build_index
from buildml.rag.retrieve import retrieve


@pytest.fixture
def index(tmp_path):
    (tmp_path / "alpha.txt").write_text("alpha unique apple orchard harvest", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta unique ocean sailing boat", encoding="utf-8")
    (tmp_path / "ignored.txt").mkdir()
    corpus = load_text_corpus(tmp_path)
    assert [d.doc_id for d in corpus.documents] == ["alpha", "beta"]
    return build_index(corpus, embedder="hashing", chunk_size=128, chunk_overlap=0)


@pytest.mark.parametrize("mode", ["document", "chunk"])
@pytest.mark.parametrize("shape", ["mapping", "plural", "alias", "single"])
def test_label_forms_match_independent_rank_metrics(index, mode, shape):
    query = "alpha apple orchard"
    hits = retrieve(index, query, k=2).hits
    ids = [h.doc_id if mode == "document" else h.chunk_id for h in hits]
    relevant = ids[-1]
    if shape == "mapping":
        labels = {query: [relevant]}
    else:
        keys = ({"plural": "relevant_doc_ids", "alias": "relevant_docs", "single": "doc_id"} if mode == "document" else {"plural": "relevant_chunk_ids", "alias": "relevant_chunks", "single": "chunk_id"})
        labels = [{"query": query, keys[shape]: relevant if shape == "single" else [relevant]}]
    result = evaluate_retrieval(index, labels, k=2, relevance_mode=mode, mode="dense")
    assert result.recall_at_k == 1.0
    assert result.mrr == pytest.approx(1 / 2)
    assert result.ndcg_at_k == pytest.approx(1 / math.log2(3))


@pytest.mark.parametrize("labels", [{}, [], {"q": []}, {"q": "alpha"}, {"q": None}, [{"query": "q", "relevant_doc_ids": "alpha"}], [{"query": "q", "relevant_doc_ids": None}], [{}], [4], [{"query": "q"}]])
def test_malformed_labels_raise_domain_error(index, labels):
    with pytest.raises(ValidationError):
        evaluate_retrieval(index, labels)


def test_file_input_decode_and_missing_errors(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        load_text_corpus(tmp_path / "missing")
    with pytest.raises(ValidationError, match="No files"):
        load_text_corpus(tmp_path)
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(ValidationError, match="decode"):
        load_text_corpus(path)


def test_single_file_keeps_provenance(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("alpha", encoding="utf-8")
    result = load_text_corpus(path)
    assert result.documents[0].text == "alpha"
    assert result.documents[0].metadata["path"] == str(path)
