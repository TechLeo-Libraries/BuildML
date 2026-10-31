"""Sparse-index reuse must preserve correctness after corpus mutation and filtering."""
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.embed import HashingEmbedder
from buildml.rag.hybrid import BM25Index
from buildml.rag.index import build_index
from buildml.rag.results import Document
from buildml.rag.retrieve import retrieve
from buildml.rag.types import RetrieveConfig


def test_sparse_query_of_index_persisted_before_cache_field():
    index = build_index(corpus_from_documents([
        Document(doc_id="a", text="apple orchard", metadata={}),
    ]), embedder=HashingEmbedder(n_features=32))
    del index._bm25_cache
    assert retrieve(index, "apple", config=RetrieveConfig(mode="bm25", k=1)).hits[0].doc_id == "a"


def test_sparse_cache_reuses_and_invalidates_on_updates(monkeypatch):
    index = build_index(corpus_from_documents([
        Document(doc_id="a", text="apple orchard", metadata={}),
        Document(doc_id="b", text="orange juice", metadata={}),
    ]), embedder=HashingEmbedder(n_features=32))
    builds = []
    original = BM25Index.build

    def build(chunks, **kwargs):
        builds.append(len(chunks))
        return original(chunks, **kwargs)

    monkeypatch.setattr(BM25Index, "build", staticmethod(build))
    cfg = RetrieveConfig(mode="bm25", k=1)
    assert retrieve(index, "apple", config=cfg).hits[0].doc_id == "a"
    retrieve(index, "orange", config=cfg)
    assert len(builds) == 1
    index.upsert_documents([Document(doc_id="c", text="apple apple apple", metadata={})])
    assert index._bm25_cache is None
    assert retrieve(index, "apple", config=cfg).hits[0].doc_id == "c"
    assert len(builds) == 2
    index.delete(doc_ids=["c"])
    assert index._bm25_cache is None
    assert retrieve(index, "apple", config=cfg).hits[0].doc_id == "a"
    assert len(builds) == 3
