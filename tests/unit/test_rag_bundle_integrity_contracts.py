"""Bundle restoration must preserve the actual retrieval vector space."""
import json

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.rag.checkpoint import load_rag_bundle, save_rag_bundle
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.embed import CallableEmbedder
from buildml.rag.index import build_index
from buildml.rag.retrieve import retrieve


@pytest.fixture
def bundle(tmp_path):
    corpus = corpus_from_documents([{"doc_id": "a", "text": "apple orchard"}, {"doc_id": "b", "text": "ocean boat"}])
    index = build_index(corpus, embedder="hashing", chunk_size=64, chunk_overlap=0)
    return save_rag_bundle(tmp_path / "bundle", index), index


@pytest.mark.parametrize("backend", ["callable", "unknown", "", None])
def test_unrecoverable_embedding_space_is_refused(bundle, backend):
    path, _ = bundle
    file = path / "meta.json"
    meta = json.loads(file.read_text())
    meta["embed_config"]["backend"] = backend
    file.write_text(json.dumps(meta))
    with pytest.raises(ValidationError, match="embedder|backend"):
        load_rag_bundle(path)


@pytest.mark.parametrize("damage", ["dim", "rows", "nan", "inf"])
def test_inconsistent_bundle_vectors_are_refused_on_load(bundle, damage):
    path, _ = bundle
    matrix = np.load(path / "embeddings.npy")
    if damage == "dim":
        matrix = matrix[:, :-1]
    elif damage == "rows":
        matrix = matrix[:-1]
    else:
        matrix[0, 0] = float(damage)
    np.save(path / "embeddings.npy", matrix)
    with pytest.raises(ValidationError):
        load_rag_bundle(path)


@pytest.mark.parametrize("filename", ["meta.json", "chunks.jsonl", "embeddings.npy"])
def test_partial_bundles_refused(bundle, filename):
    path, _ = bundle
    (path / filename).unlink()
    with pytest.raises(ValidationError):
        load_rag_bundle(path)


def test_hashing_roundtrip_preserves_rank_and_scores(bundle):
    path, index = bundle
    restored = load_rag_bundle(path)
    for query in ["apple", "boat", "unseen"]:
        before = retrieve(index, query, k=2).hits
        after = retrieve(restored, query, k=2).hits
        assert [h.chunk_id for h in before] == [h.chunk_id for h in after]
        np.testing.assert_allclose([h.score for h in before], [h.score for h in after], atol=1e-6)


def test_sentence_model_identity_is_required(bundle):
    path, _ = bundle
    file = path / "meta.json"
    meta = json.loads(file.read_text())
    meta["embed_config"].update(backend="sentence-transformers", model_name=None)
    file.write_text(json.dumps(meta))
    with pytest.raises(ValidationError, match="model name"):
        load_rag_bundle(path)


def test_real_callable_bundle_requires_original_embedding_space(tmp_path):
    def custom(texts):
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)
    corpus = corpus_from_documents(["apple orchard", "ocean boat"])
    index = build_index(corpus, embedder=CallableEmbedder(custom, dim=2), chunk_size=64, chunk_overlap=0)
    path = save_rag_bundle(tmp_path / "custom", index)
    with pytest.raises(ValidationError, match="original embedder"):
        load_rag_bundle(path)
