"""NLP resource controls retain discrete semantics and train-only probabilities."""

import copy
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from sklearn.utils import murmurhash3_32

from buildml import Session
from buildml.core.errors import ValidationError

BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")
BAD_NGRAMS = ((1.7, 2.4), (True, 2), ("1", "2"), (np.nan, 2), (1,), (1, 2, 3))
CLASSIFIER_CASES = [("ngram_range", value) for value in BAD_NGRAMS] + [
    ("n_hash_features", 32.5), ("n_hash_features", "32"),
    ("min_token_length", 1.5), ("min_token_length", True), ("min_token_length", np.nan),
    ("ngram_range", (2, 1)), ("max_features", 0), ("n_hash_features", 15),
    ("min_df", -0.1), ("max_df", 1.1),
]
TOPIC_CASES = [("ngram_range", value) for value in BAD_NGRAMS] + [
    ("n_topics", 2.4), ("n_topics", "2"), ("n_topics", np.nan), ("n_topics", np.inf),
    ("top_terms", 2.4), ("max_iter", 2.4), ("n_topics", True),
]


def _frame():
    documents = ["red red fox", "blue blue whale", "red fox jumps", "blue whale swims"] * 2
    documents += ["red future fox", "blue hidden whale", "red red unknown", "blue blue unseen"] * 2
    return pd.DataFrame({"text": documents, "label": ["a", "b", "a", "b"] * 4,
        "row": [f"record-{i}" for i in range(16)]})


def _session(backing, directory):
    frame = _frame()
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "documents.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="memory" if engine == "pandas" else "lazy")
    return session.set_roles({"text": "feature", "label": "target", "row": "id"}).inject_split(
        train_indices=list(range(8)), validation_indices=list(range(8, 12)), test_indices=list(range(12, 16)))


def _fit(session, **kwargs):
    options = dict(text_column="text", estimator="multinomial_nb", vectorizer="count",
        normalize_steps=["lowercase"], min_df=1, max_df=1.0)
    options.update(kwargs)
    return session.nlp.fit_classifier(**options)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("name,value", CLASSIFIER_CASES)
def test_invalid_classifier_counts_leave_existing_plan_and_data_unchanged(backing, name, value, tmp_path):
    session = _session(backing, tmp_path)
    _fit(session)
    plan = session.nlp_text_plan
    counts = plan.estimator_.feature_count_.copy()
    before, history = session.to_pandas().copy(deep=True), copy.deepcopy(session.history)
    options = {name: value}
    if name == "n_hash_features":
        options["vectorizer"] = "hashing"
    with pytest.raises(ValidationError):
        _fit(session, **options)
    assert session.nlp_text_plan is plan and session.history == history
    np.testing.assert_array_equal(plan.estimator_.feature_count_, counts)
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("method", ("nmf", "lda"))
@pytest.mark.parametrize("name,value", TOPIC_CASES)
def test_invalid_topic_counts_refuse_before_replacing_fitted_topics(backing, method, name, value, tmp_path):
    session = _session(backing, tmp_path)
    options = dict(method=method, n_topics=2, text_column="text", min_df=1, max_df=1.0,
        top_terms=3, max_iter=300, normalize_steps=["lowercase"], stopword_language=None)
    session.nlp.fit_topics(**options)
    plan = session.nlp_topic_plan
    basis = plan.model_.components_.copy()
    before, history = session.to_pandas().copy(deep=True), copy.deepcopy(session.history)
    options[name] = value
    with pytest.raises(ValidationError):
        session.nlp.fit_topics(**options)
    assert session.nlp_topic_plan is plan and session.history == history
    np.testing.assert_array_equal(plan.model_.components_, basis)
    pd.testing.assert_frame_equal(session.to_pandas(), before)


def _terms(document):
    words = document.lower().split()
    return words + [" ".join(words[i:i+2]) for i in range(len(words)-1)]


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("kind", ("count", "tfidf", "hashing"))
@pytest.mark.parametrize("integer_type", (int, np.int64))
def test_real_saved_classifier_matches_independent_bags_and_naive_bayes(backing, kind, integer_type, tmp_path):
    session = _session(backing, tmp_path)
    _fit(session, vectorizer=kind, ngram_range=(integer_type(1), integer_type(2)),
        n_hash_features=integer_type(32), max_features=integer_type(64), sublinear_tf=False)
    plan = session.nlp_text_plan
    documents = _frame().text.tolist()
    counters = [Counter(_terms(document)) for document in documents]
    vocabulary = sorted(set().union(*(set(row) for row in counters[:8])))
    if kind == "hashing":
        matrix = np.zeros((16, 32))
        for i, row in enumerate(counters):
            for term, count in row.items():
                matrix[i, abs(murmurhash3_32(term, seed=0)) % 32] += count
    else:
        names = plan.vectorizer_.get_feature_names_out().tolist()
        assert names == vocabulary and not any("future" in term or "hidden" in term for term in names)
        matrix = np.asarray([[row.get(term, 0) for term in vocabulary] for row in counters], dtype=float)
        if kind == "tfidf":
            df = (matrix[:8] > 0).sum(axis=0)
            matrix *= np.log(9 / (df + 1)) + 1
    if kind != "count":
        norms = np.linalg.norm(matrix, axis=1)
        matrix /= np.where(norms > 0, norms, 1)[:, None]
    np.testing.assert_allclose(plan.vectorizer_.transform(documents).toarray(), matrix, atol=1e-12)
    labels = _frame().label.iloc[:8].to_numpy()
    feature_count = np.vstack([matrix[:8][labels == label].sum(axis=0) for label in ("a", "b")])
    np.testing.assert_allclose(plan.estimator_.feature_count_, feature_count, atol=1e-12)
    log_prob = np.log((feature_count + 1) / (feature_count.sum(axis=1)[:, None] + matrix.shape[1]))
    scores = matrix[12:] @ log_prob.T + np.log(.5)
    expected = np.exp(scores - logsumexp(scores, axis=1)[:, None])
    predicted = session.nlp.predict(partition="test")
    np.testing.assert_allclose(predicted.probabilities, expected, atol=1e-12)
    np.testing.assert_array_equal(predicted.predictions, np.asarray(["a", "b"])[expected.argmax(axis=1)])
    basis = plan.estimator_.feature_count_.copy()
    path = session.nlp.save_bundle(tmp_path / "trusted-model")
    restored = _session(backing, tmp_path)
    restored.nlp.load_bundle(path, trusted=True)
    replay = restored.nlp.predict(partition="test")
    np.testing.assert_array_equal(replay.predictions, predicted.predictions)
    np.testing.assert_array_equal(replay.probabilities, predicted.probabilities)
    np.testing.assert_array_equal(plan.estimator_.feature_count_, basis)
