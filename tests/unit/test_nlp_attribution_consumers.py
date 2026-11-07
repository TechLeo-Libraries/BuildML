"""Token explanations agree with native score identities and strict JSON consumers."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import logsumexp

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.nlp.interpret import interpret_text_prediction
from buildml.nlp.predict import transform_documents

BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")
HEADS = ("logistic", "linear_svm", "sgd", "multinomial_nb", "complement_nb")
BAD_LIMITS = [("top_k", v) for v in (0, -1, True, False, 1.5, "2", None, np.nan, np.inf)]
BAD_LIMITS += [("max_documents", v) for v in (-1, True, False, 1.5, "2", None, np.nan, np.inf)]


def _frame(n_classes=2):
    variants = {
        "a": ("red red fox", "red shared fox"),
        "b": ("blue blue whale", "blue shared whale"),
        "c": ("green green bird", "green shared bird"),
    }
    sizes = (6, 6) if n_classes == 2 else (6, 4, 2)
    texts, labels = [], []
    for (label, options), size in zip(variants.items(), sizes):
        texts.extend(options[i % 2] for i in range(size))
        labels.extend([label] * size)
    probe = [
        "red fox",
        "blue whale",
        "red blue shared",
        "",
        "zzunseen nowhere",
        "green bird" if n_classes == 3 else "red red blue whale",
    ]
    texts += probe * 2
    labels += [list(variants)[i % n_classes] for i in range(12)]
    return pd.DataFrame({"text": texts, "label": labels, "row": [f"record-{i}" for i in range(24)]})


def _session(backing, directory, n_classes=2):
    frame = _frame(n_classes)
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "documents.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    return session.set_roles({"text": "feature", "label": "target", "row": "id"}).inject_split(
        train_indices=list(range(12)),
        validation_indices=list(range(12, 18)),
        test_indices=list(range(18, 24)),
    )


def _fit(session, head="logistic", kind="count"):
    session.nlp.fit_classifier(
        text_column="text",
        estimator=head,
        vectorizer=kind,
        normalize_steps=["lowercase"],
        stopword_language=None,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2),
        sublinear_tf=False,
        alpha=0.5,
        C=0.75,
        random_state=17,
    )
    return session.nlp_text_plan


def _matrix(frame, plan, kind):
    def terms(text):
        words = re.findall(r"(?u)\b\w\w+\b", text.lower())
        return Counter(words + [" ".join(pair) for pair in zip(words, words[1:])])

    bags = [terms(text) for text in frame.text]
    vocabulary = sorted(set().union(*(set(row) for row in bags[:12])))
    assert list(plan.feature_names_) == vocabulary
    assert "zzunseen" not in vocabulary and "nowhere" not in vocabulary
    matrix = np.asarray([[row.get(term, 0) for term in vocabulary] for row in bags], dtype=float)
    if kind == "tfidf":
        df = (matrix[:12] > 0).sum(axis=0)
        matrix *= np.log(13 / (1 + df)) + 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms > 0, norms, 1.0)
    return matrix


def _weights_and_scores(frame, plan, matrix, head):
    model = plan.estimator_
    if head in {"multinomial_nb", "complement_nb"}:
        y = frame.label.iloc[:12].to_numpy()
        counts = np.asarray([matrix[:12][y == label].sum(axis=0) for label in plan.classes_])
        np.testing.assert_allclose(model.feature_count_, counts, atol=1e-12)
        if head == "multinomial_nb":
            smoothed = counts + 0.5
            raw_weights = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
            prior = np.log(np.asarray([np.sum(y == label) for label in plan.classes_]) / 12)
        else:
            complement = counts.sum(axis=0) - counts + 0.5
            raw_weights = -np.log(complement / complement.sum(axis=1, keepdims=True))
            prior = np.zeros(len(plan.classes_))
        np.testing.assert_allclose(model.feature_log_prob_, raw_weights, atol=1e-12)
        raw_scores = matrix @ raw_weights.T + prior
        probability = np.exp(raw_scores - logsumexp(raw_scores, axis=1, keepdims=True))
        np.testing.assert_allclose(
            model.predict_proba(transform_documents(plan, frame.text.tolist())),
            probability,
            atol=1e-12,
        )
        return raw_weights - raw_weights.mean(axis=0), raw_scores
    weights = model.coef_.toarray() if sparse.issparse(model.coef_) else np.asarray(model.coef_)
    intercept = np.asarray(model.intercept_)
    if weights.shape[0] == 1:
        weights, intercept = np.vstack([-weights[0], weights[0]]), np.r_[-intercept, intercept]
    scores = matrix @ weights.T + intercept
    actual = model.decision_function(transform_documents(plan, frame.text.tolist()))
    np.testing.assert_allclose(
        actual, scores[:, 1] if len(plan.classes_) == 2 else scores, atol=1e-10
    )
    return weights, scores


def _assert_result(report, plan, matrix, weights, scores, positions, row_labels):
    index = list(plan.classes_).index(report.target_class)
    assert report.n_documents == len(positions) == len(report.document_attributions)
    assert report.document_row_labels == tuple(row_labels)
    expected_labels = np.asarray(plan.classes_)[scores[positions].argmax(axis=1)]
    assert tuple(report.document_predictions) == tuple(expected_labels)
    for position, tokens in zip(positions, report.document_attributions, strict=True):
        expected = {
            name: (weights[index, j], matrix[position, j])
            for j, name in enumerate(plan.feature_names_)
            if matrix[position, j] != 0
        }
        assert {t.token for t in tokens} == set(expected)
        for token in tokens:
            weight, value = expected[token.token]
            assert token.weight == pytest.approx(weight, abs=1e-11)
            assert token.value == pytest.approx(value, abs=1e-11)
            assert token.contribution == pytest.approx(weight * value, abs=1e-11)
        assert sum(t.contribution for t in tokens) == pytest.approx(
            matrix[position] @ weights[index], abs=1e-10
        )
    for class_index, label in enumerate(plan.classes_):
        tokens = report.global_top_tokens[label]
        assert {t.token for t in tokens} == set(plan.feature_names_)
        for token in tokens:
            column = list(plan.feature_names_).index(token.token)
            assert (
                token.weight
                == token.contribution
                == pytest.approx(weights[class_index, column], abs=1e-11)
            )


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("kind", ["count", "tfidf"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_native_scores_token_sums_and_saved_explanations(backing, head, kind, n_classes, tmp_path):
    session = _session(backing, tmp_path, n_classes)
    before = session.to_pandas().copy(deep=True)
    plan = _fit(session, head, kind)
    frame = _frame(n_classes)
    matrix = _matrix(frame, plan, kind)
    np.testing.assert_allclose(
        transform_documents(plan, frame.text.tolist()).toarray(), matrix, atol=1e-12
    )
    weights, scores = _weights_and_scores(frame, plan, matrix, head)
    model_state = {
        name: value.copy()
        for name, value in vars(plan.estimator_).items()
        if isinstance(value, np.ndarray)
    }
    for partition, positions in [
        ("validation", list(range(12, 18))),
        ("test", list(range(18, 24))),
    ]:
        for label in plan.classes_:
            report = session.nlp.interpret(
                partition=partition,
                target_class=label,
                top_k=len(plan.feature_names_),
                max_documents=6,
            )
            _assert_result(
                report, plan, matrix, weights, scores, positions, session.partition(partition).index
            )
    path = session.nlp.save_bundle(tmp_path / "nlp")
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    destination = _session(backing, tmp_path, n_classes)
    with pytest.raises(ValidationError, match="trust"):
        destination.nlp.load_bundle(path)
    destination.nlp.load_bundle(path, trusted=True)
    report = destination.nlp.interpret(
        partition="test",
        target_class=plan.classes_[0],
        top_k=len(plan.feature_names_),
        max_documents=6,
    )
    _assert_result(
        report,
        plan,
        matrix,
        weights,
        scores,
        list(range(18, 24)),
        destination.partition("test").index,
    )
    assert {
        p.name: sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()
    } == hashes
    for name, value in model_state.items():
        np.testing.assert_array_equal(vars(plan.estimator_)[name], value)
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("head", HEADS)
def test_global_explanation_is_consumable_by_strict_json(backing, head, tmp_path):
    session = _session(backing, tmp_path)
    _fit(session, head)
    result = session.nlp.interpret(top_k=3, max_documents=2)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    for tokens in payload["global_top_tokens"].values():
        assert all(t["value"] is None for t in tokens)
    assert all(
        isinstance(t["value"], float) for row in payload["document_attributions"] for t in row
    )


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("head", ["logistic", "linear_svm", "sgd"])
def test_native_sparse_coefficient_storage_and_saved_consumer(backing, head, tmp_path):
    session = _session(backing, tmp_path)
    plan = _fit(session, head)
    plan.estimator_.sparsify()
    assert sparse.issparse(plan.estimator_.coef_)
    before = plan.estimator_.coef_.toarray().copy()
    frame = _frame()
    matrix = _matrix(frame, plan, "count")
    weights, scores = _weights_and_scores(frame, plan, matrix, head)
    result = session.nlp.interpret(
        target_class=plan.classes_[0], top_k=len(plan.feature_names_), max_documents=6
    )
    _assert_result(
        result, plan, matrix, weights, scores, list(range(18, 24)), session.partition("test").index
    )
    path = session.nlp.save_bundle(tmp_path / "sparse")
    destination = _session(backing, tmp_path)
    destination.nlp.load_bundle(path, trusted=True)
    loaded = destination.nlp_text_plan
    assert sparse.issparse(loaded.estimator_.coef_)
    result = destination.nlp.interpret(
        target_class=plan.classes_[0], top_k=len(plan.feature_names_), max_documents=6
    )
    _assert_result(
        result,
        loaded,
        matrix,
        weights,
        scores,
        list(range(18, 24)),
        destination.partition("test").index,
    )
    np.testing.assert_array_equal(plan.estimator_.coef_.toarray(), before)
    np.testing.assert_array_equal(loaded.estimator_.coef_.toarray(), before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("integer_type", [int, np.int64])
def test_valid_caps_and_global_only_requests_preserve_discrete_counts(
    backing, integer_type, tmp_path
):
    session = _session(backing, tmp_path)
    plan = _fit(session)
    for cap in [0, 2]:
        result = session.nlp.interpret(top_k=integer_type(2), max_documents=integer_type(cap))
        assert result.n_documents == cap
        assert result.target_class == plan.classes_[-1]
        assert all(len(row) <= 2 for row in result.document_attributions)
        assert all(len(row) == 2 for row in result.global_top_tokens.values())
    result = session.nlp.interpret(include_global=False, max_documents=0)
    assert not result.document_attributions and not result.global_top_tokens


@pytest.mark.parametrize("route", ["module", "session"])
@pytest.mark.parametrize("name,value", BAD_LIMITS)
def test_bad_limits_refuse_before_changing_saved_explanation(route, name, value, tmp_path):
    session = _session("pandas", tmp_path)
    plan = _fit(session)
    prior = session.nlp.interpret()
    history = copy.deepcopy(session.history)
    weights = plan.estimator_.coef_.copy()
    with pytest.raises(ValidationError, match=name):
        if route == "module":
            interpret_text_prediction(session.dataset, plan, session.split_plan, **{name: value})
        else:
            session.nlp.interpret(**{name: value})
    assert session.nlp_text_plan is plan and session.nlp.interpret_result is prior
    assert session.history == history
    np.testing.assert_array_equal(plan.estimator_.coef_, weights)


@pytest.mark.parametrize("backing", BACKINGS)
def test_unknown_class_refuses_without_changing_fitted_head(backing, tmp_path):
    session = _session(backing, tmp_path)
    plan = _fit(session)
    before = plan.estimator_.coef_.copy()
    with pytest.raises(ValidationError, match="target_class"):
        session.nlp.interpret(target_class="never fitted")
    np.testing.assert_array_equal(plan.estimator_.coef_, before)


@pytest.mark.parametrize("backing", BACKINGS)
def test_actual_hashing_representation_refuses_token_attributions(backing, tmp_path):
    session = _session(backing, tmp_path)
    plan = _fit(session, kind="hashing")
    before = plan.estimator_.coef_.copy()
    with pytest.raises(ValidationError, match="invertible|hashing"):
        session.nlp.interpret()
    np.testing.assert_array_equal(plan.estimator_.coef_, before)
