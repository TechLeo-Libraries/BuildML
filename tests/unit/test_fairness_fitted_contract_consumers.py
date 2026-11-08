"""Fitted class identity and task must govern public fairness decisions."""

import itertools

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from buildml import Session
from buildml.core.errors import ValidationError


def _session(labels="integers"):
    rng = np.random.default_rng(614)
    x = rng.normal(size=(180, 2))
    selected = x[:, 0] - .6*x[:, 1] + rng.normal(0, .65, len(x)) > 0
    y = selected if labels == "booleans" else np.where(selected, 7, 2)
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": y,
                          "group": np.where(np.arange(len(x)) % 3, "north", "south")})
    session = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "y": "target", "group": "ignore"})
    session.inject_split(train_indices=list(range(120)),
                         validation_indices=list(range(120, 150)),
                         test_indices=list(range(150, 180)))
    return session


@pytest.fixture(scope="module", params=list(itertools.product(
    ["booleans", "integers"], ["logistic", "pipeline", "tree"])))
def fitted(request):
    labels, kind = request.param
    model = DecisionTreeClassifier(max_depth=3, random_state=19) if kind == "tree" else LogisticRegression(C=.8)
    if kind == "pipeline":
        model = Pipeline([("model", model)])
    session = _session(labels).fit(model, task="classification")
    return session, labels, kind


def _scores(session, kind, partition, positive_index):
    """Independent logit or counted training-leaf frequencies, never predict_proba."""
    rows = session.partition(partition)
    estimator = session.fit_result.estimator
    model = estimator.named_steps["model"] if kind == "pipeline" else estimator
    features = rows[["x", "z"]]
    if kind == "tree":
        train = session.partition("train")
        train_leaf = model.apply(train[["x", "z"]])
        score_leaf = model.apply(features)
        truth = train.y.to_numpy() == model.classes_[positive_index]
        scores = np.array([truth[train_leaf == leaf].mean() for leaf in score_leaf])
    else:
        logit = (features.to_numpy() @ model.coef_.T + model.intercept_)[:, 0]
        scores = 1/(1 + np.exp(-logit))
        if positive_index == 0:
            scores = 1 - scores
    return rows, scores, model.classes_[positive_index]


def _check_outputs(session, kind, partition, positive_index, positive):
    rows, scores, _ = _scores(session, kind, partition, positive_index)
    truth = rows.y.to_numpy() == positive
    fingerprint = joblib.hash(session.fit_result)
    report = session.fairness.evaluate(sensitive_column="group", partition=partition,
                                       positive_label=positive)
    for group in ["north", "south"]:
        mask = rows.group.to_numpy() == group
        pos = scores[mask & truth]
        neg = scores[mask & ~truth]
        assert len(pos) and len(neg), "The oracle needs both outcomes per group."
        auc = np.mean([float(a > b) + .5*float(a == b) for a in pos for b in neg])
        assert report.classical_metrics_by_group[group]["roc_auc"] == pytest.approx(auc)
    for policy in ["demographic_parity", "equal_opportunity"]:
        eligible = np.ones(len(truth), dtype=bool) if policy == "demographic_parity" else truth
        reference = np.mean(scores[eligible] >= .5)
        thresholds = {}
        for group in ["north", "south"]:
            mask = (rows.group.to_numpy() == group) & eligible
            thresholds[group] = min(np.linspace(0, 1, 17), key=lambda threshold: (
                abs(np.mean(scores[mask] >= threshold) - reference), threshold))
        suggestion = session.fairness.suggest_thresholds(
            sensitive_column="group", partition=partition, positive_label=positive,
            target=policy, grid_size=17)
        assert suggestion.thresholds_by_group == pytest.approx(thresholds)
        for group, threshold in thresholds.items():
            mask = rows.group.to_numpy() == group
            assert suggestion.achieved_selection_rate_by_group[group] == pytest.approx(
                np.mean(scores[mask] >= threshold))
    assert joblib.hash(session.fit_result) == fingerprint


@pytest.mark.parametrize("positive_index", [0, 1])
@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("representation", ["native", "float", "numpy_float"])
def test_equivalent_numeric_positive_labels_select_the_fitted_probability_column(
        fitted, positive_index, partition, representation):
    session, labels, kind = fitted
    raw = session.fit_result.estimator.classes_[positive_index]
    # bool False/True and numeric 0/1 have the same equality semantics in rates.
    positive = int(raw) if representation == "native" else (
        float(raw) if representation == "float" else np.float64(raw))
    before = session.to_pandas().copy(deep=True)
    _check_outputs(session, kind, partition, positive_index, positive)
    pd.testing.assert_frame_equal(session.to_pandas(), before)


def _invoke(session, operation, positive, partition="validation"):
    options = dict(sensitive_column="group", partition=partition, positive_label=positive)
    if operation == "attach":
        return session.fairness.attach_to_last_eval(**options)
    if operation == "evaluate":
        return session.fairness.evaluate(**options)
    return session.fairness.suggest_thresholds(**options, target=operation, grid_size=17)


@pytest.mark.parametrize("invalid", ["string_lookalike", "absent", "nan", "none"])
@pytest.mark.parametrize("operation", ["evaluate", "attach", "demographic_parity", "equal_opportunity"])
def test_unknown_fitted_positive_class_is_refused_before_prediction_or_output_replacement(
        fitted, invalid, operation):
    session, labels, kind = fitted
    raw = session.fit_result.estimator.classes_[0]
    positive = {"string_lookalike": str(raw), "absent": 101, "nan": np.nan, "none": None}[invalid]
    # Preserve real, previously produced outputs when the next request is invalid.
    native = session.fit_result.estimator.classes_[1]
    session.fairness.evaluate(sensitive_column="group", partition="validation", positive_label=native)
    session.fairness.suggest_thresholds(sensitive_column="group", positive_label=native)
    report, suggestion = session._fairness_report, session._fairness_mitigation_suggestion
    history, fingerprint = list(session._history), joblib.hash(session.fit_result)
    with pytest.raises(ValidationError, match="positive_label"):
        _invoke(session, operation, positive)
    assert session._history == history
    assert session._fairness_report is report
    assert session._fairness_mitigation_suggestion is suggestion
    assert joblib.hash(session.fit_result) == fingerprint


@pytest.mark.parametrize("kind", ["ridge", "tree", "pipeline"])
@pytest.mark.parametrize("operation", ["evaluate", "attach", "demographic_parity", "equal_opportunity"])
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_binary_numeric_regression_is_not_a_fitted_classifier(kind, operation, partition):
    session = _session("booleans")
    model = DecisionTreeRegressor(max_depth=3, random_state=19) if kind == "tree" else Ridge()
    if kind == "pipeline":
        model = Pipeline([("model", model)])
    session.fit(model, task="regression")
    history, fingerprint = list(session._history), joblib.hash(session.fit_result)
    with pytest.raises(ValidationError, match="classifier|classification"):
        _invoke(session, operation, 1, partition)
    assert session._history == history
    assert session._fairness_report is None and session._fairness_mitigation_suggestion is None
    assert joblib.hash(session.fit_result) == fingerprint


@pytest.mark.parametrize("positive_index", [0, 1])
def test_public_model_roundtrip_preserves_numeric_fairness_class_identity(fitted, positive_index, tmp_path):
    source, labels, kind = fitted
    path = source.save_model(tmp_path / "owned-model")
    restored = _session(labels).load_model(path, trusted=True)
    positive = float(source.fit_result.estimator.classes_[positive_index])
    assert restored.fit_result.to_dict() == source.fit_result.to_dict()
    np.testing.assert_array_equal(restored.fit_result.estimator.classes_, source.fit_result.estimator.classes_)
    pd.testing.assert_series_equal(restored.predict(), source.predict())
    pd.testing.assert_frame_equal(restored.predict(return_proba=True), source.predict(return_proba=True))
    _check_outputs(restored, kind, "test", positive_index, positive)
