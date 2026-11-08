"""Weighted evaluation and winner promotion agree with independent probability arithmetic."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.model.compare import compare_estimators
from buildml.pipeline import predict_from_pipeline

KINDS = ["binary_strings", "binary_integers", "multiclass_strings", "multiclass_integers"]


def population(kind, weighted, separated=False):
    labels = {
        "binary_strings": ["warm", "cold"],
        "binary_integers": [20, -3],
        "multiclass_strings": ["warm", "cold", "neutral"],
        "multiclass_integers": [20, -3, 5],
    }[kind]
    rng = np.random.default_rng(628)
    codes = np.arange(84) % len(labels)
    x = rng.normal(size=(84, 2)) * (0.2 if separated else 1.0)
    x += codes[:, None] * ([4.0, -3.0] if separated else [0.7, -0.2])
    frame = pd.DataFrame(
        {
            "x": x[:, 0],
            "z": x[:, 1],
            "y": np.asarray(labels)[codes],
            "weight": np.asarray([0.125, 0.3, 0.55, 0.0, 1.1])[np.arange(84) % 5]
            * (1 + 0.4 * codes),
        },
        index=[f"original-{i}" for i in range(84)],
    )
    session = Session.ingest(frame).set_roles(
        {
            "x": "feature",
            "z": "feature",
            "y": "target",
            "weight": "weight" if weighted else "ignore",
        }
    )
    return session.inject_split(
        train_indices=list(range(60)),
        validation_indices=list(range(60, 72)),
        test_indices=list(range(72, 84)),
    )


def weights(frame, weighted):
    return frame.weight.to_numpy() if weighted else np.ones(len(frame))


def gaussian_probabilities(train, score, weighted):
    classes = np.unique(train.y)
    values = train[["x", "z"]].to_numpy()
    probe = score[["x", "z"]].to_numpy()
    w = weights(train, weighted)
    logs = []
    for label in classes:
        mask = train.y.to_numpy() == label
        mean = np.average(values[mask], axis=0, weights=w[mask])
        variance = np.average((values[mask] - mean) ** 2, axis=0, weights=w[mask])
        prior = w[mask].sum() / w.sum()
        logs.append(
            np.log(prior)
            - 0.5 * np.sum(np.log(2 * np.pi * variance) + (probe - mean) ** 2 / variance, axis=1)
        )
    logits = np.column_stack(logs)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return classes, exp / exp.sum(axis=1, keepdims=True)


def binary_auc(positive, probability, w):
    a, b = probability[positive], probability[~positive]
    pairs = (a[:, None] > b).astype(float) + 0.5 * (a[:, None] == b)
    return np.sum(pairs * w[positive, None] * w[None, ~positive]) / (
        w[positive].sum() * w[~positive].sum()
    )


def average_precision(positive, probability, w):
    total = w[positive].sum()
    previous = 0.0
    result = 0.0
    for threshold in np.unique(probability)[::-1]:
        chosen = probability >= threshold
        true_weight = w[chosen & positive].sum()
        selected_weight = w[chosen].sum()
        precision = true_weight / selected_weight if selected_weight else 0.0
        result += precision * (true_weight - previous) / total
        previous = true_weight
    return result


def probability_metrics(frame, classes, probability, weighted):
    truth = frame.y.to_numpy()
    w = weights(frame, weighted)
    columns = np.asarray([np.flatnonzero(classes == value)[0] for value in truth])
    loss = -np.average(
        np.log(np.clip(probability[np.arange(len(frame)), columns], np.finfo(float).eps, 1)),
        weights=w,
    )
    prediction = classes[probability.argmax(axis=1)]
    result = {"accuracy": np.average(prediction == truth, weights=w), "log_loss": loss}
    if len(classes) == 2:
        result["roc_auc"] = binary_auc(truth == classes[1], probability[:, 1], w)
        result["average_precision"] = average_precision(truth == classes[1], probability[:, 1], w)
    else:
        supports = np.asarray([w[truth == label].sum() for label in classes])
        values = [
            binary_auc(truth == label, probability[:, i], w) for i, label in enumerate(classes)
        ]
        result["roc_auc_ovr_weighted"] = np.average(values, weights=supports)
    return result


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_evaluation_support_probability_metrics_and_console_match_independent_arithmetic(
    kind, weighted, partition, capsys
):
    session = population(kind, weighted)
    before = session.to_pandas().copy(deep=True)
    template = GaussianNB(var_smoothing=0)
    session.fit(template, task="classification")
    train, score = session.partition("train"), session.partition(partition)
    classes, probability = gaussian_probabilities(train, score, weighted)
    prediction = classes[probability.argmax(axis=1)]
    np.testing.assert_allclose(
        session.fit_result.estimator.predict_proba(score[["x", "z"]]), probability, atol=1e-12
    )
    result = session.evaluate(partition=partition)
    assert result.n_rows == len(score)
    expected = probability_metrics(score, classes, probability, weighted)
    for key, value in expected.items():
        assert result.metrics[key] == pytest.approx(value, abs=2e-12)
    w, truth = weights(score, weighted), score.y.to_numpy()
    matrix = np.asarray(
        [[w[(truth == a) & (prediction == b)].sum() for b in classes] for a in classes]
    )
    labels = result.diagnostics["confusion_matrix"]["labels"]
    order = [list(map(str, classes)).index(label) for label in labels]
    np.testing.assert_allclose(
        result.diagnostics["confusion_matrix"]["matrix"], matrix[np.ix_(order, order)]
    )
    for i, label in enumerate(classes):
        row = result.diagnostics["per_class"][str(label)]
        support = w[truth == label].sum()
        assert row["support"] == pytest.approx(support)
        assert type(row["support"]) is (float if weighted else int)
        assert result.diagnostics["classification_report"][str(label)]["support"] == pytest.approx(
            support
        )
        precision = matrix[i, i] / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        recall = matrix[i, i] / support
        assert row["precision"] == pytest.approx(precision)
        assert row["recall"] == pytest.approx(recall)
    assert sum(row["support"] for row in result.diagnostics["per_class"].values()) == pytest.approx(
        w.sum()
    )
    encoded = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert encoded["diagnostics"]["per_class"] == result.diagnostics["per_class"]
    history = session.history
    result.show()
    text = capsys.readouterr().out
    assert f"partition={partition}" in text and f"log_loss: {expected['log_loss']:.6f}" in text
    assert session.history == history and not hasattr(template, "classes_")
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("metric", ["log_loss", "auc", "accuracy"])
@pytest.mark.parametrize("reverse", [False, True])
def test_comparison_promotes_and_saves_only_a_winner_with_available_ranking_evidence(
    kind, weighted, metric, reverse, tmp_path, capsys
):
    session = population(kind, weighted)
    before = session.to_pandas().copy(deep=True)
    candidates = {
        "no_probability": LinearSVC(dual=False, max_iter=5000),
        "gaussian": GaussianNB(var_smoothing=0),
        "prior": DummyClassifier(strategy="prior"),
    }
    if reverse:
        candidates = dict(reversed(list(candidates.items())))
    train, score = session.partition("train"), session.partition("validation")
    classes, probability = gaussian_probabilities(train, score, weighted)
    chosen_metric = (
        ("roc_auc" if len(classes) == 2 else "roc_auc_ovr_weighted") if metric == "auc" else metric
    )
    prior = np.asarray(
        [weights(train, weighted)[train.y.to_numpy() == label].sum() for label in classes]
    )
    prior /= prior.sum()
    prior_probability = np.tile(prior, (len(score), 1))
    expected = {
        "gaussian": probability_metrics(score, classes, probability, weighted),
        "prior": probability_metrics(score, classes, prior_probability, weighted),
    }
    result = session.compare_models(
        candidates, task="classification", partition="validation", ranking_metric=chosen_metric
    )
    assert all(row["n_rows"] == len(score) for row in result.rows)
    fitted = result.fits["no_probability"].estimator
    logits = score[["x", "z"]].to_numpy() @ fitted.coef_.T + fitted.intercept_
    predicted = (
        fitted.classes_[(logits[:, 0] > 0).astype(int)]
        if len(classes) == 2
        else fitted.classes_[logits.argmax(axis=1)]
    )
    expected["no_probability"] = {
        "accuracy": np.average(predicted == score.y.to_numpy(), weights=weights(score, weighted))
    }
    finite = [name for name in candidates if chosen_metric in expected[name]]
    finite.sort(key=lambda name: expected[name][chosen_metric], reverse=metric != "log_loss")
    missing = [name for name in candidates if name not in finite]
    assert [row["model"] for row in result.rows] == finite + missing
    for row in result.rows:
        if row["model"] in finite:
            assert row[chosen_metric] == pytest.approx(
                expected[row["model"]][chosen_metric], abs=2e-12
            )
        else:
            assert chosen_metric not in row
    assert list(result.to_frame().model) == finite + missing
    assert [
        row["model"] for row in json.loads(json.dumps(result.to_dict(), allow_nan=False))["rows"]
    ] == finite + missing
    assert session.fit_result is result.fits[finite[0]]
    assert session.history[-1]["result_summary"]["winner"] == finite[0]
    if missing:
        assert any("no_probability" in note and "last" in note for note in result.recommendations)
    result.show()
    shown = capsys.readouterr().out
    assert shown.index(finite[0]) < shown.index((finite + missing)[-1])
    assert all(not hasattr(candidate, "classes_") for candidate in candidates.values())
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    if metric == "log_loss":
        path = session.save_pipeline(tmp_path / "winner", evaluate_partition=None)
        incoming = score[["z", "x"]].copy()
        saved = predict_from_pipeline(path, incoming, trusted=True, return_proba=True)
        winner_probability = probability if finite[0] == "gaussian" else prior_probability
        np.testing.assert_array_equal(saved.predictions, classes[winner_probability.argmax(axis=1)])
        np.testing.assert_allclose(saved.probabilities.to_numpy(), winner_probability, atol=1e-12)


@pytest.mark.parametrize("fault", ["missing_loss", "missing_auc", "unknown_metric", "undefined_r2"])
@pytest.mark.parametrize("route", ["session", "direct"])
def test_no_finite_comparison_evidence_refuses_without_replacing_session_state(fault, route):
    if fault == "undefined_r2":
        session = Session.ingest(
            pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0) + 0.3})
        ).set_roles({"x": "feature", "y": "target"})
        session.inject_split(train_indices=list(range(6)), validation_indices=[6], test_indices=[7])
        session.compare_models(
            {"existing": LinearRegression()}, task="regression", partition="train"
        )
        candidates = {"ridge": Ridge(), "mean": DummyRegressor()}
        task, metric = "regression", "r2"
    else:
        session = population("binary_strings", True)
        session.compare_models({"existing": GaussianNB()}, task="classification", partition="train")
        candidates = {"svc": LinearSVC(dual=False, max_iter=5000)}
        task = "classification"
        metric = {
            "missing_loss": "log_loss",
            "missing_auc": "roc_auc",
            "unknown_metric": "unmeasured_score",
        }[fault]
    previous_fit, previous_comparison, history = (
        session.fit_result,
        session._last_comparison,
        session.history,
    )

    def attempt():
        if route == "session":
            return session.compare_models(candidates, task=task, ranking_metric=metric)
        return compare_estimators(
            session.dataset, session.split_plan, candidates, task=task, ranking_metric=metric
        )

    if fault == "undefined_r2":
        with pytest.warns(UndefinedMetricWarning, match="not well-defined") as warnings:
            with pytest.raises(ValidationError, match="finite.*ranking|ranking.*finite"):
                attempt()
        assert len(warnings) == 2
    else:
        with pytest.raises(ValidationError, match="finite.*ranking|ranking.*finite"):
            attempt()
    assert session.fit_result is previous_fit and session._last_comparison is previous_comparison
    assert session.history == history
    assert all(
        not hasattr(candidate, "classes_") and not hasattr(candidate, "coef_")
        for candidate in candidates.values()
    )


@pytest.mark.parametrize("kind", ["multiclass_strings", "multiclass_integers"])
@pytest.mark.parametrize("weighted", [False, True])
def test_missing_holdout_class_retains_log_loss_and_marks_auc_unavailable(kind, weighted):
    session = population(kind, weighted, separated=True)
    frame = session.to_pandas()
    absent = frame.y.iloc[2]
    test = [i for i in range(60, 84) if frame.y.iloc[i] != absent]
    validation = [i for i in range(60, 84) if frame.y.iloc[i] == absent]
    session.inject_split(
        train_indices=list(range(60)), validation_indices=validation, test_indices=test
    )
    session.fit(GaussianNB(var_smoothing=0), task="classification")
    train, score = session.partition("train"), session.partition("test")
    classes, probability = gaussian_probabilities(train, score, weighted)
    result = session.evaluate(partition="test")
    columns = np.asarray([np.flatnonzero(classes == label)[0] for label in score.y])
    expected = -np.average(
        np.log(np.clip(probability[np.arange(len(score)), columns], np.finfo(float).eps, 1)),
        weights=weights(score, weighted),
    )
    assert result.metrics["log_loss"] == pytest.approx(expected, abs=1e-12)
    assert "roc_auc_ovr_weighted" not in result.metrics
    assert any("Probability metrics unavailable" in note for note in result.recommendations)
    assert sum(row["support"] for row in result.diagnostics["per_class"].values()) == pytest.approx(
        weights(score, weighted).sum()
    )


@pytest.mark.parametrize("route", ["session", "direct"])
def test_empty_comparison_refuses_without_changing_existing_winner(route):
    session = population("binary_strings", False)
    session.fit(GaussianNB(), task="classification")
    before, history = session.fit_result, session.history
    with pytest.raises(ValueError, match="empty"):
        if route == "session":
            session.compare_models({})
        else:
            compare_estimators(session.dataset, session.split_plan, {})
    assert session.fit_result is before and session.history == history
