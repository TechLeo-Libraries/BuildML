"""Evaluation populations and regression failures obey actual row-level math."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.symbolic.evaluate import evaluate_neuro_symbolic, evaluate_symbolic


def _session(task, path):
    x = np.linspace(-2, 2, 80)
    frame = pd.DataFrame({"x": x, "y": (x > 0).astype(int) if task == "classification" else 2 * x + 0.5})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )
    if path == "symbolic":
        rules = [{"if": [{"column": "x", "op": op, "value": 0}], "then": label}
                 for op, label in [("<=", 0), (">", 1 if task == "classification" else 2)]]
        session.symbolic.fit(source="declared", task=task, rules=rules)
    else:
        session.symbolic.fit_neuro(mode=path, task=task,
            base_estimator="logistic_regression" if task == "classification" else "ridge",
            rule_source="declared", rules=[{"if": [{"column": "x", "op": ">", "value": 1000}], "then": 0}])
    return session


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("path", ["symbolic", "constraint_overlay", "constraint_repair", "rules_as_features"])
@pytest.mark.parametrize("partition", ["train", "all", "validation", "test"])
def test_population_disclosures_and_metrics_match_independent_fit(task, path, partition):
    session = _session(task, path)
    frame = session.dataset.frame
    train = session.partition("train")
    selected = frame if partition == "all" else session.partition(partition)
    original = frame.copy(deep=True)
    if path == "symbolic":
        prediction = np.where(selected.x > 0, 1 if task == "classification" else 2, 0)
        evaluate = session.symbolic.evaluate
        model_state = None
    else:
        x, z = train[["x"]].to_numpy(), selected[["x"]].to_numpy()
        if path == "rules_as_features":
            x, z = np.column_stack([x, np.zeros(len(x))]), np.column_stack([z, np.zeros(len(z))])
        if task == "classification":
            prediction = LogisticRegression(max_iter=500, random_state=0).fit(x, train.y).predict(z)
        else:
            # Ridge normal equations, with an unpenalized intercept.
            centered = x - x.mean(axis=0)
            beta = np.linalg.solve(centered.T @ centered + np.eye(x.shape[1]), centered.T @ (train.y.to_numpy() - train.y.mean()))
            prediction = (z - x.mean(axis=0)) @ beta + train.y.mean()
        evaluate = session.symbolic.evaluate_neuro
        model_state = session.neuro_symbolic_plan.estimator_.coef_.copy()
    scored = evaluate(partition=partition)
    assert scored.n_rows == len(selected)
    if task == "classification":
        assert scored.metrics["accuracy"] == pytest.approx(np.mean(prediction == selected.y))
    else:
        residual = selected.y.to_numpy() - prediction
        assert scored.metrics["rmse"] == pytest.approx(np.sqrt(np.mean(residual ** 2)))
        assert scored.metrics["mae"] == pytest.approx(np.mean(np.abs(residual)))
        assert scored.metrics["r2"] == pytest.approx(1 - np.sum(residual ** 2) / np.sum((selected.y - selected.y.mean()) ** 2))
    if partition in {"train", "all"}:
        assert any("training rows" in warning.lower() for warning in scored.warnings)
        assert not any("holdout evaluation only" in text.lower() for text in scored.disclosures)
    else:
        assert not any("training rows" in warning.lower() for warning in scored.warnings)
    if model_state is not None:
        np.testing.assert_array_equal(session.neuro_symbolic_plan.estimator_.coef_, model_state)
    pd.testing.assert_frame_equal(session.dataset.frame, original)


@pytest.mark.parametrize("path", ["symbolic", "constraint_overlay"])
@pytest.mark.parametrize("bad_target", [float("inf"), float("-inf"), complex(1, 2), "unscorable"])
def test_regression_bad_targets_raise_actionable_validation_error(path, bad_target):
    session = _session("regression", path)
    frame = session.dataset.frame.copy()
    frame["y"] = frame.y.astype(object)
    frame.iloc[session.split_plan.test_indices[0], frame.columns.get_loc("y")] = bad_target
    external = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    function = evaluate_symbolic if path == "symbolic" else evaluate_neuro_symbolic
    plan = session.symbolic_plan if path == "symbolic" else session.neuro_symbolic_plan
    with pytest.raises(ValidationError, match="regression.*finite|finite.*regression"):
        function(external.dataset, plan, session.split_plan, partition="test")


@pytest.mark.parametrize("bad_prediction", [None, float("nan"), float("inf"), complex(1, 2), "unscorable"])
def test_declared_regression_bad_prediction_is_refused(bad_prediction):
    session = _session("regression", "symbolic")
    session.symbolic.fit(source="declared", task="regression", rules=[{
        "if": [{"column": "x", "op": ">", "value": 1000}], "then": 0
    }], default_consequent=bad_prediction)
    with pytest.raises(ValidationError, match="regression"):
        session.symbolic.evaluate(partition="test")


@pytest.mark.parametrize("path", ["symbolic", "constraint_overlay"])
@pytest.mark.parametrize("bad_target", ["missing", "null"])
def test_missing_or_null_target_is_refused_without_silent_drop(path, bad_target):
    session = _session("classification", path)
    frame = session.dataset.frame.copy()
    if bad_target == "missing":
        frame = frame.drop(columns="y")
    else:
        frame["y"] = frame.y.astype(object)
        frame.iloc[session.split_plan.test_indices[0], frame.columns.get_loc("y")] = None
    external = Session.ingest(frame)
    original = external.dataset.frame.copy(deep=True)
    function = evaluate_symbolic if path == "symbolic" else evaluate_neuro_symbolic
    plan = session.symbolic_plan if path == "symbolic" else session.neuro_symbolic_plan
    with pytest.raises(ValidationError, match="missing|Missing|null"):
        function(external.dataset, plan, session.split_plan, partition="test")
    pd.testing.assert_frame_equal(external.dataset.frame, original)
