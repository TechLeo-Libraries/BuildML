"""Agent decision policies checked against independent holdout cost calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, arguments, *, success=True):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed is success, result.error
    return result.result if success else result.error


def _session():
    x, y = make_classification(n_samples=180, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=3, random_state=921)
    frame = pd.DataFrame(x, columns=list("abcde"))
    frame["outcome"] = np.asarray(["apple", "mango", "pear"])[y]
    return (Session.ingest(frame)
            .set_roles({**dict.fromkeys(list("abcde"), "feature"), "outcome": "target"})
            .split(test_size=.25, validation_size=.25, random_state=7)
            .fit(LogisticRegression(max_iter=300), task="classification")
            .ai.configure(provider="mock"))


@pytest.mark.parametrize("partition", ["test", "validation"])
def test_agent_bayes_actions_and_realized_cost_match_numpy(partition, tmp_path):
    session = _session()
    matrix = np.asarray([[0., 2., 9.], [3., 0., 1.], [8., 4., 0.]])
    train_before = tuple(session.split_plan.train_indices)
    holdout_before = session.partition("test").copy()
    _run(session, "fit_decision_policy", {
        "method": "cost_matrix", "backend": "native", "partition": "validation",
        "cost_matrix": matrix.tolist(), "class_labels": ["apple", "mango", "pear"],
    })
    frame = session.partition(partition)
    estimator = session._fit_result.estimator
    probabilities = estimator.predict_proba(frame[list("abcde")])
    expected_indices = (probabilities @ matrix).argmin(axis=1)
    expected_actions = np.asarray(estimator.classes_)[expected_indices]
    label_indices = {label: i for i, label in enumerate(estimator.classes_)}
    true_indices = frame.outcome.map(label_indices).to_numpy()
    expected_cost = matrix[true_indices, expected_indices].sum()
    applied = _run(session, "apply_decisions", {"partition": partition})
    np.testing.assert_array_equal(applied.decisions, expected_actions)
    evaluated = _run(session, "evaluate_decisions", {"partition": partition})
    assert evaluated.realized_cost == pytest.approx(expected_cost)
    assert evaluated.metrics["realized_cost_mean"] == pytest.approx(expected_cost / len(frame))
    assert evaluated.metrics["accuracy"] == pytest.approx((expected_actions == frame.outcome).mean())
    assert any("in-sample" in warning for warning in evaluated.warnings) is (partition == "validation")
    assert tuple(session.split_plan.train_indices) == train_before
    pd.testing.assert_frame_equal(session.partition("test"), holdout_before)

    path = str(tmp_path / "decision")
    _run(session, "save_decision_bundle", {"path": path})
    restored = _session()
    error = _run(restored, "load_decision_bundle", {"path": path}, success=False)
    assert "trusted=True" in error
    assert restored.decision_plan is None
    _run(restored, "load_decision_bundle", {"path": path, "trusted": True})
    after = _run(restored, "evaluate_decisions", {"partition": partition})
    applied_again = _run(restored, "apply_decisions", {"partition": partition})
    np.testing.assert_array_equal(applied_again.decisions, expected_actions)
    assert after.metrics == evaluated.metrics


def test_agent_refuses_test_policy_tuning_without_changing_session():
    session = _session()
    error = _run(session, "fit_decision_policy", {
        "method": "cost_matrix", "backend": "native", "partition": "test",
        "cost_matrix": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        "class_labels": ["apple", "mango", "pear"],
    }, success=False)
    assert "allow_test_tuning" in error
    assert session.decision_plan is None


@pytest.mark.parametrize("method", ["topk", "knapsack", "lp_allocate"])
def test_agent_allocation_holdout_value_and_utilization_match_sorted_oracle(method):
    frame = pd.DataFrame({"value": np.arange(1., 61.), "cost": 2.})
    session = (Session.ingest(frame).split(test_size=.3, validation_size=.3, random_state=19)
               .ai.configure(provider="mock"))
    arguments = {"method": method, "backend": "native", "partition": "validation",
                 "score_source": "column", "score_column": "value",
                 "value_column": "value", "cost_column": "cost"}
    arguments.update({"capacity": 2} if method == "topk" else {"budget": 5.})
    _run(session, "fit_decision_policy", arguments)
    holdout = session.partition("test")
    values = np.sort(holdout.value.to_numpy())[::-1]
    expected_value = values[:2].sum()
    expected_cost = 4.
    if method == "lp_allocate":
        expected_value += .5 * values[2]
        expected_cost = 5.
    applied = _run(session, "apply_decisions", {"partition": "test"})
    evaluated = _run(session, "evaluate_decisions", {"partition": "test"})
    assert applied.selected_value == pytest.approx(expected_value)
    assert evaluated.selected_value == pytest.approx(expected_value)
    assert evaluated.selected_cost == pytest.approx(expected_cost)
    assert evaluated.n_selected == (3 if method == "lp_allocate" else 2)
    if method == "topk":
        assert evaluated.metrics["capacity_utilization"] == 1.
    else:
        assert evaluated.metrics["budget_utilization"] == pytest.approx(expected_cost / 5.)
