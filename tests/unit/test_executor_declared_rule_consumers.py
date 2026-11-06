"""Confirmed agent rule workflows preserve the caller's actual constraints."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def raw(kind):
    x = np.tile([-1.5, -0.8, 0.0, 0.4, 1.4, 2.1], 12)
    z = np.random.default_rng(874).normal(size=len(x))
    signal = 1.4 * x - 0.2 * z
    y = signal + 0.1 * np.sin(np.arange(len(x)))
    if kind == "strings":
        y = np.where(signal > 0, "grant", "deny")
    elif kind == "integers":
        y = np.where(signal > 0, 23, -7)
    elif kind == "booleans":
        y = signal > 0
    return pd.DataFrame({"x": x, "z": z, "target": y}, index=1000 + 3 * np.arange(len(x)))


def ready(kind="regression"):
    return (
        Session.ingest(raw(kind))
        .set_roles({"x": "feature", "z": "feature", "target": "target"})
        .inject_split(
            train_indices=list(range(48)),
            validation_indices=list(range(48, 60)),
            test_indices=list(range(60, 72)),
        )
    )


def values(kind):
    return {
        "regression": (10.0, -10.0),
        "strings": ("grant", "deny"),
        "integers": (23, -7),
        "booleans": (True, False),
    }[kind]


def rules(kind):
    positive, negative = values(kind)
    return [
        {
            "rule_id": "reviewed-hard",
            "if": [{"column": "x", "op": ">=", "value": 1.0}],
            "then": negative,
            "priority": 10000,
            "kind": "constraint",
            "hardness": "hard",
        },
        {
            "rule_id": "reviewed-soft",
            "if": [{"column": "x", "op": ">=", "value": 0.0}],
            "then": positive,
            "priority": 9000,
            "kind": "regression" if kind == "regression" else "classification",
            "hardness": "soft",
            "strength": 1.0,
        },
    ]


def run(s, name, args, *, confirmed=True):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, copy.deepcopy(args), registry)
    return execute_tool(s, proposal, confirmed=confirmed, registry=registry)


def success(s, name, args):
    result = run(s, name, args)
    assert result.executed and result.error is None, result.error
    return result.result


def options(kind, hybrid=False):
    result = {
        "backend": "sklearn",
        "task": "regression" if kind == "regression" else "classification",
        "rules": rules(kind),
    }
    if hybrid:
        result.update(
            rule_source="declared",
            mode="constraint_overlay",
            base_estimator="ridge" if kind == "regression" else "logistic_regression",
            soft_strength=0.25,
        )
    else:
        result.update(source="declared", default_consequent=values(kind)[0])
    return result


def equal_values(actual, expected, kind):
    if kind == "regression":
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    else:
        assert list(actual) == list(expected)
        assert all(type(a) is type(b) for a, b in zip(actual, expected))


def bundle_roundtrip(s, kind, hybrid, tmp_path):
    name = "predict_neuro_symbolic" if hybrid else "predict_symbolic"
    before = success(s, name, {"partition": "test", "return_traces": True})
    path = tmp_path / "actual-agent-bundle"
    saved = success(s, "save_symbolic_bundle", {"path": str(path)})
    assert saved["path"] == str(path)
    restored = ready(kind)
    refusal = run(restored, "load_symbolic_bundle", {"path": str(path)})
    assert not refusal.executed and "trusted" in refusal.error
    success(restored, "load_symbolic_bundle", {"path": str(path), "trusted": True})
    after = success(restored, name, {"partition": "test", "return_traces": True})
    assert after.to_dict() == before.to_dict()
    assert after.traces == before.traces
    equal_values(after.predictions, before.predictions, kind)
    json.dumps(after.to_dict(), allow_nan=False)


@pytest.mark.parametrize("kind", ["regression", "strings", "integers", "booleans"])
@pytest.mark.parametrize("default_positive", [False, True])
def test_declared_symbolic_agent_rules_fallback_labels_metrics_and_bundle(
    kind, default_positive, tmp_path
):
    s = ready(kind)
    original = s.dataset.to_pandas().copy(deep=True)
    args = options(kind)
    positive, negative = values(kind)
    fallback = positive if default_positive else negative
    args["default_consequent"] = fallback
    success(s, "fit_symbolic", args)
    q = s.partition("test")
    expected = [negative if x >= 1 else positive if x >= 0 else fallback for x in q.x]
    prediction = success(s, "predict_symbolic", {"partition": "test", "return_traces": True})
    equal_values(prediction.predictions, expected, kind)
    assert [t.row_index for t in prediction.traces] == q.index.tolist()
    assert [t.chosen_rule_id for t in prediction.traces] == [
        "reviewed-hard" if x >= 1 else "reviewed-soft" if x >= 0 else None for x in q.x
    ]
    evaluation = success(s, "evaluate_symbolic", {"partition": "test"})
    if kind == "regression":
        assert evaluation.metrics["mae"] == pytest.approx(
            np.abs(np.asarray(expected) - q.target).mean()
        )
    else:
        assert evaluation.metrics["accuracy"] == pytest.approx(
            np.mean(np.asarray(expected) == q.target)
        )
    assert s._symbolic_plan.knowledge_base.provenance == "declared"
    bundle_roundtrip(s, kind, False, tmp_path)
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


@pytest.mark.parametrize("kind", ["regression", "strings", "integers", "booleans"])
@pytest.mark.parametrize("mode", ["constraint_overlay", "constraint_repair", "rules_as_features"])
def test_declared_hybrid_agent_predictions_match_linear_algebra_rule_order_and_bundles(
    kind, mode, tmp_path
):
    s = ready(kind)
    args = options(kind, True)
    args["mode"] = mode
    success(s, "fit_neuro_symbolic", args)
    plan = s._neuro_symbolic_plan
    train, q = s.partition("train"), s.partition("test")
    x_train, x = train[["x", "z"]].to_numpy(), q[["x", "z"]].to_numpy()
    if mode == "rules_as_features":
        x_train = np.column_stack([x_train, train.x >= 1, train.x >= 0])
        x = np.column_stack([x, q.x >= 1, q.x >= 0])
    if kind == "regression":
        centered = x_train - x_train.mean(axis=0)
        coefficients = np.linalg.solve(
            centered.T @ centered + np.eye(x_train.shape[1]),
            centered.T @ (train.target.to_numpy() - train.target.mean()),
        )
        neural = (x - x_train.mean(axis=0)) @ coefficients + train.target.mean()
        np.testing.assert_allclose(plan.estimator_.coef_, coefficients, rtol=1e-11, atol=1e-12)
    else:
        logits = (x @ plan.estimator_.coef_.T + plan.estimator_.intercept_).ravel()
        probabilities = 1 / (1 + np.exp(-logits))
        labels = plan.label_encoder_.classes_.tolist()
        neural = [labels[int(value >= 0.5)] for value in probabilities]
    positive, negative = values(kind)
    expected = []
    for value, base in zip(q.x, neural):
        if mode == "rules_as_features":
            prediction = base
        elif value >= 1:
            prediction = negative
        elif value >= 0 and kind == "regression":
            prediction = 0.75 * float(base) + 0.25 * positive
        else:
            prediction = base
        expected.append(prediction)
    result = success(s, "predict_neuro_symbolic", {"partition": "test", "return_traces": True})
    equal_values(result.neural_predictions, neural, kind)
    equal_values(result.predictions, expected, kind)
    assert [t.row_index for t in result.traces] == q.index.tolist()
    bundle_roundtrip(s, kind, True, tmp_path)


@pytest.mark.parametrize("kind", ["regression", "strings"])
def test_agent_merges_supplied_constraints_with_real_train_induction(kind):
    agent, direct = ready(kind), ready(kind)
    args = options(kind, True)
    args.update(rule_source="decision_tree", max_depth=2)
    success(agent, "fit_neuro_symbolic", args)
    direct.symbolic.fit_neuro(**args)
    actual = success(agent, "predict_neuro_symbolic", {"partition": "test"})
    expected = direct.symbolic.predict_neuro(partition="test")
    assert actual.to_dict() == expected.to_dict()
    assert actual.traces == expected.traces
    assert agent._neuro_symbolic_plan.knowledge_base.provenance == "mixed_declared_and_induced"
    _, negative = values(kind)
    for value, predicted in zip(agent.partition("test").x, actual.predictions):
        if value >= 1:
            assert predicted == negative


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("fault", ["empty", "target", "missing-column", "operator", "malformed"])
def test_agent_bad_declared_rules_preserve_existing_real_plan_and_evidence(hybrid, fault):
    s = ready()
    args = options("regression", hybrid)
    if hybrid:
        s.symbolic.fit_neuro(**args)
    else:
        s.symbolic.fit(**args)
    name = "fit_neuro_symbolic" if hybrid else "fit_symbolic"
    plan = s._neuro_symbolic_plan if hybrid else s._symbolic_plan
    fingerprint, history, original = (
        joblib.hash(plan),
        s.history,
        s.dataset.to_pandas().copy(deep=True),
    )
    if fault == "empty":
        args["rules"] = []
    elif fault == "malformed":
        args["rules"] = ["not a rule mapping"]
    elif fault == "operator":
        args["rules"][0]["if"][0]["op"] = "unsupported"
    else:
        args["rules"][0]["if"][0]["column"] = "target" if fault == "target" else "absent"
    result = run(s, name, args)
    assert not result.executed and result.error
    assert (s._neuro_symbolic_plan if hybrid else s._symbolic_plan) is plan
    assert joblib.hash(plan) == fingerprint and s.history == history
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


@pytest.mark.parametrize("hybrid", [False, True])
def test_rule_payload_is_declared_in_provider_schema_and_still_requires_confirmation(hybrid):
    name = "fit_neuro_symbolic" if hybrid else "fit_symbolic"
    registry = build_default_registry()
    properties = registry.get(name).parameters["properties"]
    assert properties["rules"]["type"] == "array"
    if not hybrid:
        assert "default_consequent" in properties
    s = ready()
    history = s.history
    result = run(s, name, options("regression", hybrid), confirmed=False)
    assert not result.executed and "confirmation" in result.error.lower()
    assert s._symbolic_plan is None and s._neuro_symbolic_plan is None and s.history == history
