"""Rule inputs cannot read scoring truth; valid hybrids obey scalar arithmetic."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.symbolic.rules import parse_declared_rules

KINDS = ["strings", "integers", "booleans", "regression"]
MODES = ["constraint_overlay", "rules_as_features", "constraint_repair"]


def _frame(kind):
    rng = np.random.default_rng(8517)
    x = np.tile([-1.5, -.5, .25, 1.25], 24) + rng.normal(0, .1, 96)
    z = rng.normal(size=96)
    signal = 1.7*x - .4*z + rng.normal(0, .5, 96)
    labels = {"strings": ("001", "010"), "integers": (2, 10), "booleans": (False, True)}
    y = signal if kind == "regression" else [labels[kind][int(value > 0)] for value in signal]
    frame = pd.DataFrame({"x": x, "z": z, "label": y,
                          "cohort": ["audit", "other", None]*32, "row_id": np.arange(96)})
    frame.index = pd.Index(np.arange(96)*7 % 19, name="duplicate_index")
    return frame


def _session(kind):
    frame = _frame(kind)
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "label": "target",
                                             "cohort": "group", "row_id": "id"})
    session.inject_split(train_indices=list(range(64)), validation_indices=list(range(64, 80)),
                         test_indices=list(range(80, 96)))
    return session


def _rule(kind, column=None):
    op, value = {"strings": ("==", "audit"), "integers": ("not_in", ["other"]),
                 "booleans": ("notna", None), "regression": ("isna", None)}[kind]
    if column == "label":
        op, value = "notna", None
    return {"rule_id": "context", "if": [{"column": column or "cohort", "op": op, "value": value}],
            "then": {"strings": "010", "integers": 10, "booleans": True, "regression": 4.5}[kind],
            "kind": "regression" if kind == "regression" else "classification",
            "hardness": "hard", "priority": 100}


def _fit(session, kind, *, mode="constraint_overlay", base="linear", source="declared", rules=None, **options):
    task = "regression" if kind == "regression" else "classification"
    if rules is None:
        rules = [_rule(kind)]
    if mode == "symbolic":
        return session.symbolic.fit(source="declared", task=task, rules=rules, **options)
    if base == "linear":
        base = "ridge" if kind == "regression" else "logistic_regression"
    return session.symbolic.fit_neuro(mode=mode, base_estimator=base, task=task,
        rule_source=source, rules=rules, max_depth=2, min_samples_leaf=4,
        max_rules=12, prefer_reduce_components=False, random_state=5, **options)


def _matches(row, rule):
    for p in rule.antecedents:
        value = row[p.column]
        if p.op == "isna":
            match = pd.isna(value)
        elif p.op == "notna":
            match = not pd.isna(value)
        elif p.op == "in":
            match = value in p.value
        elif p.op == "not_in":
            match = value not in p.value
        elif pd.isna(value):
            match = False
        elif p.op == "==":
            match = value == p.value
        elif p.op == "!=":
            match = value != p.value
        elif p.op == "<=":
            match = float(value) <= float(p.value)
        elif p.op == "<":
            match = float(value) < float(p.value)
        elif p.op == ">":
            match = float(value) > float(p.value)
        else:
            match = float(value) >= float(p.value)
        if not match:
            return False
    return True


def _leaf_value(tree, values, classification):
    node = 0
    structure = tree.tree_
    while structure.children_left[node] != structure.children_right[node]:
        value = float(np.float32(values[structure.feature[node]]))
        node = (structure.children_left[node] if value <= structure.threshold[node]
                else structure.children_right[node])
    value = structure.value[node].ravel()
    return value / value.sum() if classification else float(value[0])


def _neural(plan, values):
    model = plan.estimator_
    classification = plan.task == "classification"
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_).ravel()
        value = sum(float(a)*float(b) for a, b in zip(values, coefficients)) + float(np.asarray(model.intercept_).ravel()[0])
        code = int(value > 0)
    else:
        trees = list(model.estimators_) if hasattr(model, "estimators_") else [model]
        leaves = [_leaf_value(tree, values, classification) for tree in trees]
        if classification:
            probabilities = [sum(float(leaf[j]) for leaf in leaves)/len(leaves) for j in range(len(plan.classes_))]
            code = max(range(len(probabilities)), key=lambda index: probabilities[index])
        else:
            value = sum(leaves)/len(leaves)
    return plan.classes_[code] if classification else value


def _check(session, kind, partition):
    frame = session.partition(partition)
    plan = session.neuro_symbolic_plan
    before = session.to_pandas()
    fingerprint = joblib.hash(plan)
    rules = plan.knowledge_base.rules
    expected, neural, chosen, fired, repairs = [], [], [], [], []
    for row in frame.to_dict("records"):
        active = [rule for rule in rules if _matches(row, rule)]
        values = [row[column] for column in plan.columns]
        if plan.mode == "rules_as_features":
            values.extend(float(_matches(row, rule)) for rule in rules)
        base = _neural(plan, values)
        prediction = active[0].consequent if active and plan.mode != "rules_as_features" else base
        expected.append(prediction)
        neural.append(base)
        chosen.append(active[0].rule_id if active else None)
        fired.append(tuple(rule.rule_id for rule in (active if plan.mode == "rules_as_features" else active[:1])))
        repairs.append(bool(active and plan.mode == "constraint_repair" and prediction != base))
    result = session.symbolic.predict_neuro(partition=partition)
    if kind == "regression":
        np.testing.assert_allclose(result.predictions, expected, atol=1e-12)
        np.testing.assert_allclose(result.neural_predictions, neural, atol=1e-12)
    else:
        assert result.predictions == tuple(expected) and result.neural_predictions == tuple(neural)
    assert [trace.row_index for trace in result.traces] == frame.index.tolist()
    assert [trace.chosen_rule_id for trace in result.traces] == chosen
    assert [trace.fired_rule_ids for trace in result.traces] == fired
    assert result.n_repaired == sum(repairs)
    report = session.symbolic.evaluate_neuro(partition=partition)
    if kind == "regression":
        residual = [float(a)-float(b) for a, b in zip(frame.label, expected)]
        assert report.metrics["mae"] == pytest.approx(sum(abs(v) for v in residual)/len(residual))
        assert report.metrics["rmse"] == pytest.approx((sum(v*v for v in residual)/len(residual))**.5)
    else:
        assert report.metrics["accuracy"] == sum(a == b for a, b in zip(frame.label, expected))/len(frame)
    assert report.repair_rate == pytest.approx(sum(repairs)/len(frame))
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    json.dumps(report.to_dict(), allow_nan=False)
    return result


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("base", ["linear", "decision_tree", "random_forest"])
@pytest.mark.parametrize("source", ["declared", "decision_tree", "decision_list"])
@pytest.mark.parametrize("mode", MODES)
def test_context_and_induced_hybrids_match_scalar_models_and_rule_priority(kind, base, source, mode, tmp_path):
    session = _session(kind)
    fit = _fit(session, kind, mode=mode, base=base, source=source)
    plan = session.neuro_symbolic_plan
    assert plan.knowledge_base.rules[0].rule_id == "context"
    assert fit.n_rules == len(plan.knowledge_base.rules)
    assert fit.rule_provenance == ("declared" if source == "declared" else "mixed_declared_and_induced")
    _check(session, kind, "validation")
    expected = _check(session, kind, "test")
    path = session.symbolic.save_bundle(tmp_path / "owned")
    restored = _session(kind)
    restored.symbolic.load_bundle(path, trusted=True)
    assert _check(restored, kind, "test").predictions == expected.predictions


PATHS = [("symbolic", "declared")] + [(mode, source) for mode in MODES
                                          for source in ["declared", "decision_tree", "decision_list"]]


def _snapshot(session):
    return {"history": copy.deepcopy(session._history),
            "symbolic": session.symbolic_plan, "neuro": session.neuro_symbolic_plan,
            "symbolic_fit": session._symbolic_fit_result, "neuro_fit": session._neuro_symbolic_fit_result,
            "symbolic_pred": session._symbolic_predict_result, "neuro_pred": session._neuro_symbolic_predict_result,
            "hash": joblib.hash(session.neuro_symbolic_plan or session.symbolic_plan)}


def _assert_snapshot(session, before):
    assert session._history == before["history"]
    assert session.symbolic_plan is before["symbolic"] and session.neuro_symbolic_plan is before["neuro"]
    assert session._symbolic_fit_result is before["symbolic_fit"] and session._neuro_symbolic_fit_result is before["neuro_fit"]
    assert session._symbolic_predict_result is before["symbolic_pred"] and session._neuro_symbolic_predict_result is before["neuro_pred"]
    assert joblib.hash(session.neuro_symbolic_plan or session.symbolic_plan) == before["hash"]


def _ready(kind, mode):
    session = _session(kind)
    _fit(session, kind, mode=mode)
    (session.symbolic.predict if mode == "symbolic" else session.symbolic.predict_neuro)(partition="test")
    return session


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode,source", PATHS)
def test_target_predicates_cannot_replace_a_working_fit(kind, mode, source):
    session = _ready(kind, mode)
    before = _snapshot(session)
    with pytest.raises(ValidationError, match="target|Target"):
        _fit(session, kind, mode=mode, source=source, rules=[_rule(kind, "label")])
    _assert_snapshot(session, before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("source", ["decision_tree", "decision_list"])
def test_merged_declared_rules_validate_columns_before_replacing_fit(kind, mode, source):
    session = _ready(kind, mode)
    before = _snapshot(session)
    with pytest.raises(ValidationError, match="missing_context"):
        _fit(session, kind, mode=mode, source=source, rules=[_rule(kind, "missing_context")])
    _assert_snapshot(session, before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["symbolic", *MODES])
@pytest.mark.parametrize("operation", ["predict", "evaluate"])
def test_supplied_target_dependent_knowledge_base_refuses_scoring(kind, mode, operation, tmp_path):
    session = _ready(kind, mode)
    plan = session.symbolic_plan if mode == "symbolic" else session.neuro_symbolic_plan
    # A real fitted estimator with supplied rule metadata, not a fake backend.
    plan.knowledge_base = parse_declared_rules([_rule(kind, "label")], provenance="declared")
    path = session.symbolic.save_bundle(tmp_path / "supplied")
    receiver = _session(kind)
    receiver.symbolic.load_bundle(path, trusted=True)
    before = _snapshot(receiver)
    name = operation if mode == "symbolic" else operation + "_neuro"
    with pytest.raises(ValidationError, match="target|Target"):
        getattr(receiver.symbolic, name)(partition="test")
    _assert_snapshot(receiver, before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("options,match", [
    ({"mode": "unknown"}, "mode"), ({"soft_strength": -1}, "soft_strength"),
    ({"soft_strength": np.nan}, "soft_strength"), ({"rules": []}, "requires rules"),
    ({"source": "unknown"}, "rule_source"), ({"base": "unknown"}, "base_estimator"),
])
def test_invalid_hybrid_protocol_preserves_fitted_plan_and_history(kind, options, match):
    session = _ready(kind, "constraint_overlay")
    before = _snapshot(session)
    with pytest.raises(ValidationError, match=match):
        _fit(session, kind, **options)
    _assert_snapshot(session, before)
