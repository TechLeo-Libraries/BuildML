"""Model allocations consume costs and identifiers in candidate row order."""

import copy
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError

KINDS = ["strings", "integers", "booleans", "multiclass"]
METHODS = ["topk", "knapsack", "lp_allocate"]


def _session(kind="strings", index="duplicate"):
    rng = np.random.default_rng(1841)
    values = rng.normal(size=(48, 2))
    signal = values[:, 0]-.7*values[:, 1]+rng.normal(0, .25, 48)
    labels = {"strings": np.where(signal > 0, "010", "001"),
              "integers": np.where(signal > 0, 10, 2), "booleans": signal > 0,
              "multiclass": np.where(signal < -.4, "a", np.where(signal > .4, "c", "b"))}[kind]
    frame = pd.DataFrame({"x": values[:, 0], "z": values[:, 1], "target": labels,
                          "cost": pd.Series([0., 1., 2., 3., 4., 2., 1., 2.]*6, dtype=object),
                          "uid": [f"item-{i:03d}" for i in range(48)]})
    if index == "duplicate":
        frame.index = pd.Index(np.arange(48) % 4, name="duplicate_labels")
    elif index == "multi":
        frame.index = pd.MultiIndex.from_tuples([(f"group-{i % 2}", i % 3) for i in range(48)], names=["g", "n"])
    else:
        frame.index = pd.Index(100+np.arange(48)*3, name="unique_labels")
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target",
                                             "cost": "ignore", "uid": "id"})
    session.inject_split(train_indices=list(range(32)), validation_indices=list(range(32, 40)),
                         test_indices=list(range(40, 48)))
    return session.fit(LogisticRegression(C=.65, max_iter=500), task="classification")


def _arguments(method, source, use_cost=True, backend="native", floor=None):
    options = dict(method=method, backend=backend, partition="validation", score_source=source,
                   cost_column="cost" if use_cost else None, id_column="uid", min_score=floor)
    options.update(capacity=3) if method == "topk" else options.update(budget=5.)
    return options


def _scores(session, frame, source):
    model = session.fit_result.estimator
    logits = frame[["x", "z"]].to_numpy() @ model.coef_.T + model.intercept_
    if source == "model_decision_function":
        return logits[:, 0] if logits.shape[1] == 1 else logits.max(axis=1)
    if logits.shape[1] == 1:
        return 1/(1+np.exp(-logits[:, 0]))
    shifted = logits-logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)/np.exp(shifted).sum(axis=1, keepdims=True)
    return probabilities.max(axis=1)


def _oracle(values, costs, method, floor):
    eligible = [i for i, value in enumerate(values) if floor is None or value >= floor]
    fractions = np.zeros(len(values))
    if method == "topk":
        fractions[sorted(eligible, key=lambda i: (-values[i], i))[:3]] = 1
    elif method == "knapsack":
        feasible = []
        for bits in itertools.product([0, 1], repeat=len(eligible)):
            chosen = [i for i, bit in zip(eligible, bits) if bit]
            if sum(costs[i] for i in chosen) <= 5:
                feasible.append((sum(values[i] for i in chosen), chosen))
        _, chosen = max(feasible, key=lambda row: row[0])
        fractions[chosen] = 1
    else:
        for i in eligible:
            if costs[i] == 0 and values[i] > 0:
                fractions[i] = 1
        remaining = 5.
        paid = [i for i in eligible if costs[i] > 0 and values[i] > 0]
        for i in sorted(paid, key=lambda i: -values[i]/costs[i]):
            fractions[i] = min(1., remaining/costs[i])
            remaining -= fractions[i]*costs[i]
    return fractions


def _check(session, method, source, use_cost, floor=None):
    fingerprint, classifier = joblib.hash(session.decision_plan), joblib.hash(session.fit_result)
    original = session.to_pandas()
    for partition in ["validation", "test"]:
        frame = session.partition(partition)
        values = _scores(session, frame, source)
        costs = frame.cost.to_numpy(dtype=float) if use_cost else np.ones(len(frame))
        expected = _oracle(values, costs, method, floor)
        actual = session.decision.apply(partition=partition)
        vector = np.zeros(len(frame))
        for position, fraction, identifier, score in zip(actual.selected_indices, actual.fractions,
                                                         actual.selected_ids, actual.scores, strict=True):
            assert identifier == frame.uid.iloc[position]
            assert score == pytest.approx(values[position])
            vector[position] = fraction
        np.testing.assert_allclose(vector, expected, atol=2e-7, rtol=2e-7)
        assert actual.n_rows == len(frame)
        if session.decision_plan.config.get("backend") == "cvxpy":
            # An approximate LP may retain tiny positive fractions on other rows.
            assert actual.n_selected == len(actual.selected_indices)
        else:
            assert actual.n_selected == np.count_nonzero(expected > 1e-8)
        assert actual.selected_value == pytest.approx(float(values @ expected), abs=2e-7)
        assert actual.selected_cost == pytest.approx(float(costs @ expected), abs=2e-7)
        report = session.decision.evaluate(partition=partition)
        assert report.metrics["selected_value"] == pytest.approx(float(values @ expected), abs=2e-7)
        assert report.metrics["selected_cost"] == pytest.approx(float(costs @ expected), abs=2e-7)
        classes = session.fit_result.estimator.classes_
        if len(classes) == 2 and actual.n_selected:
            positive = np.mean(frame.target.iloc[list(actual.selected_indices)].to_numpy() == classes[1])
            assert report.metrics["selected_positive_rate"] == pytest.approx(positive)
        else:
            assert "selected_positive_rate" not in report.metrics
        assert any("in-sample" in warning for warning in report.warnings) is (partition == "validation")
        json.dumps(report.to_dict(), allow_nan=False)
    assert joblib.hash(session.decision_plan) == fingerprint and joblib.hash(session.fit_result) == classifier
    pd.testing.assert_frame_equal(session.to_pandas(), original)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("index", ["unique", "duplicate", "multi"])
@pytest.mark.parametrize("source", ["model_proba", "model_decision_function"])
@pytest.mark.parametrize("use_cost", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_model_allocations_keep_positional_costs_and_explicit_ids(kind, index, source, use_cost, method, tmp_path):
    session = _session(kind, index)
    session.decision.fit(**_arguments(method, source, use_cost))
    _check(session, method, source, use_cost)
    metadata = session.decision_plan.to_dict()
    bundle = session.decision.save_bundle(tmp_path / "owned")
    session.decision.load_bundle(bundle, trusted=True)
    assert session.decision_plan.to_dict() == metadata
    _check(session, method, source, use_cost)


@pytest.mark.parametrize("source", ["model_proba", "model_decision_function"])
@pytest.mark.parametrize("index", ["unique", "duplicate"])
@pytest.mark.parametrize("use_cost", [False, True])
@pytest.mark.parametrize("floor", [None, 100.])
def test_actual_cvxpy_matches_fractional_oracle_and_empty_eligibility(source, index, use_cost, floor):
    pytest.importorskip("cvxpy")
    session = _session("strings", index)
    session.decision.fit(**_arguments("lp_allocate", source, use_cost, backend="cvxpy", floor=floor))
    _check(session, "lp_allocate", source, use_cost, floor)


def _snapshot(session):
    return (copy.deepcopy(session.history), session.decision_plan, session.decision_fit_result,
            session.decision_apply_result, session.decision_eval_result,
            joblib.hash(session.decision_plan), joblib.hash(session.fit_result))


def _unchanged(session, snapshot):
    assert session.history == snapshot[0] and session.decision_plan is snapshot[1]
    assert session.decision_fit_result is snapshot[2] and session.decision_apply_result is snapshot[3]
    assert session.decision_eval_result is snapshot[4]
    assert joblib.hash(session.decision_plan) == snapshot[5] and joblib.hash(session.fit_result) == snapshot[6]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("operation", ["fit", "apply", "evaluate"])
@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -1., "not-a-cost"])
def test_invalid_model_costs_refuse_without_replacing_working_policy(method, operation, bad):
    session = _session(index="unique")
    options = _arguments(method, "model_proba")
    session.decision.fit(**options)
    session.decision.apply(partition="test")
    session.decision.evaluate(partition="test")
    position = 32 if operation == "fit" else 40
    session.dataset.frame.iloc[position, session.dataset.frame.columns.get_loc("cost")] = bad
    snapshot = _snapshot(session)
    with pytest.raises(ValidationError, match="cost"):
        if operation == "fit":
            session.decision.fit(**options)
        else:
            getattr(session.decision, operation)(partition="test")
    _unchanged(session, snapshot)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("operation", ["fit", "apply", "evaluate"])
@pytest.mark.parametrize("missing", ["cost", "uid", "uid_unit_cost"])
def test_explicit_missing_allocation_columns_refuse_atomically(method, operation, missing):
    use_cost = missing != "uid_unit_cost"
    session = _session(index="unique")
    options = _arguments(method, "model_proba", use_cost)
    session.decision.fit(**options)
    session.decision.apply(partition="test")
    session.decision.evaluate(partition="test")
    session.dataset.frame.drop(columns=["cost" if missing == "cost" else "uid"], inplace=True)
    snapshot = _snapshot(session)
    with pytest.raises(ValidationError, match="cost|uid"):
        if operation == "fit":
            session.decision.fit(**options)
        else:
            getattr(session.decision, operation)(partition="test")
    _unchanged(session, snapshot)
