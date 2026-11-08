"""Real hybrid predictions and traces agree with independent ordered rule arithmetic."""

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.fixture(scope="module", params=["regression", "strings", "integers", "booleans"])
def data(request):
    rng = np.random.default_rng(333)
    x = np.tile([-1.5, -0.25, 0.0, 0.25, 1.0, 1.5], 12)
    z = rng.normal(size=72)
    signal = 1.7 * x - 0.3 * z + rng.normal(0, 0.2, 72)
    kind = request.param
    y = {
        "regression": signal,
        "strings": np.where(signal > 0, "yes", "no"),
        "integers": np.where(signal > 0, 7, 2),
        "booleans": signal > 0,
    }[kind]
    return pd.DataFrame({"x": x, "z": z, "target": y}), kind


def ready(frame):
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"})
    s.inject_split(
        train_indices=list(range(36)),
        validation_indices=list(range(36, 54)),
        test_indices=list(range(54, 72)),
    )
    return s


@pytest.mark.parametrize("mode", ["constraint_overlay", "constraint_repair"])
@pytest.mark.parametrize("strength", [0.0, np.nextafter(0.5, 0), 0.5, 1.0])
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_ordered_overlay_blends_thresholds_traces_and_bundle(
    data, mode, strength, partition, tmp_path
):
    frame, kind = data
    s = ready(frame)
    regression = kind == "regression"
    task = "regression" if regression else "classification"
    positive = {"regression": 10.0, "strings": "yes", "integers": 7, "booleans": True}[kind]
    negative = {"regression": -10.0, "strings": "no", "integers": 2, "booleans": False}[kind]
    rules = [
        {
            "rule_id": "soft",
            "if": [{"column": "x", "op": ">=", "value": 0.0}],
            "then": positive,
            "priority": 30,
            "hardness": "soft",
            "strength": 1.0,
            "kind": task,
        },
        {
            "rule_id": "hard",
            "if": [{"column": "x", "op": ">=", "value": 1.0}],
            "then": negative,
            "priority": 20,
            "hardness": "hard",
            "kind": task,
        },
        {
            "rule_id": "constraint",
            "if": [{"column": "x", "op": "<", "value": -0.5}],
            "then": negative,
            "priority": 10,
            "hardness": "hard",
            "kind": "constraint",
        },
    ]
    s.symbolic.fit_neuro(
        mode=mode,
        base_estimator="ridge" if regression else "logistic_regression",
        task=task,
        rule_source="declared",
        rules=rules,
        soft_strength=strength,
    )
    plan = s._neuro_symbolic_plan
    fingerprint = joblib.hash(plan)
    rows = s.partition(partition)
    model = plan.estimator_
    linear = (
        rows[["x", "z"]].to_numpy() @ np.asarray(model.coef_).reshape(-1)
        + np.asarray(model.intercept_).reshape(-1)[0]
    )
    neural = (
        linear if regression else np.asarray(plan.label_encoder_.classes_)[(linear > 0).astype(int)]
    )
    expected = []
    fired = []
    chosen = []
    repaired = []
    for value, base in zip(rows.x, neural, strict=True):
        prediction = base
        ids = []
        selected = None
        repair = False
        if value >= 0:
            ids.append("soft")
            if regression:
                prediction = (1 - strength) * float(base) + strength * positive
                selected = "soft"
            elif strength >= 0.5:
                prediction = positive
                selected = "soft"
        if selected is None and value >= 1:
            ids.append("hard")
            selected = "hard"
            prediction = negative
            repair = mode == "constraint_repair" and base != negative
        if selected is None and value < -0.5:
            ids.append("constraint")
            selected = "constraint"
            prediction = negative
            repair = base != negative
        expected.append(prediction)
        fired.append(tuple(ids))
        chosen.append(selected)
        repaired.append(bool(repair))
    result = s.symbolic.predict_neuro(partition=partition, return_traces=True)
    np.testing.assert_allclose(
        result.neural_predictions, neural
    ) if regression else np.testing.assert_array_equal(result.neural_predictions, neural)
    np.testing.assert_allclose(
        result.predictions, expected
    ) if regression else np.testing.assert_array_equal(result.predictions, expected)
    assert result.n_repaired == sum(repaired)
    assert [t.row_index for t in result.traces] == list(rows.index)
    assert [t.fired_rule_ids for t in result.traces] == fired
    assert [t.chosen_rule_id for t in result.traces] == chosen
    assert [t.repaired for t in result.traces] == repaired
    for trace, value, base in zip(result.traces, expected, neural, strict=True):
        assert trace.prediction == pytest.approx(value) if regression else trace.prediction == value
        assert (
            trace.neural_prediction == pytest.approx(base)
            if regression
            else trace.neural_prediction == base
        )
    no_trace = s.symbolic.predict_neuro(partition=partition, return_traces=False)
    assert not no_trace.traces
    assert no_trace.predictions == result.predictions and no_trace.n_repaired == result.n_repaired
    assert joblib.hash(plan) == fingerprint
    path = s.symbolic.save_bundle(tmp_path / "hybrid")
    restored = ready(frame)
    restored.symbolic.load_bundle(path, trusted=True)
    again = restored.symbolic.predict_neuro(partition=partition)
    assert again.predictions == result.predictions and again.traces == result.traces
    pd.testing.assert_frame_equal(s.to_pandas(), frame)
