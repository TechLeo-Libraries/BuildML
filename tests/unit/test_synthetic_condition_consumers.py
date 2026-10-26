"""Requested synthetic populations survive real AI and frozen-plan consumers."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.data.splits import frame_for_partition


def _session(method="gaussian_copula"):
    x = np.arange(160, dtype=float)
    frame = pd.DataFrame({"x": x, "category": ["wanted", "other"] * 80,
                          "y": 1.3 * x + np.sin(x)})
    session = (Session.ingest(frame)
               .set_roles({"x": "feature", "category": "ignore", "y": "target"})
               .split(test_size=.2, validation_size=.2, random_state=17)
               .ai.configure(provider="mock"))
    session.fit(Ridge(alpha=.4), task="regression")
    session.synthetic.fit(method=method, columns=["x", "category", "y"], random_state=3)
    return session


def _sample(session, route, arguments):
    if route == "direct":
        try:
            return session.synthetic.sample(**arguments), None
        except ValidationError as exc:
            return None, str(exc)
    if route == "executor":
        registry = build_default_registry()
        proposal = propose_tool_execution("sample_synthetic", arguments, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)
        return result.result, result.error
    session._ai_provider.queue_responses([json.dumps({"steps": [
        {"operation": "sample_synthetic", "parameters": arguments},
    ]})])
    if route == "autonomous":
        result = session.ai.run_autonomous(
            "Sample precisely the specified population", confirm_autonomy=True,
            max_steps=1, tool_allowlist=("sample_synthetic",),
        )
        assert json.dumps(result.steps[0].arguments, sort_keys=True) == json.dumps(arguments, sort_keys=True)
        error = result.steps[0].error
    else:
        plan = session.ai.plan("Sample precisely the specified population")
        assert json.dumps(plan.steps[0].parameters, sort_keys=True) == json.dumps(arguments, sort_keys=True)
        preview = session.ai.run_plan(plan)
        assert preview.completed_steps == 0 and preview.requires_confirmation_at == 0
        result = session.ai.run_plan(plan, confirmations={0: True})
        error = result.step_executions[0].error
    assert result.completed_steps == int(error is None)
    return session.synthetic_sample_result if error is None else None, error


@pytest.mark.parametrize("route", ["direct", "executor", "confirmed_plan", "autonomous"])
@pytest.mark.parametrize("merge_mode", ["none", "extend_train"])
@pytest.mark.parametrize("label", ["wanted", "other"])
def test_population_controls_holdouts_roles_and_refit_predictions(route, merge_mode, label):
    session = _session()
    before = session.dataset.frame.copy(deep=True)
    split = session.split_plan
    fitted = session.fit_result
    generator = session.synthesizer_plan
    arguments = dict(n=32, random_state=11, condition={"category": label},
                     merge_mode=merge_mode, provenance_column="generated_here", validate=True)
    result, error = _sample(session, route, arguments)
    assert error is None, error
    assert result.n_rows == 32 and result.frame.category.eq(label).all()
    assert result.frame.x.between(before.x.min(), before.x.max()).all()
    assert session.synthesizer_plan is generator
    sample_history = [entry for entry in session.history if entry["action"] == "sample_synthetic"]
    assert sample_history[-1]["parameters"]["condition"] == {"category": label}
    if merge_mode == "none":
        assert not result.merged and session.split_plan is split
        assert session.fit_result is fitted
        pd.testing.assert_frame_equal(session.dataset.frame, before)
        return
    assert result.merged and result.provenance_column == "generated_here"
    assert session.fit_result is None
    assert len(session.split_plan.train_indices) == len(split.train_indices) + 32
    assert session.dataset.roles["generated_here"].value == "ignore"
    train = frame_for_partition(session.dataset, session.split_plan, "train")
    assert train.generated_here.sum() == 32
    pd.testing.assert_frame_equal(train.iloc[-32:].drop(columns="generated_here")
                                  .reset_index(drop=True), result.frame)
    for partition in ("validation", "test"):
        actual = frame_for_partition(session.dataset, session.split_plan, partition)
        assert not actual.generated_here.any()
        expected = before.iloc[list(getattr(split, f"{partition}_indices"))].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual.drop(columns="generated_here").reset_index(drop=True),
                                      expected)
    # Refit a real consumer on the augmented population; provenance stays excluded.
    session.fit(Ridge(alpha=.4), task="regression")
    test = frame_for_partition(session.dataset, session.split_plan, "test")
    centered_x = train.x.to_numpy() - train.x.mean()
    slope = centered_x @ (train.y.to_numpy() - train.y.mean()) / (centered_x @ centered_x + .4)
    expected_predictions = train.y.mean() + slope * (test.x.to_numpy() - train.x.mean())
    np.testing.assert_allclose(session.predict(partition="test"),
                               expected_predictions, atol=1e-10, rtol=1e-10)


BAD_CONDITIONS = [[], "wanted", 1, False, {"missing": "wanted"},
                  {"category": ["wanted"]}, {"category": {"value": "wanted"}},
                  {"category": "absent"}, {"category": float("inf")},
                  {"category": float("nan")}, {3: "wanted"}]


@pytest.mark.parametrize("route", ["direct", "executor", "confirmed_plan", "autonomous"])
@pytest.mark.parametrize("condition", BAD_CONDITIONS)
def test_bad_population_request_cannot_silently_merge_or_replace_prior_result(route, condition):
    session = _session()
    prior = session.synthetic.sample(n=3, random_state=6)
    before = session.dataset.frame.copy(deep=True)
    split, fitted, generator = session.split_plan, session.fit_result, session.synthesizer_plan
    history = copy.deepcopy(session.history)
    result, error = _sample(session, route, dict(n=12, condition=condition, random_state=11,
                                                merge_mode="extend_train"))
    assert result is None and error and "condition" in error.lower()
    assert session.split_plan is split and session.fit_result is fitted
    assert session.synthesizer_plan is generator and session.synthetic_sample_result is prior
    assert not any(entry["action"] == "sample_synthetic" for entry in session.history[len(history):])
    pd.testing.assert_frame_equal(session.dataset.frame, before)


@pytest.mark.parametrize("route", ["direct", "executor", "confirmed_plan", "autonomous"])
def test_unsupported_method_refuses_requested_condition_instead_of_unconditional_rows(route):
    session = _session("bootstrap")
    before = session.dataset.frame.copy(deep=True)
    result, error = _sample(session, route, dict(n=12, condition={"category": "wanted"},
                                                merge_mode="extend_train"))
    assert result is None and "gaussian_copula" in error
    pd.testing.assert_frame_equal(session.dataset.frame, before)


@pytest.mark.parametrize("route", ["direct", "executor"])
@pytest.mark.parametrize("dtype", [object, "string", "category"])
@pytest.mark.parametrize("label", [None, "None", "<NA>"])
def test_null_and_literal_null_labels_are_distinct_populations(route, dtype, label, tmp_path):
    values = pd.Series([None, "None", "<NA>", "ordinary"] * 60, dtype=dtype)
    session = (Session.ingest(pd.DataFrame({"category": values}))
               .set_roles({"category": "feature"})
               .split(test_size=.2, random_state=17))
    session.synthetic.fit(method="gaussian_copula", random_state=3)
    path = tmp_path / "frozen"
    session.synthetic.save_bundle(path)
    session.synthetic.load_bundle(path, trusted=True)
    sample, error = _sample(session, route, dict(n=20, condition={"category": label}, random_state=11))
    assert error is None
    if label is None:
        assert sample.frame.category.isna().all()
    else:
        assert sample.frame.category.notna().all() and sample.frame.category.eq(label).all()
    assert sample.frame.category.dtype == values.dtype


@pytest.mark.parametrize("values,wrong", [([0, 1] * 80, "1"),
                                         ([False, True] * 80, "True")])
def test_text_condition_does_not_select_a_numeric_or_boolean_label(values, wrong):
    session = (Session.ingest(pd.DataFrame({"category": values}))
               .set_roles({"category": "feature"})
               .split(test_size=.2, random_state=17))
    session.synthetic.fit(method="gaussian_copula")
    with pytest.raises(ValidationError, match="condition"):
        session.synthetic.sample(n=4, condition={"category": wrong}, random_state=11)


@pytest.mark.parametrize("route", ["direct", "executor", "confirmed_plan", "autonomous"])
@pytest.mark.parametrize("merge_mode", ["none", "extend_train"])
def test_joint_boolean_and_integer_conditions_require_every_predicate(route, merge_mode):
    frame = pd.DataFrame({"choice": [0, 1, 0, 1] * 60,
                          "flag": [False, False, True, True] * 60})
    session = (Session.ingest(frame).set_roles({"choice": "feature", "flag": "feature"})
               .split(test_size=.2, validation_size=.2, random_state=17)
               .ai.configure(provider="mock"))
    session.synthetic.fit(method="gaussian_copula", random_state=3)
    sample, error = _sample(session, route, dict(n=32, condition={"choice": 1, "flag": True},
                                                merge_mode=merge_mode, random_state=11))
    assert error is None
    assert sample.n_rows == 32
    assert sample.frame.choice.eq(1).all() and sample.frame.flag.eq(True).all()
    assert sample.frame.choice.dtype == frame.choice.dtype
    assert sample.frame.flag.dtype == frame.flag.dtype


@pytest.mark.parametrize("route", ["direct", "executor", "confirmed_plan", "autonomous"])
@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
@pytest.mark.parametrize("condition", [None, {}])
def test_optional_empty_condition_retains_seed_and_unconditional_population(route, method, condition):
    session = _session(method)
    train = frame_for_partition(session.dataset, session.split_plan, "train").reset_index(drop=True)
    expected = session.synthetic.sample(n=24, random_state=11).frame
    if method == "bootstrap":
        indices = np.random.default_rng(11).integers(0, len(train), size=24)
        pd.testing.assert_frame_equal(expected, train.iloc[indices].reset_index(drop=True))
    sample, error = _sample(session, route, dict(n=24, condition=condition, random_state=11))
    assert error is None
    pd.testing.assert_frame_equal(sample.frame, expected)


def test_provider_advertises_condition_values_without_changing_confirmation_policy():
    spec = build_default_registry().get("sample_synthetic")
    schema = spec.to_openai_tool()["function"]["parameters"]["properties"]["condition"]
    assert "object" in schema["type"]
    assert spec.confirm_policy.value == "confirm" and not spec.read_only
    proposal = propose_tool_execution("sample_synthetic", {"condition": {"category": "wanted"}},
                                      build_default_registry())
    assert proposal.requires_confirmation
    assert any("wanted" in change for change in proposal.expected_changes)
