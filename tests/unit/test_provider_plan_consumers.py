"""Scripted provider plans pass through real Session planning and execution."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.advisor import run_advisor
from buildml.ai.provider import MockProvider
from buildml.ai.tools import build_default_registry
from buildml.data.splits import frame_for_partition


def configured_session():
    x = np.arange(40, dtype=float)
    frame = pd.DataFrame({"x": x, "y": 2.7 * x - 5 + np.cos(x)})
    return (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
            .ai.configure(provider="mock"))


def plan_data():
    return {
        "steps": [
            {"operation": "split", "description": "Reserve a quarter", "rationale": "Hold out rows",
             "prerequisites": ["roles assigned"], "expected_changes": ["30 train, 10 test"],
             "evidence": ["40 rows"], "warnings": ["No population guarantee"],
             "parameters": {"test_size": .25, "random_state": 17}},
            {"operation": "fit", "description": "Fit OLS", "rationale": "Numeric target",
             "parameters": {"estimator": "LinearRegression", "task": "regression"}},
        ],
        "current_state_summary": "40 rows with roles; no model fitted",
        "assumptions": ["Rows can be shuffled"], "limitations": ["Scripted protocol check"],
        "alternatives": ["Chronological split for ordered data"],
    }


MALFORMED = [("root", value) for value in [None, [], "plan", 3, False]]
MALFORMED += [("steps", value) for value in [None, {}, "split", 3, False]]
MALFORMED += [("step", value) for value in [None, [], "split", 3, False]]
MALFORMED += [("operation", value) for value in [None, [], {}, 3, False, "", "  "]]
MALFORMED += [(key, value) for key in ["description", "rationale", "current_state_summary"]
              for value in [None, 3, {}]]
MALFORMED += [(key, value) for key in ["prerequisites", "expected_changes", "evidence", "warnings",
                                     "assumptions", "limitations", "alternatives"]
              for value in ["not a list", {"key": "value"}, 3, ["text", False]]]
MALFORMED += [("parameters", value) for value in [None, [], "arguments", 3, False]]


@pytest.mark.parametrize("field,value", MALFORMED)
def test_malformed_provider_plan_fails_as_a_whole_and_retains_usage(field, value):
    session = configured_session()
    data = plan_data()
    if field == "root":
        data = value
    elif field == "step":
        data["steps"].append(value)
    elif field in {"steps", "current_state_summary", "assumptions", "limitations", "alternatives"}:
        data[field] = value
    else:
        data["steps"][1][field] = value
    raw = json.dumps(data)
    session._ai_provider.queue_responses([raw])
    original = session.dataset.frame.copy(deep=True)

    result = session.ai.plan("Fit a model")

    assert result.steps == ()
    assert result.current_state_summary == "Failed to parse plan response."
    assert result.limitations and "schema" in result.limitations[0]
    assert result.raw_response == raw
    assert result.usage["total_tokens"] == 150
    assert session.ai.status()["budget"]["tokens_used"] == 150
    assert session._ai_plan_result is result
    assert session._split_plan is None and session._fit_result is None
    pd.testing.assert_frame_equal(session.dataset.frame, original)
    assert session._ai_transcript.entries[-1].entry_type == "egress"
    assert session.history[-1]["action"] == "ai_plan"
    assert result.egress_manifest.level.value == "stats_only"
    assert result.egress_manifest.rows_sent == 0
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("raw", ['{"steps":', 'not JSON'])
def test_invalid_json_retains_existing_failure_result_contract(raw):
    session = configured_session()
    session._ai_provider.queue_responses([raw])
    result = session.ai.plan("Inspect")
    assert result.steps == ()
    assert result.limitations == ("Response was not valid JSON.",)
    assert result.usage["total_tokens"] == 150
    assert result.raw_response == raw


@pytest.mark.parametrize("autonomous", [False, True])
def test_provider_parameters_reach_real_fit_and_match_independent_ols(autonomous):
    session = configured_session()
    data = plan_data()
    raw = json.dumps(data)
    session._ai_provider.queue_responses([raw])
    original = session.dataset.frame.copy(deep=True)
    if autonomous:
        result = session.ai.run_autonomous("Fit the specified model", confirm_autonomy=True,
                                           max_steps=2, tool_allowlist=("split", "fit"))
        plan = result.plan
    else:
        plan = session.ai.plan("Fit the specified model")
        assert session._split_plan is None and session._fit_result is None
        preview = session.ai.run_plan(plan)
        assert preview.completed_steps == 0 and preview.requires_confirmation_at == 0
        assert session._split_plan is None and session._fit_result is None
        result = session.ai.run_plan(plan, confirmations={0: True, 1: True})

    assert [step.parameters for step in plan.steps] == [step["parameters"] for step in data["steps"]]
    assert plan.steps[0].prerequisites == ("roles assigned",)
    assert plan.steps[0].evidence == ("40 rows",)
    assert plan.steps[0].warnings == ("No population guarantee",)
    assert plan.assumptions == tuple(data["assumptions"])
    assert plan.limitations == tuple(data["limitations"])
    assert plan.alternatives == tuple(data["alternatives"])
    assert plan.raw_response == raw
    assert result.completed_steps == 2
    executions = result.steps if autonomous else result.step_executions
    assert all(step.executed for step in executions)
    permutation = np.random.RandomState(17).permutation(40)
    np.testing.assert_array_equal(session.split_plan.test_indices, permutation[:10])
    np.testing.assert_array_equal(session.split_plan.train_indices, permutation[10:])
    train = original.iloc[permutation[10:]]
    test = frame_for_partition(session.dataset, session.split_plan, "test")
    design = np.column_stack([np.ones(len(train)), train["x"]])
    weights = np.linalg.lstsq(design, train["y"], rcond=None)[0]
    expected = np.column_stack([np.ones(len(test)), test["x"]]) @ weights
    actual = session._fit_result.estimator.predict(test[["x"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(session.dataset.frame, original)
    assert session.ai.status()["budget"]["tokens_used"] == 150
    assert len(session._ai_provider.calls) == 1
    request = session._ai_provider.calls[0]["messages"]
    assert '"parameters"' in request[0]["content"]
    assert "[UNTRUSTED DATA FROM USER]" in request[1]["content"]


@pytest.mark.parametrize("data", [{}, {"steps": []}])
def test_valid_empty_plan_is_distinct_from_parsing_failure(data):
    session = configured_session()
    session._ai_provider.queue_responses([json.dumps(data)])
    result = session.ai.plan("Nothing to do")
    assert result.steps == () and result.limitations == ()
    assert result.current_state_summary == ""
    assert result.usage["total_tokens"] == 150


def test_structurally_valid_plan_still_requires_tool_arguments_before_execution():
    session = configured_session()
    session._ai_provider.queue_responses([json.dumps({
        "steps": [{"operation": "fit", "parameters": {"task": "regression"}}],
    })])
    plan = session.ai.plan("Fit without the required estimator")
    assert plan.steps[0].parameters == {"task": "regression"}
    result = session.ai.run_plan(plan, confirmations={0: True})
    assert result.completed_steps == 0
    assert not result.step_executions[0].executed
    assert "estimator" in result.step_executions[0].error
    assert session._split_plan is None and session._fit_result is None


def test_unknown_provider_operation_is_reported_without_becoming_a_tool():
    session = configured_session()
    session._ai_provider.queue_responses([json.dumps({
        "steps": [{"operation": "unregistered_operation", "parameters": {"x": 3}}],
    })])
    plan = session.ai.plan("An unavailable operation")
    result = session.ai.run_plan(plan, confirmations={0: True})
    assert result.completed_steps == 0
    assert result.step_executions[0].skipped
    assert "not in tool registry" in result.step_executions[0].skip_reason
    assert session._split_plan is None and session._fit_result is None


@pytest.mark.parametrize("optional_nulls", [False, True])
def test_minimal_valid_steps_keep_optional_defaults(optional_nulls):
    session = configured_session()
    data = {"steps": [{"operation": "describe_dataset"}]}
    if optional_nulls:
        for field in ["assumptions", "limitations", "alternatives"]:
            data[field] = None
        for field in ["prerequisites", "expected_changes", "evidence", "warnings"]:
            data["steps"][0][field] = None
    session._ai_provider.queue_responses([json.dumps(data)])
    plan = session.ai.plan("Describe")
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert (step.description, step.rationale, step.parameters) == ("", "", {})
    assert (step.prerequisites, step.expected_changes, step.evidence, step.warnings) == ((), (), (), ())
    result = session.ai.run_plan(plan)
    assert result.completed_steps == 1
    assert session._split_plan is None and session._fit_result is None


@pytest.mark.parametrize("tool,args,accepted", [
    ("describe_dataset", {}, True), ("workflow_status", {}, True),
    ("drop_columns", {"columns": ["x"]}, False),
    ("fit_classical", {"estimator": "LinearRegression"}, False),
    ("unregistered_operation", {}, False),
])
@pytest.mark.parametrize("rounds", [1, 2])
def test_advisor_tool_rounds_preserve_state_permissions_and_accounting(tool, args, accepted, rounds):
    session = configured_session()
    original = session.dataset.frame.copy(deep=True)
    history = copy.deepcopy(session.history)
    provider = MockProvider(default_response="40 rows; consider a held-out evaluation.")
    provider.set_next_tool_call(tool, args)
    result = run_advisor(session, "Inspect this data", provider,
                         registry=build_default_registry(), max_iterations=rounds)
    assert result.usage == {"prompt_tokens": 100 * rounds,
                            "completion_tokens": 50 * rounds, "total_tokens": 150 * rounds}
    assert len(result.tool_calls_made) == int(accepted)
    if rounds == 1:
        assert "Maximum iterations reached" in result.answer
        assert result.limitations == ("Max iterations reached.",)
    else:
        assert "40 rows" in result.answer
        messages = provider.calls[1]["messages"]
        assistant = [message for message in messages if message["role"] == "assistant"][-1]
        reply = [message for message in messages if message["role"] == "tool"][-1]
        assert reply["name"] == tool
        assert reply["tool_call_id"] == assistant["tool_calls"][0]["call_id"]
        assert ("not available in advisor mode" in reply["content"]) is (not accepted)
    advertised = {entry["function"]["name"] for entry in provider.calls[0]["tools"]}
    assert "describe_dataset" in advertised
    assert "fit_classical" not in advertised and "drop_columns" not in advertised
    assert session._split_plan is None and session._fit_result is None
    assert session.history == history
    pd.testing.assert_frame_equal(session.dataset.frame, original)
