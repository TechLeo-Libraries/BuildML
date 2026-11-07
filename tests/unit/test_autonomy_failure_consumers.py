"""Autonomous failures retain an audit trail and obey stop/continue policy."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.ai.autonomous import AutonomyConfig, run_autonomous
from buildml.ai.results import PlanResult, PlanStep
from buildml.data.splits import frame_for_partition


def step(operation, **parameters):
    return PlanStep(operation=operation, description=operation, rationale="consumer check",
                    prerequisites=(), expected_changes=(), parameters=parameters)


def workflow(steps):
    x = np.arange(40, dtype=float)
    frame = pd.DataFrame({"x": x, "y": 3 * x - 4 + np.sin(x)})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).ai.configure(provider="mock")
    plan = PlanResult(goal="fit with an audited failed first step", steps=tuple(steps),
                      current_state_summary="roles set", assumptions=())
    return session, plan


@pytest.mark.parametrize("stop_on_error", [False, True])
@pytest.mark.parametrize("failure", ["missing_argument", "invalid_split", "runtime"])
def test_failure_is_recorded_and_continuation_matches_direct_model(failure, stop_on_error):
    bad = {"missing_argument": step("fit"), "invalid_split": step("split", test_size="invalid"),
           "runtime": step("fit", estimator="LinearRegression", task="regression")}[failure]
    session, plan = workflow([bad, step("split", test_size=0.25, random_state=7),
                              step("fit", estimator="LinearRegression", task="regression")])
    result = run_autonomous(session, plan.goal, plan=plan, provider_plan=False,
                            confirm_autonomy=True,
                            config=AutonomyConfig(max_steps=3, stop_on_error=stop_on_error))
    assert result.steps[0].error
    assert not result.steps[0].executed
    assert result.steps[0].auto_confirmed is True
    errors = [entry for entry in session._ai_transcript.entries if entry.entry_type == "error"]
    assert len(errors) == 1
    assert errors[0].error == result.steps[0].error
    assert result.residual_risks
    if stop_on_error:
        assert result.completed_steps == 0
        assert result.stopped_at_step == 0
        assert len(result.steps) == 1
        assert session._split_plan is None
        assert session._fit_result is None
    else:
        assert result.completed_steps == 2
        assert result.stopped_at_step is None
        assert [record.executed for record in result.steps] == [False, True, True]
        train = frame_for_partition(session.dataset, session._split_plan, "train")
        test = frame_for_partition(session.dataset, session._split_plan, "test")
        expected = LinearRegression().fit(train[["x"]], train["y"]).predict(test[["x"]])
        np.testing.assert_allclose(session._fit_result.estimator.predict(test[["x"]]), expected, rtol=1e-12, atol=1e-12)


def test_skipped_step_consumes_iteration_budget_before_later_fit():
    session, plan = workflow([step("unregistered_operation"),
                              step("split", test_size=0.25, random_state=7),
                              step("fit", estimator="LinearRegression", task="regression")])
    session._ai_max_iterations = 2
    result = run_autonomous(session, plan.goal, plan=plan, provider_plan=False,
                            confirm_autonomy=True, config=AutonomyConfig(max_steps=3))
    assert result.steps[0].skipped
    assert result.completed_steps == 1
    assert result.stopped_at_step == 2
    assert session._split_plan is not None
    assert session._fit_result is None


def test_destructive_step_stops_before_mutating_real_dataset():
    session, plan = workflow([step("drop_columns", columns=["x"]),
                              step("split", test_size=0.25)])
    original = session.dataset.frame.copy(deep=True)
    result = run_autonomous(session, plan.goal, plan=plan, provider_plan=False,
                            confirm_autonomy=True, config=AutonomyConfig(
                                tool_allowlist=("drop_columns", "split"), max_steps=2))
    assert result.completed_steps == 0
    assert result.stopped_at_step == 0
    assert len(result.steps) == 1
    assert result.steps[0].skipped
    assert not result.steps[0].auto_confirmed
    assert "Destructive" in result.stop_reason
    assert session._split_plan is None
    pd.testing.assert_frame_equal(session.dataset.frame, original)
