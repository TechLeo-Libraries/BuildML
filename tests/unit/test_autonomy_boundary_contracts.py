"""Autonomous execution refuses ambiguous permission and budget inputs."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.autonomous import AutonomyConfig, run_autonomous
from buildml.ai.results import PlanResult, PlanStep
from buildml.core.errors import ValidationError


def workflow():
    session = Session.ingest(pd.DataFrame({"x": range(20), "y": [0, 1] * 10})).set_roles({"x": "feature", "y": "target"}).ai.configure(provider="mock")
    step = PlanStep(operation="split", description="split", rationale="contract", prerequisites=(), expected_changes=(), parameters={"test_size": 0.25, "random_state": 0})
    plan = PlanResult(goal="split", steps=(step,), current_state_summary="roles", assumptions=(), limitations=(), raw_response="")
    return session, plan


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None])
@pytest.mark.parametrize("field", ["confirm_autonomy", "provider_plan", "allow_destructive"])
def test_session_rejects_ambiguous_permissions_before_mutation(field, value):
    session, plan = workflow()
    kwargs = {"confirm_autonomy": True, "provider_plan": False, field: value}
    with pytest.raises(ValidationError, match=field):
        session.ai.run_autonomous("split", plan=plan, **kwargs)
    assert session._split_plan is None
    assert getattr(session, "_ai_autonomy_result", None) is None


@pytest.mark.parametrize("value", [True, False, 1.5, float("nan"), float("inf"), "2", 0, -1])
def test_caps_require_positive_integers_before_mutation(value):
    session, plan = workflow()
    with pytest.raises(ValidationError, match="max_steps"):
        session.ai.run_autonomous("split", plan=plan, confirm_autonomy=True, provider_plan=False, max_steps=value)
    assert session._split_plan is None


@pytest.mark.parametrize("field", ["require_explicit", "stop_on_error"])
def test_direct_config_rejects_truthy_strings(field):
    session, plan = workflow()
    config = AutonomyConfig(**{field: "false"})
    with pytest.raises(ValidationError, match=field):
        run_autonomous(session, "split", plan=plan, config=config, confirm_autonomy=True, provider_plan=False)
    assert session._split_plan is None


def test_numpy_integer_budget_runs_real_split():
    session, plan = workflow()
    result = session.ai.run_autonomous("split", plan=plan, confirm_autonomy=True, provider_plan=False, max_steps=np.int64(1))
    assert result.completed_steps == 1
    assert session._split_plan is not None
