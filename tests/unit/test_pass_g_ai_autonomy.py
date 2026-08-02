"""Pass G autonomy mode: explicit opt-in, allowlist, MockProvider multi-step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.autonomous import DEFAULT_AUTONOMY_ALLOWLIST
from buildml.ai.results import PlanResult, PlanStep
from buildml.ai.tools import registered_tool_names
from buildml.core.errors import ValidationError


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": np.asarray([0, 1] * (n // 2), dtype=np.int64),
        }
    )


def _step(operation: str, *, parameters: dict | None = None) -> PlanStep:
    return PlanStep(
        operation=operation,
        description=f"run {operation}",
        rationale="test",
        prerequisites=(),
        expected_changes=(),
        parameters=parameters or {},
    )


def test_registry_includes_pass_g_tools() -> None:
    names = set(registered_tool_names())
    for required in (
        "make_multimodal_torch_loaders",
        "make_image_multimodal_torch_loaders",
        "search_torch",
        "nested_cv_torch",
        "export_torch",
    ):
        assert required in names


def test_autonomy_requires_explicit_confirm() -> None:
    session = Session.ingest(_frame()).ai_configure(provider="mock")
    with pytest.raises(ValidationError, match="confirm_autonomy"):
        session.ai_run_autonomous("describe the dataset")


def test_autonomy_executes_allowlisted_plan_with_mock() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .ai_configure(provider="mock", egress_level="stats_only")
    )
    plan = PlanResult(
        goal="split then describe",
        steps=(
            _step("split", parameters={"test_size": 0.25, "random_state": 0}),
            _step("describe_dataset"),
            _step("workflow_status"),
        ),
        current_state_summary="roles set",
        assumptions=(),
        limitations=(),
        raw_response="mock plan",
    )
    result = session.ai_run_autonomous(
        "prepare a split",
        plan=plan,
        confirm_autonomy=True,
        max_steps=5,
        provider_plan=False,
    )
    assert result.completed_steps >= 1
    assert any(s.auto_confirmed and s.executed for s in result.steps)
    assert result.residual_risks
    status = session.ai_status()
    assert status["autonomy"]["autonomy_enabled_last_run"] is True
    assert session._split_plan is not None


def test_autonomy_blocks_sample_egress() -> None:
    session = Session.ingest(_frame()).ai_configure(provider="mock")
    from buildml.ai.privacy import EgressConfig, EgressLevel

    session._ai_egress_config = EgressConfig(level=EgressLevel.FULL_SAMPLE)
    plan = PlanResult(
        goal="x",
        steps=(_step("describe_dataset"),),
        current_state_summary="s",
        assumptions=(),
        limitations=(),
        raw_response="",
    )
    with pytest.raises(ValidationError, match="egress"):
        session.ai_run_autonomous(
            "x", plan=plan, confirm_autonomy=True, provider_plan=False
        )


def test_autonomy_skips_tools_outside_allowlist() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .ai_configure(provider="mock")
    )
    plan = PlanResult(
        goal="unknown",
        steps=(_step("totally_unknown_op"),),
        current_state_summary="s",
        assumptions=(),
        limitations=(),
        raw_response="",
    )
    result = session.ai_run_autonomous(
        "x",
        plan=plan,
        confirm_autonomy=True,
        provider_plan=False,
        tool_allowlist=("describe_dataset",),
    )
    assert result.steps[0].skipped
    assert "allowlist" in result.steps[0].skip_reason.lower() or "not in" in result.steps[
        0
    ].skip_reason.lower()


def test_default_allowlist_is_subset_of_registry() -> None:
    registry = set(registered_tool_names())
    builtins = {
        "describe_dataset",
        "ai_status",
        "workflow_status",
        "explain_operation",
        "eda_summary",
    }
    for name in DEFAULT_AUTONOMY_ALLOWLIST:
        assert name in registry or name in builtins


def test_pass_g_tools_have_executor_dispatch() -> None:
    """Registry tools must be wired in executor._dispatch_tool (Pass H/K)."""
    from buildml.ai.executor import execute_tool, propose_tool_execution
    from buildml.ai.tools import build_default_registry

    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .ai_configure(provider="mock")
    )
    registry = build_default_registry()
    # search_torch without space should fail inside Session, not "No dispatch handler".
    proposal = propose_tool_execution("search_torch", {}, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.error is not None
    assert "No dispatch handler" not in result.error
    err = result.error.lower()
    assert (
        "param_grid" in err
        or "param_distributions" in err
        or "torch" in err  # broken/missing local torch still proves dispatch wired
    )

    proposal = propose_tool_execution(
        "export_torch",
        {"path": "unused.pt"},
        registry,
    )
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.error is not None
    assert "No dispatch handler" not in result.error

    # Pass J/K: image multimodal tool must dispatch (missing image_column → Session error).
    proposal = propose_tool_execution(
        "make_image_multimodal_torch_loaders",
        {},
        registry,
    )
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.error is not None
    assert "No dispatch handler" not in result.error
    assert "image_column" in result.error.lower()


def test_search_torch_tool_schema_includes_param_grid() -> None:
    from buildml.ai.tools import build_default_registry

    registry = build_default_registry()
    spec = registry.get("search_torch")
    assert spec is not None
    props = spec.parameters.get("properties", {})
    assert "param_grid" in props
    nested = registry.get("nested_cv_torch")
    assert nested is not None
    assert "param_grid" in nested.parameters.get("properties", {})
