"""Every mutating tool must reject missing approval before interpreting its arguments."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import ToolRegistry, build_default_registry
from buildml.ai.types import ConfirmPolicy

REGISTRY = build_default_registry()
MUTATING_TOOLS = [tool.name for tool in REGISTRY.tools if not tool.read_only or tool.destructive]


@pytest.mark.parametrize("name", MUTATING_TOOLS)
def test_unapproved_mutating_tool_preserves_data_and_split(name):
    frame = pd.DataFrame({"x": range(30), "y": [0, 1] * 15})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    session.ai.configure(provider="mock")
    before = session.to_pandas().copy()
    proposal = propose_tool_execution(name, {}, REGISTRY)
    outcome = execute_tool(session, proposal, confirmed=False, registry=REGISTRY)
    assert not outcome.executed
    assert "confirm" in outcome.error.lower()
    assert outcome.result is None
    assert outcome.state_changes == ()
    assert session.split_plan is None
    pd.testing.assert_frame_equal(session.to_pandas(), before)


def test_new_stricter_policy_cannot_be_bypassed_by_an_old_proposal():
    session = Session.ingest(pd.DataFrame({"x": range(10)}))
    head = REGISTRY.get("head")
    automatic = ToolRegistry(tools=(head,))
    proposal = propose_tool_execution("head", {"n": 3}, automatic)
    assert not proposal.requires_confirmation
    stricter = ToolRegistry(tools=(replace(head, confirm_policy=ConfirmPolicy.CONFIRM),))
    outcome = execute_tool(session, proposal, confirmed=False, registry=stricter)
    assert not outcome.executed
    assert "current tool policy" in outcome.error.lower()


@pytest.mark.parametrize("confirmed", ["false", "true", 0, 1, None, [], {}])
def test_truthy_confirmation_values_cannot_authorize_a_split(confirmed):
    session = Session.ingest(pd.DataFrame({"x": range(30), "y": [0, 1] * 15}))
    proposal = propose_tool_execution("split", {"test_size": .2}, REGISTRY)
    outcome = execute_tool(session, proposal, confirmed=confirmed, registry=REGISTRY)
    assert not outcome.executed
    assert "confirmed must be a boolean" in outcome.error
    assert session.split_plan is None
