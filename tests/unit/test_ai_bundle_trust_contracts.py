"""Bundle restoration requires both a boolean trust decision and confirmation."""

import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.ai.types import ConfirmPolicy

BUNDLE_TOOLS = tuple(tool.name for tool in build_default_registry().tools
                     if tool.name.startswith('load_') and tool.name.endswith('_bundle'))


@pytest.mark.parametrize('name', BUNDLE_TOOLS)
def test_bundle_schema_exposes_explicit_trust_without_weakening_confirmation(name):
    registry = build_default_registry()
    spec = registry.get(name)
    assert spec.confirm_policy == ConfirmPolicy.CONFIRM
    assert not spec.read_only
    assert spec.parameters['properties']['trusted']['type'] == 'boolean'
    assert 'trusted' not in spec.parameters.get('required', [])


@pytest.mark.parametrize('name', BUNDLE_TOOLS)
@pytest.mark.parametrize('trusted', ['true', 'false', 1, None])
def test_malformed_trust_is_refused_before_any_bundle_path_is_opened(name, trusted, tmp_path):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, {'path': str(tmp_path / 'does-not-exist'),
                                             'trusted': trusted}, registry)
    result = execute_tool(Session(), proposal, confirmed=True, registry=registry)
    assert not result.executed
    assert 'trusted must be a JSON boolean' in result.error


@pytest.mark.parametrize('name', BUNDLE_TOOLS)
def test_trust_alone_does_not_authorize_execution(name, tmp_path):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, {'path': str(tmp_path / 'does-not-exist'),
                                             'trusted': True}, registry)
    assert any('executable model deserialization' in warning for warning in proposal.warnings)
    result = execute_tool(Session(), proposal, confirmed=False, registry=registry)
    assert not result.executed and 'confirm' in result.error.lower()
