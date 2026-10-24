"""Causal assumptions require literal booleans across public entry points."""

import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.causal.types import CausalAssumptions
from buildml.core.errors import ValidationError

BASE = {'treatment': 't', 'outcome': 'y', 'confounders': ['x'],
        'acknowledge_unconfoundedness': True, 'acknowledge_positivity': True,
        'allow_empty_confounders': False}


@pytest.mark.parametrize('field', ['acknowledge_unconfoundedness', 'acknowledge_positivity',
                                  'allow_empty_confounders'])
@pytest.mark.parametrize('value', ['false', 1, 0, None, [], {}])
@pytest.mark.parametrize('surface', ['instance', 'mapping', 'session', 'agent'])
def test_causal_flags_cannot_be_acknowledged_by_truthiness(field, value, surface):
    arguments = {**BASE, field: value}
    if surface in ('instance', 'mapping'):
        with pytest.raises(ValidationError):
            assumptions = (CausalAssumptions(**arguments) if surface == 'instance'
                           else CausalAssumptions.from_mapping(arguments))
            assumptions.validate()
        return
    s = Session.ingest(pd.DataFrame({'x': [1., 2., 3.], 't': [0, 1, 0], 'y': [2., 3., 4.]}))
    s.causal.declare_assumptions(**BASE)
    original = s._causal_assumptions
    before, history = s.to_pandas().copy(deep=True), list(s._history)
    if surface == 'session':
        with pytest.raises(ValidationError):
            s.causal.declare_assumptions(**arguments)
    else:
        registry = build_default_registry()
        result = execute_tool(s, propose_tool_execution('declare_causal_assumptions', arguments, registry),
                              confirmed=True, registry=registry)
        assert not result.executed and result.error and result.state_changes == ()
    assert s._causal_assumptions is original and s._history == history
    pd.testing.assert_frame_equal(s.to_pandas(), before)


def test_boolean_empty_confounder_waiver_remains_explicit_and_roundtrips():
    arguments = {**BASE, 'confounders': [], 'allow_empty_confounders': True}
    assumptions = CausalAssumptions.from_mapping(arguments)
    assumptions.validate()
    assert assumptions.to_dict()['allow_empty_confounders'] is True
    restored = CausalAssumptions.from_mapping(assumptions.to_dict())
    restored.validate()
    assert restored == assumptions
    with pytest.raises(ValidationError, match='confounders is empty'):
        CausalAssumptions.from_mapping({**arguments, 'allow_empty_confounders': False}).validate()
