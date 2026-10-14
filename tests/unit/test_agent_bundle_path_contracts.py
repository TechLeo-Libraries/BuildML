"""Missing bundle paths never cause agent writes or mutate an existing Session."""
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry

BUNDLE_TOOLS = [
    spec.name for spec in build_default_registry().tools
    if spec.name.startswith(('save_', 'load_')) and spec.name.endswith('_bundle')
]


@pytest.mark.parametrize('tool', BUNDLE_TOOLS)
@pytest.mark.parametrize('arguments', [{}, {'path': ''}])
def test_missing_path_never_writes_or_mutates(tool, arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = Session.ingest(pd.DataFrame({'x': [1., 2., 3., 4.], 'y': [0, 1, 0, 1]}))
    session.set_roles({'y': 'target'})
    before = dict(vars(session))
    registry = build_default_registry()
    proposal = propose_tool_execution(tool, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert not result.executed
    assert result.error
    assert not result.state_changes
    assert not list(tmp_path.iterdir())
    assert vars(session).keys() == before.keys()
    for name, value in before.items():
        assert vars(session)[name] is value, f'{tool} changed {name} without a path'
