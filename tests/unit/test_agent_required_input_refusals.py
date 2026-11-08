"""Malformed agent requests fail before altering data or writing artifacts."""

import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


@pytest.mark.parametrize('tool,arguments', [
    ('set_roles', {}), ('drop_columns', {}), ('checkpoint_save', {}),
    ('evolutionary_search', {}),
    ('evolutionary_search', {'param_space': {'alpha': [.1, 1.]}}),
    ('set_graph', {}), ('rag_retrieve', {}), ('rag_generate', {}),
    ('rag_ingest_corpus', {}), ('export_torch', {}), ('transcribe_speech', {}),
    ('load_pretrained_backbone', {}), ('pack_torchserve', {}),
    ('prepare_tensorrt_export', {}), ('emit_k8s_ddp_job', {}),
    ('attach_backbone_head', {}), ('evaluate_asr', {}),
    ('emit_k8s_serve_deployment', {}),
])
def test_missing_required_inputs_produce_refusal_without_side_effects(tool, arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = (Session.ingest(pd.DataFrame({'x': [1., 2., 3., 4.], 'y': [0, 1, 0, 1]}))
         .set_roles({'x': 'feature', 'y': 'target'}))
    frame = s.to_pandas().copy(deep=True)
    roles, history = dict(s.dataset.roles), list(s._history)
    registry = build_default_registry()
    proposal = propose_tool_execution(tool, arguments, registry)
    result = execute_tool(s, proposal, confirmed=True, registry=registry)
    assert not result.executed and 'requires' in result.error
    assert result.state_changes == ()
    assert s._fit_result is None and s._split_plan is None
    assert s._history == history and s.dataset.roles == roles
    pd.testing.assert_frame_equal(s.to_pandas(), frame)
    assert not list(tmp_path.iterdir())
