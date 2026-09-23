"""Timeseries teaching layer and capability-matrix wiring tests."""

from __future__ import annotations

import pandas as pd

from buildml import Session
from buildml.explain.beginner import BEGINNER_LAYERS
from buildml.explain.concepts import CONCEPT_NOTES, get_concept


def test_timeseries_concepts_have_beginner_layers() -> None:
    for key in (
        "ts-decomposition",
        "ts-stationarity-diagnostics",
        "ts-changepoint-detection",
        "ts-analysis-before-forecast",
    ):
        assert key in CONCEPT_NOTES
        assert key in BEGINNER_LAYERS
        note = get_concept(key)
        assert note.plain_summary


def test_session_graph_and_causal_capability_matrices() -> None:
    causal = Session.ingest(pd.DataFrame({"x": [1]})).causal.capability_matrix()
    graph = Session.ingest(pd.DataFrame({"x": [1]})).graph.capability_matrix()
    assert "backends" in causal
    assert "backends" in graph


def test_ai_executor_dispatches_rl_capability_matrix() -> None:
    from buildml.ai.executor import execute_tool, propose_tool_execution
    from buildml.ai.tools import build_default_registry

    registry = build_default_registry()
    proposal = propose_tool_execution("rl_capability_matrix", {}, registry)
    session = Session.ingest(__import__("pandas").DataFrame({"a": [1.0], "y": [0]}))
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed
    assert result.error is None
    assert "rl_backends" in (result.result or {})
