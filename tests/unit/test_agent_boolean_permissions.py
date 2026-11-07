"""Malformed booleans cannot authorize deserialization or test-set tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_nonboolean_cannot_enable_test_partition_tuning(value):
    frame = pd.DataFrame({"x": list(range(100)), "y": [0] * 50 + [1] * 50})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    session.split(test_size=.2, validation_size=.2, random_state=3)
    session.fit(LogisticRegression(), task="classification")
    session.ai.configure(provider="mock")
    registry = build_default_registry()
    proposal = propose_tool_execution("fit_decision_policy", {
        "partition": "test", "allow_test_tuning": value,
    }, registry)
    outcome = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert not outcome.executed
    assert "allow_test_tuning must be a JSON boolean" in outcome.error
    assert session.decision_plan is None


def test_string_false_cannot_authorize_loading_a_real_nlp_bundle(tmp_path):
    frame = pd.DataFrame({"text": ["apple orchard fruit", "engine oil valve"] * 30,
                          "label": [0, 1] * 30})

    def make_session():
        return Session.ingest(frame).set_roles({"text": "feature", "label": "target"}).split(
            test_size=.2, random_state=3
        ).ai.configure(provider="mock")

    trained = make_session()
    trained.nlp.fit_classifier(text_column="text")
    path = str(tmp_path / "nlp")
    trained.nlp.save_bundle(path)
    restored = make_session()
    registry = build_default_registry()
    for trusted in ("false", False):
        proposal = propose_tool_execution("load_nlp_bundle", {"path": path, "trusted": trusted}, registry)
        outcome = execute_tool(restored, proposal, confirmed=True, registry=registry)
        assert not outcome.executed
        assert "trust" in outcome.error.lower()
        assert restored.nlp.text_plan is None
    proposal = propose_tool_execution("load_nlp_bundle", {"path": path, "trusted": True}, registry)
    outcome = execute_tool(restored, proposal, confirmed=True, registry=registry)
    assert outcome.executed, outcome.error
    expected = trained.nlp.predict(partition="test")
    actual = restored.nlp.predict(partition="test")
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
