"""Case retrieval exposes train-only evidence and retention preserves its boundary."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, arguments):
    registry = build_default_registry()
    return execute_tool(session, propose_tool_execution(name, arguments, registry),
                        confirmed=True, registry=registry)


def _session():
    rng = np.random.default_rng(730)
    frame = pd.DataFrame(rng.normal(size=(80, 2)), columns=["x", "z"])
    frame["y"] = (frame.x > 0).astype(int)
    session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
               .split(test_size=.25, validation_size=.25, random_state=11)
               .ai.configure(provider="mock"))
    fitted = _run(session, "fit_cbr", {"backend": "sklearn", "metric": "euclidean",
                                      "standardize": False, "k": 3})
    assert fitted.executed, fitted.error
    return session


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_agent_neighbors_match_brute_force_training_distances(partition):
    session = _session()
    retrieved = _run(session, "retrieve_cases", {"partition": partition, "k": 3})
    assert retrieved.executed, retrieved.error
    train = session.partition("train")
    queries = session.partition(partition)
    for trace, (index, query) in zip(retrieved.result.traces, queries.iterrows(), strict=True):
        distances = np.linalg.norm(train[["x", "z"]].to_numpy() - query[["x", "z"]].to_numpy(), axis=1)
        nearest = np.argsort(distances)[:3]
        assert trace.query_index == index
        assert tuple(trace.neighbor_row_indices) == tuple(train.index[nearest])
        np.testing.assert_allclose(trace.distances, distances[nearest])
        assert not set(trace.neighbor_row_indices) & set(queries.index)


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_agent_retention_refuses_holdout_without_mutating_memory(partition):
    session = _session()
    plan = session.cbr_plan
    size = plan.case_base.n_cases
    result = _run(session, "retain_cbr", {
        "row_indices": [int(session.partition(partition).index[0])],
        "source_disclosure": "Attempted holdout retention must be refused",
    })
    assert not result.executed
    assert "validation/test" in result.error
    assert session.cbr_plan is plan
    assert plan.case_base.n_cases == size


def test_agent_retention_duplicate_train_rows_preserve_case_count():
    session = _session()
    size = session.cbr_plan.case_base.n_cases
    result = _run(session, "retain_cbr", {
        "row_indices": [int(session.partition("train").index[0])],
        "source_disclosure": "Training case already in memory",
    })
    assert result.executed, result.error
    assert "n_added=0" in result.result
    assert "n_skipped=1" in result.result
    assert session.cbr_plan.case_base.n_cases == size


def test_retention_requires_cases_in_the_advertised_schema():
    registry = build_default_registry()
    schema = registry.get("retain_cbr").parameters
    assert "row_indices" in schema["required"]
    assert schema["properties"]["row_indices"]["type"] == "array"
    session = _session()
    plan = session.cbr_plan
    result = _run(session, "retain_cbr", {"source_disclosure": "Reviewed cases"})
    assert not result.executed
    assert "row_indices" in result.error
    assert session.cbr_plan is plan


def test_agent_neural_case_memory_honors_platform_guard_and_epoch_budget(monkeypatch):
    monkeypatch.delenv("BUILDML_ALLOW_CBR_TORCH_WINDOWS", raising=False)
    torch = pytest.importorskip("torch")
    session = _session()
    steps = []
    original_step = torch.optim.Adam.step

    def counted_step(optimizer, *args, **kwargs):
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    schema = build_default_registry().get("fit_cbr").parameters["properties"]
    assert "device" in schema and "torch_epochs" in schema
    result = _run(session, "fit_cbr", {"backend": "torch", "device": "cpu",
                                       "torch_epochs": 2, "random_state": 19, "k": 3})
    if sys.platform == "win32":
        assert not result.executed
        assert "refused on Windows" in result.error
        assert steps == []
        assert session.cbr_plan.backend == "sklearn"
        return
    assert result.executed, result.error
    assert len(steps) == 2
    assert session.cbr_plan.backend == "torch"
    assert session.cbr_plan.config["device"] == "cpu"
    assert {p.device.type for p in session.cbr_plan.case_base.torch_encoder_.parameters()} == {"cpu"}
    predicted = _run(session, "predict_cbr", {"partition": "test", "return_traces": True})
    assert predicted.executed, predicted.error
    assert len(predicted.result.predictions) == len(session.partition("test"))
