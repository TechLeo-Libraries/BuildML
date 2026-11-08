"""AI proposals must execute real offline retrieval and Torch training paths."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, **arguments):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_agent_indexes_retrieves_and_invalidates_a_real_offline_corpus(mode):
    pytest.importorskip("sentence_transformers")
    session = Session()
    documents = [{"doc_id": "planet", "text": "jupiter gas giant planet astronomy"}]
    documents += [{"doc_id": f"other-{i}", "text": f"nectarine fruit orchard harvest cultivar {i}"}
                  for i in range(30)]
    _run(session, "rag_ingest_corpus", documents=documents)
    _run(session, "rag_embed_and_index", embedder="hashing")
    result = _run(session, "rag_retrieve", query="jupiter gas giant planet astronomy", k=3, mode=mode)
    assert result.hits[0].doc_id == "planet"
    assert len(result.hits) == 3
    _run(session, "rag_ingest_corpus", documents=[{"doc_id": "new", "text": "saturn rings"}])
    registry = build_default_registry()
    proposal = propose_tool_execution("rag_retrieve", {"query": "jupiter", "k": 1}, registry)
    refused = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert not refused.executed
    assert "index" in refused.error.lower()
    _run(session, "rag_embed_and_index", embedder="hashing")
    assert _run(session, "rag_retrieve", query="saturn rings", k=1, mode=mode).hits[0].doc_id == "new"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_agent_trains_cpu_torch_and_exports_numerically_matching_model(task, tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(45)
    rng = np.random.default_rng(29)
    points = rng.normal(size=(80, 2))
    frame = pd.DataFrame(points, columns=["x", "z"])
    frame["y"] = (points[:, 0] > 0).astype(int) if task == "classification" else 2 * points[:, 0] + points[:, 1]
    session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
               .split(test_size=0.2, validation_size=0.2, random_state=4))
    before = session.dataset.frame.copy(deep=True)
    _run(session, "make_torch_loaders", batch_size=16, normalize=True, apply_plans=False)
    _run(session, "fit_torch", epochs=2, learning_rate=0.01, device="cpu")
    trained = session.dl_train_result
    assert trained.n_epochs_ran == 2
    assert next(trained.module.parameters()).device.type == "cpu"
    scores = _run(session, "evaluate_torch", partition="test")
    assert scores.n_rows == len(session.split_plan.test_indices)
    assert np.isfinite(list(scores.metrics.values())).all()
    pd.testing.assert_frame_equal(session.dataset.frame, before)
    destination = tmp_path / "model.pt"
    _run(session, "export_torch", path=str(destination), format="torchscript")
    restored = torch.jit.load(str(destination)).eval()
    trained.module.eval()
    inputs = torch.tensor([[0.2, 1.0], [-0.8, 0.3]], dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), trained.module(inputs))
