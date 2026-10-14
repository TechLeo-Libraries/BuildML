"""Real agent-selected Torch budgets and artifacts retain train-only contracts."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, args):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, args, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("format", ["torchscript", "onnx"])
def test_real_agent_training_and_export_match_heldout_inference(device, format, tmp_path):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime unavailable")
    if format == "onnx":
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
    rng = np.random.default_rng(321)
    x = rng.normal(size=(128, 2))
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": 3 * x[:, 0] - 2 * x[:, 1]})
    session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
               .split(test_size=.2, validation_size=.2, random_state=19))
    heldout = session.partition("test").copy(deep=True)
    _run(session, "make_torch_loaders", {"batch_size": 16, "normalize": True, "apply_plans": False})
    bundle = session._torch_loaders
    train = session.partition("train")[["x", "z"]].to_numpy()
    np.testing.assert_allclose(bundle.contract.normalize_mean, train.mean(axis=0), atol=1e-6)
    _run(session, "fit_torch", {"epochs": 2, "learning_rate": .015, "device": device})
    trained = session.dl_train_result
    assert trained.n_epochs_ran == 2
    assert trained.config.learning_rate == .015
    assert next(trained.module.parameters()).device.type == device
    inputs = torch.cat([batch[0] for batch in bundle.loaders["test"]])
    with torch.no_grad():
        expected = trained.module.eval()(inputs.to(device)).cpu().numpy()
    assert np.isfinite(expected).all()
    destination = tmp_path / ("model.pt" if format == "torchscript" else "model.onnx")
    exported = _run(session, "export_torch", {"path": str(destination), "format": format})
    assert destination.is_file()
    assert exported.meta["task"] == "regression"
    if format == "torchscript":
        artifact = torch.jit.load(str(destination), map_location="cpu")
        with torch.no_grad():
            actual = artifact(inputs).numpy()
    else:
        artifact = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        actual = artifact.run(None, {artifact.get_inputs()[0].name: inputs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    # Portable export's documented CPU move must be explicit in this workflow.
    assert next(trained.module.parameters()).device.type == "cpu"
    pd.testing.assert_frame_equal(session.partition("test"), heldout)
