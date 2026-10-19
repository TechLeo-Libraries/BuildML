"""Exported artifacts are consumed after relocation, through real agent tools."""

from __future__ import annotations

import ast
import json
import shlex
import shutil
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

if find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)
import torch  # noqa: E402

from buildml import Session  # noqa: E402
from buildml.ai.executor import execute_tool, propose_tool_execution  # noqa: E402
from buildml.ai.tools import build_default_registry  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _run(session, name, arguments, *, success=True):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed is success, result.error
    return result.result if success else result.error


@pytest.fixture
def trained():
    values = np.random.default_rng(29).normal(size=(80, 2))
    frame = pd.DataFrame(values, columns=["x", "z"])
    frame["y"] = 1.3 * frame.x - 0.4 * frame.z
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=7)
    )
    session.dl.make_loaders(task="regression", batch_size=16, normalize=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        session.dl.fit(torch.nn.Linear(2, 1), epochs=1, device="cpu")
    session.ai.configure(provider="mock")
    return session


def _generated_handler(path):
    # Execute the generated methods with a minimal BaseHandler protocol shim.
    # This exercises actual TorchScript inference, not a running TorchServe server.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "ts.torch_handler.base_handler")
    ]
    scope = {"BaseHandler": object}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["BuildMLTorchScriptHandler"]()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("request_sizes", [[1], [3], [2, 1, 3]])
def test_agent_torchserve_pack_survives_relocation_and_preserves_each_request(
    trained, explicit, request_sizes, tmp_path
):
    source = tmp_path / "trained model.pt"
    _run(trained, "export_torch", {"path": str(source), "format": "torchscript"})
    arguments = {"output_dir": str(tmp_path / "original package"), "model_name": "risk-model"}
    if explicit:
        arguments["torchscript_path"] = str(source)
    packed = _run(trained, "pack_torchserve", arguments)
    relocated = tmp_path / "relocated package"
    shutil.move(str(packed.path), relocated)
    source.unlink()
    handler = _generated_handler(relocated / "handler.py")
    handler.initialize(SimpleNamespace(system_properties={"model_dir": str(relocated)}))
    probe = torch.arange(sum(request_sizes) * 2, dtype=torch.float32).reshape(-1, 2) / 7
    expected = trained._dl_train_result.module.eval()(probe).detach().numpy()
    requests = []
    offset = 0
    for i, count in enumerate(request_sizes):
        payload = {"instances" if i % 2 == 0 else "inputs": probe[offset : offset + count].tolist()}
        body = json.dumps(payload).encode() if i % 2 == 0 else payload
        requests.append({"body" if i % 2 == 0 else "data": body})
        offset += count
    output = handler.postprocess(handler.inference(handler.preprocess(requests)))
    assert len(output) == len(request_sizes)
    offset = 0
    for reply, count in zip(output, request_sizes):
        np.testing.assert_allclose(reply, expected[offset : offset + count], rtol=1e-6, atol=1e-7)
        offset += count
    config = dict(
        line.split("=", 1)
        for line in (relocated / "config.properties").read_text().splitlines()
        if "=" in line
    )
    assert (relocated / config["model_store"]).resolve() == relocated.resolve()
    manifest = json.loads((relocated / "manifest.json").read_text())
    assert manifest["model"]["modelName"] == "risk-model"


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_agent_tensorrt_recipe_relocates_and_references_a_runnable_onnx(
    trained, fp16, explicit, tmp_path
):
    if find_spec("onnx") is None or find_spec("onnxruntime") is None:
        pytest.skip("ONNX and ONNX Runtime are not installed")
    import onnxruntime as ort

    source = tmp_path / "trained model.onnx"
    _run(trained, "export_torch", {"path": str(source), "format": "onnx"})
    arguments = {
        "output_dir": str(tmp_path / "original plan"),
        "engine_name": "risk engine.plan",
        "fp16": fp16,
    }
    if explicit:
        arguments["onnx_path"] = str(source)
    packed = _run(trained, "prepare_tensorrt_export", arguments)
    relocated = tmp_path / "relocated plan"
    shutil.move(str(packed.path), relocated)
    source.unlink()
    plan = json.loads((relocated / "tensorrt_plan.json").read_text())
    artifact = relocated / plan["onnx_path"]
    assert artifact.is_file()
    tokens = shlex.split(plan["trtexec_example"])
    assert "--onnx=trained model.onnx" in tokens
    assert "--saveEngine=risk engine.plan" in tokens
    assert ("--fp16" in tokens) is fp16
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(
        str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
    )
    probe = np.array([[0.1, 0.2], [-0.3, 0.7], [1.0, -2.0]], dtype=np.float32)
    actual = runtime.run(None, {runtime.get_inputs()[0].name: probe})[0]
    expected = trained._dl_train_result.module.eval()(torch.tensor(probe)).detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert not (relocated / "risk engine.plan").exists()  # recipe only


@pytest.mark.parametrize("tool", ["pack_torchserve", "prepare_tensorrt_export"])
def test_agent_packaging_without_a_matching_export_fails_without_publishing_state(tool, tmp_path):
    session = Session.ingest(pd.DataFrame({"x": [1.0, 2.0]})).ai.configure(provider="mock")
    destination = tmp_path / "must not exist"
    before = getattr(session, "_dl_packaging_result", None)
    error = _run(session, tool, {"output_dir": str(destination)}, success=False)
    assert "requires" in error
    assert not destination.exists()
    assert getattr(session, "_dl_packaging_result", None) is before
