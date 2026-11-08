"""ONNX export metadata describes the interface exposed by the saved graph."""

import numpy as np
import pytest

from buildml.dl.export import export_onnx, smoke_load_onnx


@pytest.fixture
def onnx_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1

    def load(path):
        return ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    return torch, load


@pytest.mark.parametrize("dynamic", [False, True])
def test_export_reports_surviving_inputs_and_all_automatic_outputs(onnx_runtime, tmp_path, dynamic):
    torch, load = onnx_runtime

    class TwoOutputs(torch.nn.Module):
        def forward(self, unused_mask, features):
            return features * 2 + 3, features.sum(dim=1)

    exported = export_onnx(
        TwoOutputs(),
        tmp_path / "pruned.onnx",
        example_input=(torch.ones(3, 1), torch.ones(3, 2)),
        input_names=["mask", "features"],
        dynamic_batch=dynamic,
    )
    client = load(exported.path)
    runtime_inputs = client.get_inputs()
    runtime_outputs = client.get_outputs()
    runtime_names = [node.name for node in runtime_outputs]
    assert [node.name for node in runtime_inputs] == exported.meta["input_names"] == ["features"]
    assert len(runtime_names) == 2
    assert exported.meta["output_names"] == runtime_names
    assert exported.example_input_shapes == ((3, 1), (3, 2))
    assert exported.dynamic_axes == {
        node.name: {0: node.shape[0]}
        for node in [*runtime_inputs, *runtime_outputs]
        if dynamic and node.shape and isinstance(node.shape[0], str)
    }
    info = smoke_load_onnx(exported.path)
    assert info["inputs"] == ["features"]
    assert info["outputs"] == runtime_names
    for rows in (1, 5) if dynamic else (3,):
        values = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        first, second = client.run(exported.meta["output_names"], {"features": values})
        np.testing.assert_allclose(first, 2 * values + 3)
        np.testing.assert_allclose(second, values.sum(axis=1))


def test_scalar_output_has_no_batch_axis_in_export_metadata(onnx_runtime, tmp_path):
    torch, load = onnx_runtime

    class Average(torch.nn.Module):
        def forward(self, features):
            return features.mean()

    exported = export_onnx(Average(), tmp_path / "scalar.onnx", example_input=torch.ones(3, 2))
    client = load(exported.path)
    assert client.get_outputs()[0].shape == []
    assert exported.dynamic_axes == {"input": {0: "batch"}}
    for rows in (1, 5):
        values = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        (actual,) = client.run(exported.meta["output_names"], {"input": values})
        assert actual.shape == ()
        np.testing.assert_allclose(actual, values.mean())
