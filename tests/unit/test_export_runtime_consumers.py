"""Actual serialized graphs preserve control flow, input/output names and batches."""

from importlib.util import find_spec

import numpy as np
import pytest

if find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)
import torch  # noqa: E402

from buildml.core.errors import ValidationError  # noqa: E402
from buildml.dl.export import export_onnx, export_torchscript, load_torchscript  # noqa: E402


class Branch(torch.nn.Module):
    def forward(self, x):
        if bool(x.sum() > 0):
            return x * 2
        return x - 7


class TwoOutputs(torch.nn.Module):
    def forward(self, left, right):
        return left + right, left - right


class Unscriptable(torch.nn.Module):
    def forward(self, x):
        return (lambda value: value * 2)(x)


def test_scripted_artifact_executes_both_value_dependent_branches(tmp_path):
    model = Branch()
    exported = export_torchscript(
        model, tmp_path / "branch.pt", example_input=torch.ones(2, 2), method="script"
    )
    loaded = load_torchscript(exported.path, trusted=True)
    for value in (torch.ones(1, 2), -torch.ones(5, 2)):
        torch.testing.assert_close(loaded(value), model(value))
    assert not any("used trace" in text for text in exported.warnings)


def test_trace_discloses_frozen_branch_instead_of_implying_control_flow_equivalence(tmp_path):
    model = Branch()
    exported = export_torchscript(model, tmp_path / "trace.pt", example_input=torch.ones(2, 2))
    loaded = load_torchscript(exported.path, trusted=True)
    assert any("control-flow" in text for text in exported.warnings)
    assert not torch.equal(loaded(-torch.ones(2, 2)), model(-torch.ones(2, 2)))


def test_invalid_torchscript_method_cannot_publish_mislabeled_artifact(tmp_path):
    destination = tmp_path / "existing.pt"
    destination.write_bytes(b"prior artifact")
    with pytest.raises(ValidationError, match="method|trace|script"):
        export_torchscript(
            torch.nn.Linear(2, 1), destination, example_input=torch.ones(2, 2), method="typo"
        )
    assert destination.read_bytes() == b"prior artifact"


def test_failed_scripting_preserves_previous_artifact(tmp_path):
    destination = tmp_path / "existing.pt"
    destination.write_bytes(b"prior artifact")
    with pytest.raises(ValidationError, match="script"):
        export_torchscript(
            Unscriptable(), destination, example_input=torch.ones(2, 2), method="script"
        )
    assert destination.read_bytes() == b"prior artifact"


@pytest.mark.parametrize("dynamic", [False, True])
def test_multi_input_multi_output_onnx_shapes_and_numerics(dynamic, tmp_path):
    if find_spec("onnx") is None or find_spec("onnxruntime") is None:
        pytest.skip("ONNX and ONNX Runtime are not installed")
    import onnx
    import onnxruntime as ort

    model = TwoOutputs()
    exported = export_onnx(
        model,
        tmp_path / "multi.onnx",
        example_input=(torch.ones(3, 2), torch.ones(3, 2)),
        input_names=["left", "right"],
        output_names=["sum", "difference"],
        dynamic_batch=dynamic,
    )
    graph = onnx.load(str(exported.path)).graph
    assert set(exported.dynamic_axes) == ({"left", "right", "sum", "difference"} if dynamic else set())
    for value in [*graph.input, *graph.output]:
        dimension = value.type.tensor_type.shape.dim[0]
        assert (dimension.dim_param == "batch") is dynamic
        if not dynamic:
            assert dimension.dim_value == 3
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(
        str(exported.path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in runtime.get_outputs()] == ["sum", "difference"]
    for rows in [1, 5] if dynamic else [3]:
        left = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        right = left / 2 + 0.3
        plus, minus = runtime.run(None, {"left": left, "right": right})
        np.testing.assert_allclose(plus, left + right, rtol=1e-6)
        np.testing.assert_allclose(minus, left - right, rtol=1e-6)


def test_onnx_input_name_arity_is_checked_before_writing_artifact(tmp_path):
    if find_spec("onnx") is None:
        pytest.skip("ONNX is not installed")
    destination = tmp_path / "bad.onnx"
    with pytest.raises(ValidationError, match="input_names"):
        export_onnx(
            TwoOutputs(),
            destination,
            example_input=(torch.ones(3, 2), torch.ones(3, 2)),
            input_names=["only-one"],
        )
    assert not destination.exists()
