"""Saved TorchScript models preserve their output structure across HTTP."""

import numpy as np
import pytest

ENDPOINTS = ("/predict", "/predict/batch")
INPUTS = {"inputs": [[1.0, 2.0], [3.0, -4.0]]}


@pytest.fixture(scope="module")
def serving_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from buildml.serving.app import create_serving_app

    return torch, TestClient, create_serving_app


@pytest.fixture(scope="module")
def multi_output_model(serving_runtime):
    torch, _, _ = serving_runtime

    class MultiOutput(torch.nn.Module):
        def forward(self, inputs):
            score = 2 * inputs[:, 0:1] - inputs[:, 1:2] + 3
            margins = {"margins": [score - 3, score + 3]}
            labels = {4: [score.round().long()]}
            flags = {"positive": inputs[:, 0] > 0}
            return score, margins, labels, flags, ("units", 2, 0.125, True, None)

    return torch.jit.script(MultiOutput())


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("layout", ("file", "preferred", "fallback"))
def test_structured_scores_keep_tensor_axes_and_json_types(
    endpoint, layout, serving_runtime, multi_output_model, tmp_path
):
    _, client_class, create_app = serving_runtime
    preferred = tmp_path / "model.ts.pt"
    fallback = tmp_path / "model.pt"
    selected = preferred if layout == "preferred" else fallback
    multi_output_model.save(str(selected))
    if layout == "fallback":
        preferred.mkdir()
    elif layout == "preferred":
        fallback.write_bytes(b"not a model")

    app = create_app(selected if layout == "file" else tmp_path, kind="torchscript", trusted=True)
    with client_class(app) as client:
        metadata = client.get("/metadata").json()
        assert metadata["kind"] == "torchscript" and metadata["path"] == str(selected)
        response = client.post(endpoint, json=INPUTS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "torchscript" and body["n_rows"] == 2
    assert body.get("batch", False) is (endpoint == "/predict/batch")
    score, margins, labels, flags, units = body["outputs"]
    inputs = np.asarray(INPUTS["inputs"])
    expected = inputs @ np.array([[2.0], [-1.0]]) + 3
    np.testing.assert_allclose(score, expected)
    np.testing.assert_allclose(margins["margins"], [expected - 3, expected + 3])
    assert labels == {"4": expected.astype(int).reshape(1, 2, 1).tolist()}
    assert type(labels["4"][0][0][0]) is int
    assert flags["positive"] == [True, True] and type(flags["positive"][0]) is bool
    assert units == ["units", 2, 0.125, True, None] and type(units[3]) is bool


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "bad_output", ("nonfinite_tensor", "nonfinite_scalar", "complex", "device")
)
def test_bad_structured_outputs_return_sanitized_errors(
    endpoint, bad_output, serving_runtime, tmp_path
):
    torch, client_class, create_app = serving_runtime

    class NonfiniteTensor(torch.nn.Module):
        def forward(self, inputs):
            return {"private_scores": [inputs, inputs / (inputs - inputs)]}

    class NonfiniteScalar(torch.nn.Module):
        def forward(self, inputs):
            return inputs, {"private_score": float("inf")}

    class ComplexTensor(torch.nn.Module):
        def forward(self, inputs):
            return inputs, inputs.to(dtype=torch.complex64)

    class UnsupportedDevice(torch.nn.Module):
        def forward(self, inputs):
            return inputs, inputs.device

    model_type = {
        "nonfinite_tensor": NonfiniteTensor,
        "nonfinite_scalar": NonfiniteScalar,
        "complex": ComplexTensor,
        "device": UnsupportedDevice,
    }[bad_output]
    path = tmp_path / "model.ts"
    torch.jit.script(model_type()).save(str(path))
    app = create_app(tmp_path, kind="torchscript", trusted=True)
    with client_class(app) as client:
        response = client.post(endpoint, json=INPUTS)
        assert client.get("/health").json()["ok"] is True
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed due to an internal serving error"}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("payload", "detail"),
    (
        ("{}", "expects JSON"),
        ('{"inputs": [[1e309, 0]]}', "finite numeric values"),
    ),
)
def test_invalid_requests_leave_structured_model_available(
    endpoint, payload, detail, serving_runtime, multi_output_model, tmp_path
):
    _, client_class, create_app = serving_runtime
    path = tmp_path / "model.pt"
    multi_output_model.save(str(path))
    with client_class(create_app(path, kind="torchscript", trusted=True)) as client:
        invalid = client.post(
            endpoint, content=payload, headers={"Content-Type": "application/json"}
        )
        assert invalid.status_code == 400 and detail in invalid.json()["detail"]
        valid = client.post(endpoint, json=INPUTS)
    assert valid.status_code == 200, valid.text
    assert valid.json()["outputs"][0] == [[3.0], [13.0]]
