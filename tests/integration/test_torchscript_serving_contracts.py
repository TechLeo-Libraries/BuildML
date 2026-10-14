"""Real TorchScript scoring must honor model placement and batched inputs."""
import numpy as np
import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("endpoint", ["/predict", "/predict/batch"])
@pytest.mark.parametrize("storage", ["parameters", "buffers"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_real_torchscript_service_places_inputs_with_model_and_scores_known_function(device, endpoint, storage, precision, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from buildml.serving.app import create_serving_app

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime unavailable")
    if storage == "parameters":
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2., -1.]]))
            model.bias.fill_(3.)
    else:
        class BufferModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("coefficients", torch.tensor([2., -1.]))
            def forward(self, inputs):
                return (inputs * self.coefficients).sum(dim=1, keepdim=True) + 3
        model = BufferModel()
    dtype = getattr(torch, precision)
    model = model.to(dtype=dtype)
    path = tmp_path / "model.ts.pt"
    torch.jit.trace(model.eval(), torch.ones(1, 2, dtype=dtype)).save(str(path))
    app = create_serving_app(path, kind="torchscript", map_location=device, trusted=True, api_keys="test-key")
    with TestClient(app) as client:
        assert client.post(endpoint, json={"inputs": [[1., 2.]]}).status_code == 401
        response = client.post(endpoint, headers={"X-API-Key": "test-key"}, json={"inputs": [[1., 2.], [3., -4.]]})
        assert response.status_code == 200, response.text
        assert response.json()["n_rows"] == 2
        np.testing.assert_allclose(response.json()["outputs"], [[3.], [13.]])
        stored = getattr(app.state.buildml_serving_state.torchscript_module, storage)
        assert next(stored()).device.type == device


@pytest.mark.parametrize("inputs", [[], [1., 2.], 3., "invalid", [[1., 2.], [3.]], [["nan", 2.]]])
def test_malformed_torchscript_batches_are_client_errors(inputs, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from buildml.serving.app import create_serving_app

    path = tmp_path / "model.ts.pt"
    torch.jit.trace(torch.nn.Linear(2, 1), torch.ones(1, 2)).save(str(path))
    with TestClient(create_serving_app(path, kind="torchscript", trusted=True)) as client:
        response = client.post("/predict", json={"inputs": inputs})
        assert response.status_code == 400, response.text


def test_nonfinite_model_outputs_return_sanitized_internal_error(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from buildml.serving.app import create_serving_app

    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.bias.fill_(float("nan"))
    path = tmp_path / "model.ts.pt"
    torch.jit.trace(model, torch.ones(1, 2), check_trace=False).save(str(path))
    with TestClient(create_serving_app(path, kind="torchscript", trusted=True)) as client:
        response = client.post("/predict", json={"inputs": [[1., 2.]]})
        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction failed due to an internal serving error"}
