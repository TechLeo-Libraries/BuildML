"""Real loopback serving must score correctly, enforce auth and release its port."""
from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.serving.launch import serve_bundle


def _free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def test_real_http_prediction_authentication_schema_and_shutdown(tmp_path):
    pytest.importorskip("uvicorn")
    httpx = pytest.importorskip("httpx")
    frame = pd.DataFrame({"x": np.arange(40, dtype=float)})
    frame["y"] = 2 * frame.x + 3
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    session.split(test_size=.2, random_state=0)
    session.fit(LinearRegression(), task="regression")
    bundle = tmp_path / "pipeline"
    session.save_pipeline(bundle)
    port = _free_port()
    handle = serve_bundle(bundle, port=port, api_keys="contract-key", trusted=True)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            payload = {"rows": [{"x": 4.}, {"x": -7.}]}
            assert client.post("/predict", json=payload).status_code == 401
            assert client.post("/predict", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
            headers = {"Authorization": "Bearer contract-key"}
            assert client.get("/health", headers=headers).status_code == 200
            assert client.get("/docs", headers=headers).status_code == 404
            response = client.post("/predict", json=payload, headers=headers)
            assert response.status_code == 200
            np.testing.assert_allclose(response.json()["predictions"], [11., -11.])
            invalid = client.post("/predict", json={"rows": [{"wrong": 3}]}, headers=headers)
            assert invalid.status_code in {400, 422}
            assert "predictions" not in invalid.json()
    finally:
        handle.stop()
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize("options", [
    {"port": 0}, {"port": 65536}, {"port": True}, {"port": 8080.5}, {"host": ""},
    {"host": "0.0.0.0"}, {"ssl_certfile": "missing.pem"},
])
def test_invalid_launch_rejected_before_artifact_deserialization(options, tmp_path):
    pytest.importorskip("uvicorn")
    with pytest.raises(ValidationError):
        serve_bundle(tmp_path / "artifact-does-not-exist", **options)
