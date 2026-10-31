"""Real configuration startup and bounded startup-failure cleanup."""

import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.serving.config import ServeConfig
from buildml.serving.launch import ServingLaunchError, serve_bundle


@pytest.fixture
def bundle(tmp_path):
    frame = pd.DataFrame({"x": np.arange(40, dtype=float)})
    frame["y"] = 2 * frame.x + 3
    session = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
               .split(test_size=0.2, random_state=4).fit(LinearRegression(), task="regression"))
    return session.save_pipeline(tmp_path / "pipeline")


def _free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


@pytest.mark.parametrize("override", [False, True])
def test_configured_real_server_uses_credential_precedence_and_explicit_docs(bundle, override):
    pytest.importorskip("uvicorn")
    httpx = pytest.importorskip("httpx")
    config = ServeConfig(bundle=str(bundle), port=_free_port(), title="Configured contract",
                         trusted=True, api_keys=("configured-key",), docs_enabled=True)
    options = {"api_keys": "override-key", "title": "Override contract"} if override else {}
    handle = serve_bundle(config=config, **options)
    correct = "override-key" if override else "configured-key"
    wrong = "configured-key" if override else "override-key"
    try:
        assert handle.is_running
        with httpx.Client(base_url=handle.url, timeout=5) as client:
            invalid = client.post("/predict", headers={"X-API-Key": wrong},
                                  json={"rows": [{"x": 4.0}]})
            assert invalid.status_code == 401
            headers = {"X-API-Key": correct}
            assert client.get("/docs", headers=headers).status_code == 200
            schema = client.get("/openapi.json", headers=headers).json()
            assert schema["info"]["title"] == options.get("title", config.title)
            response = client.post("/predict", headers=headers, json={"rows": [{"x": 4.0}]})
            assert response.status_code == 200
            np.testing.assert_allclose(response.json()["predictions"], [11.0])
    finally:
        handle.stop()
        handle.stop()
    assert not handle.is_running
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", handle.port)) != 0


def test_startup_timeout_refuses_success_and_stops_the_unready_worker(bundle, monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    from buildml.serving import launch

    stopped = threading.Event()
    workers = []

    class UnreadyServer:
        started = False
        should_exit = False

        def __init__(self, config):
            workers.append(self)

        def run(self):
            while not self.should_exit:
                time.sleep(0.001)
            stopped.set()

    clock_values = iter([0.0, 11.0])
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(clock_values, 11.0))
    monkeypatch.setattr(uvicorn, "Server", UnreadyServer)
    with pytest.raises(ServingLaunchError, match="ready within"):
        serve_bundle(bundle, port=_free_port(), trusted=True)
    assert workers[0].should_exit
    assert stopped.is_set()
