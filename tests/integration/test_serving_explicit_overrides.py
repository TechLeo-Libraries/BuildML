"""Explicit False/default-valued kwargs must override a configured permission."""
import socket
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.serving.app import ServingState, create_serving_app
from buildml.serving.config import ServeConfig
from buildml.serving.launch import serve_bundle


@pytest.mark.parametrize("entry", ["app", "server"])
def test_explicit_false_revokes_configured_artifact_trust_before_deserialization(entry, tmp_path, monkeypatch):
    pytest.importorskip("uvicorn")
    import joblib

    frame = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    s = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
         .split(test_size=.2, random_state=7).fit(LinearRegression(), task="regression"))
    bundle = s.save_pipeline(tmp_path / "model")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = ServeConfig(bundle=str(bundle), trusted=True, port=port)

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit trust refusal reached joblib deserialization")

    monkeypatch.setattr(joblib, "load", forbidden)
    with pytest.raises(ValidationError, match="trust"):
        if entry == "app":
            create_serving_app(bundle, config=config, trusted=False)
        else:
            serve_bundle(config=config, trusted=False)


def test_explicit_default_app_kind_title_and_cpu_override_configuration(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from buildml.serving import app as module

    captured = {}

    def load(path, **kwargs):
        captured.update(kwargs)
        return ServingState(kind=kwargs["kind"], path=path)

    monkeypatch.setattr(module, "load_serving_bundle", load)
    config = ServeConfig(bundle=str(tmp_path), kind="torchscript", title="Configured", map_location="cuda", trusted=True)
    app = create_serving_app(tmp_path, config=config, kind="pipeline", title="BuildML Serve", map_location="cpu")
    assert captured == {"kind": "pipeline", "map_location": "cpu", "trusted": True}
    with TestClient(app) as client:
        assert client.get("/health").json()["title"] == "BuildML Serve"
    module.clear_serving_state(app)


def test_explicit_default_server_values_override_configuration_without_binding(monkeypatch, tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    from buildml.serving import launch

    captured = {}

    def app(path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(state=SimpleNamespace(buildml_serving_state=None))

    class FakeServer:
        started = True
        should_exit = False

        def __init__(self, config):
            captured["host"] = config.host
            captured["port"] = config.port

        def run(self):
            while not self.should_exit:
                time.sleep(.002)

    monkeypatch.setattr(launch, "create_serving_app", app)
    monkeypatch.setattr(launch, "_ensure_port_available", lambda host, port: None)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    config = ServeConfig(bundle=str(tmp_path), host="0.0.0.0", port=8123, kind="torchscript",
                         title="Configured", map_location="cuda", trusted=True, api_keys=("test-key",),
                         allow_insecure_public_bind=True)
    handle = serve_bundle(config=config, host="127.0.0.1", port=8080, kind="pipeline",
                          title="BuildML Serve", map_location="cpu", allow_insecure_public_bind=False)
    try:
        assert handle.host == "127.0.0.1" and handle.port == 8080
        assert captured["kind"] == "pipeline"
        assert captured["title"] == "BuildML Serve"
        assert captured["map_location"] == "cpu"
    finally:
        handle.stop()


def test_session_wrapper_keeps_omitted_permissions_unset_but_forwards_explicit_false(monkeypatch, tmp_path):
    from buildml.core._defaults import _INHERIT_PERMISSION
    from buildml.serving import launch

    calls = []

    def observe(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(url="http://127.0.0.1:8123")

    monkeypatch.setattr(launch, "serve_bundle", observe)
    s = Session.ingest(pd.DataFrame({"x": [1.]}))
    config = ServeConfig(bundle=str(tmp_path), trusted=True, port=8123)
    s.dl.serve(config=config)
    assert calls[-1]["trusted"] is _INHERIT_PERMISSION
    assert calls[-1]["host"] is None and calls[-1]["port"] is None
    s.dl.serve(config=config, trusted=False, allow_insecure_public_bind=False)
    assert calls[-1]["trusted"] is False
    assert calls[-1]["allow_insecure_public_bind"] is False
