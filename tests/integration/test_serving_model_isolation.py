"""Two live HTTP servers must retain independent models through shutdown."""
import concurrent.futures
import socket

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.serving.launch import serve_bundle


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("stop_first", [0, 1])
def test_two_live_models_do_not_contaminate_predictions_or_shutdown_each_other(tmp_path, stop_first):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("uvicorn")
    handles = []
    try:
        for i, (slope, intercept) in enumerate([(2., 3.), (-4., 17.)]):
            frame = pd.DataFrame({"x": np.arange(40, dtype=float)})
            frame["y"] = slope * frame.x + intercept
            session = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
                       .split(test_size=.2, random_state=7).fit(LinearRegression(), task="regression"))
            bundle = session.save_pipeline(tmp_path / f"model-{i}")
            handles.append(serve_bundle(bundle, port=port(), trusted=True, title=f"Model {i}"))

        def request(i):
            with httpx.Client(base_url=handles[i].url, timeout=5) as client:
                expected = [11., -11.] if i == 0 else [1., 45.]
                response = client.post("/predict", json={"rows": [{"x": 4.}, {"x": -7.}]})
                assert response.status_code == 200
                np.testing.assert_allclose(response.json()["predictions"], expected)
                assert client.get("/health").json()["title"] == f"Model {i}"
                assert client.get("/metadata").json()["path"].endswith(f"model-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(request, [0, 1] * 8))
        handles[stop_first].stop()
        request(1 - stop_first)
    finally:
        for handle in handles:
            handle.stop()
