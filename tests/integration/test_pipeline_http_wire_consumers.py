"""Saved pipelines expose the contract their HTTP scoring routes use."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session

ENDPOINTS = ("/predict", "/predict/batch")


@pytest.fixture(scope="module")
def pipeline_service(tmp_path_factory):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from buildml.serving.app import create_serving_app

    signal = np.arange(40, dtype=float)
    frame = pd.DataFrame({"signal": signal, "label": np.where(signal < 20, "cold", "warm")})
    session = (
        Session.ingest(frame)
        .set_roles({"signal": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
    )
    session.scale(method="standard")
    session.fit(LogisticRegression(), task="classification")
    root = tmp_path_factory.mktemp("scoring-bundle")
    session.save_pipeline(root)
    fitted = session.fit_result.estimator
    train = signal[list(session.split_plan.train_indices)]

    def probabilities(values):
        scaled = (np.asarray(values) - train.mean()) / train.std()
        logits = scaled * fitted.coef_[0, 0] + fitted.intercept_[0]
        warm = 1 / (1 + np.exp(-logits))
        return np.column_stack((1 - warm, warm))

    def client(path=root):
        return TestClient(create_serving_app(path, kind="pipeline", trusted=True))

    return root, client, probabilities


@pytest.mark.parametrize("legacy", (False, True), ids=("current", "legacy"))
def test_metadata_reports_persisted_contract_when_present(pipeline_service, legacy, tmp_path):
    root, client, _ = pipeline_service
    contract_file = root / "schema_contract.json"
    expected = json.loads(contract_file.read_text(encoding="utf-8"))
    if legacy:
        root = tmp_path / "legacy"
        shutil.copytree(
            contract_file.parent, root, ignore=shutil.ignore_patterns("schema_contract.json")
        )

    with client(root) as http:
        response = http.get("/metadata")
        assert response.status_code == 200
        assert response.json()["kind"] == "pipeline"
        assert response.json().get("schema_contract") == (None if legacy else expected)
        assert http.post("/predict", json={"rows": [{"signal": 7.0}]}).status_code == 200


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("alias", ("rows", "instances"))
def test_pipeline_rows_keep_order_probabilities_and_batch_metadata(
    pipeline_service, endpoint, alias
):
    _, client, probabilities = pipeline_service
    values = [34.0, 7.0, 21.0]
    with client() as http:
        response = http.post(
            endpoint, json={alias: [{"signal": v} for v in values], "return_proba": True}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "pipeline" and body["n_rows"] == len(values)
    assert body.get("batch", False) is (endpoint == "/predict/batch")
    expected = probabilities(values)
    assert body["predictions"] == np.where(expected[:, 1] > 0.5, "warm", "cold").tolist()
    np.testing.assert_allclose(body["probabilities"]["proba_cold"], expected[:, 0], atol=1e-12)
    np.testing.assert_allclose(body["probabilities"]["proba_warm"], expected[:, 1], atol=1e-12)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("rows", ([{"signal": 1}, 7], [{}, None], [[2.0]], [None], [], "bad"))
def test_non_object_rows_are_client_errors(pipeline_service, endpoint, rows):
    _, client, _ = pipeline_service
    with client() as http:
        invalid = http.post(endpoint, json={"instances": rows})
        assert invalid.status_code == 400, invalid.text
        assert 'expects JSON {"rows":' in invalid.json()["detail"]
        assert http.post(endpoint, json={"rows": [{"signal": 7.0}]}).status_code == 200
