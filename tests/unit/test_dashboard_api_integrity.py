"""Real dashboard routes must retain report evidence and reject unknown resources."""

import csv
import io

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.fixture
def studio():
    pytest.importorskip("fastapi")
    pytest.importorskip("plotly")
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    values = np.arange(60, dtype=float)
    frame = pd.DataFrame({"x": values, "related": 2 * values,
                          "site": ["east", "west"] * 30, "y": [0, 1] * 30})
    frame.loc[:14, "x"] = np.nan
    session = (Session.ingest(frame)
               .set_roles({"x": "feature", "related": "feature", "site": "feature", "y": "target"})
               .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=4))
    report = session.eda(include_plots=False, show=False, max_plots=0)
    payload = report.to_dict()
    set_state(DashboardState(report=report, report_dict=payload, title="API evidence",
                             session_meta={"has_split": True}))
    try:
        with TestClient(create_app()) as client:
            yield client, payload
    finally:
        clear_state()


def test_health_meta_and_raw_report_preserve_actual_population(studio):
    client, payload = studio
    assert client.get("/api/health").json()["title"] == "API evidence"
    meta = client.get("/api/meta").json()
    assert meta["overview"]["n_rows"] == 60
    assert meta["session"]["has_split"] is True
    raw = client.get("/api/report")
    assert raw.status_code == 200
    assert raw.json()["overview"]["n_rows"] == payload["overview"]["n_rows"]
    assert raw.json()["quality"]["missing_cell_count"] == 15


@pytest.mark.parametrize("path", ["/api/domains/no-such-domain", "/api/charts/no-such-chart",
                                  "/api/concepts/no-such-concept", "/api/export/csv/no-such-section"])
def test_unknown_resources_are_explicit_json_404s(studio, path):
    client, _ = studio
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["detail"]
    assert "Traceback" not in response.text


def test_concept_search_and_detail_keep_the_source_key(studio):
    client, _ = studio
    results = client.get("/api/concepts", params={"q": " LEAKAGE "}).json()
    assert results["count"] == len(results["concepts"])
    assert any(item["key"] == "leakage-boundary" for item in results["concepts"])
    detail = client.get("/api/concepts/leakage-boundary").json()
    assert detail["concept"]["key"] == "leakage-boundary"
    assert detail["concept"]["assumptions"]
    empty = client.get("/api/search", params={"q": "zzzz_no_existing_concept_or_finding"}).json()
    assert empty["findings"] == empty["concepts"] == empty["domains"] == []
    found = client.get("/api/search", params={"q": "missing"}).json()
    assert found["findings"] or found["concepts"]


def test_downloaded_missingness_csv_matches_actual_cells(studio):
    client, _ = studio
    response = client.get("/api/export/csv/missing_rates")
    assert response.status_code == 200
    assert "buildml_eda_missing_rates.csv" in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    row = next(item for item in rows if item["column"] == "x")
    assert float(row["missing_rate"]) == pytest.approx(15 / 60)


@pytest.mark.parametrize("domain", ["academy", "gates"])
def test_derived_boards_keep_domain_identity_and_report_context(studio, domain):
    client, _ = studio
    response = client.get(f"/api/domains/{domain}")
    assert response.status_code == 200
    assert response.json()["domain"]["key"] == domain


def test_chart_catalog_and_selected_chart_agree_across_themes(studio):
    client, _ = studio
    light = client.get("/api/charts", params={"theme": "light"}).json()
    dark = client.get("/api/charts", params={"theme": "dark"}).json()
    assert set(light) == set(dark)
    identifier = "missing_rates"
    figure = client.get(f"/api/charts/{identifier}", params={"theme": "DARK"}).json()
    assert figure["id"] == identifier
    assert figure["figure"] == dark[identifier]
