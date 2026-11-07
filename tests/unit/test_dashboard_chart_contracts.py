"""Dashboard chart responses preserve source values and request-local themes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.fixture(scope="module")
def client_type():
    pytest.importorskip("plotly")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    return TestClient


@pytest.fixture
def dashboard(client_type):
    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    def setup(report, payload=None):
        set_state(
            DashboardState(
                report=report,
                report_dict=report.to_dict() if payload is None else payload,
                title="Chart contract",
                session_meta={},
            )
        )
        return create_app()

    try:
        yield setup
    finally:
        clear_state()


@pytest.mark.parametrize("single_dark_chart", (False, True))
def test_concurrent_chart_requests_keep_each_theme(
    dashboard, client_type, monkeypatch, single_dark_chart
):
    from buildml.dashboard import charts
    from buildml.eda.industry_tokens import INDUSTRY_CHART_DARK, INDUSTRY_CHART_LIGHT

    data = pd.DataFrame({"x": [0, 1, 2, 3, None, 4] * 4, "z": np.arange(24)})
    report = Session.ingest(data).eda(max_plots=0)
    app = dashboard(report)
    with client_type(app) as client:
        warm = client.get("/api/charts")
        assert warm.status_code == 200
        chart_ids = set(warm.json())

    barrier = Barrier(2)
    real_builder = charts._fig_severity

    def synchronized_first_figure(go, data):
        barrier.wait(timeout=20)
        figure = real_builder(go, data)
        barrier.wait(timeout=20)
        return figure

    monkeypatch.setattr(charts, "_fig_severity", synchronized_first_figure)

    def request(path, theme):
        with client_type(app) as client:
            response = client.get(path, params={"theme": theme})
            assert response.status_code == 200, response.text
            return response.json()

    dark_path = "/api/charts/severity_map" if single_dark_chart else "/api/charts"
    with ThreadPoolExecutor(max_workers=2) as workers:
        light = workers.submit(request, "/api/charts", "light")
        dark = workers.submit(request, dark_path, "dark")
        light_figures = light.result(timeout=45)
        dark_result = dark.result(timeout=45)
    if single_dark_chart:
        assert dark_result["id"] == "severity_map"
        dark_figures = {dark_result["id"]: dark_result["figure"]}
    else:
        dark_figures = dark_result
        assert set(dark_figures) == chart_ids
    assert set(light_figures) == chart_ids

    for name, palette, figures in (
        ("light", INDUSTRY_CHART_LIGHT, light_figures),
        ("dark", INDUSTRY_CHART_DARK, dark_figures),
    ):
        for chart, figure in figures.items():
            assert figure["layout"]["meta"]["buildml_theme"] == name, chart
            assert figure["layout"]["font"]["color"] == palette["ink"], chart
            assert figure["layout"]["xaxis"]["gridcolor"] == palette["grid"], chart


@pytest.mark.parametrize("legacy", (False, True))
def test_real_wide_correlations_keep_labels_and_values(dashboard, client_type, legacy):
    names = ["columns", "matrix", *[f"feature_{i}" for i in range(18)]]
    values = np.random.default_rng(19).normal(size=(40, len(names)))
    report = Session.ingest(pd.DataFrame(values, columns=names)).eda(max_plots=0)
    payload = report.to_dict()
    if legacy:
        for method in ("pearson", "spearman"):
            source = payload["bivariate"][method]
            payload["bivariate"][method] = {
                "columns": names,
                "matrix": [[source[row][column] for column in names] for row in names],
            }
    app = dashboard(report, payload)
    with client_type(app, raise_server_exceptions=False) as client:
        response = client.get("/api/charts", params={"theme": "dark"})
    assert response.status_code == 200, response.text
    figures = response.json()
    expected = {
        "correlation_heatmap": np.corrcoef(values[:, :18], rowvar=False),
        "spearman_heatmap": np.corrcoef(
            np.argsort(np.argsort(values[:, :18], axis=0), axis=0), rowvar=False
        ),
    }
    for chart, correlation in expected.items():
        (heatmap,) = figures[chart]["data"]
        assert heatmap["x"] == heatmap["y"] == names[:18]
        np.testing.assert_allclose(heatmap["z"], correlation, atol=1e-12)


def test_legacy_outlier_rate_keeps_measured_zero(dashboard, client_type):
    report = Session.ingest(pd.DataFrame({"clean": np.arange(32)})).eda(max_plots=0)
    payload = report.to_dict()
    source = payload["outliers"]["per_column"]["clean"]
    assert source["iqr_outlier_rate"] == source["iqr_outlier_count"] == 0
    payload["outliers"]["per_column"] = {"clean": {"rate": 0.0, "count": 0}}
    with client_type(dashboard(report, payload)) as client:
        response = client.get("/api/charts/outlier_rates")
    assert response.status_code == 200
    (bar,) = response.json()["figure"]["data"]
    assert bar["x"] == [0.0] and bar["y"] == ["clean"] and bar["customdata"] == [0]
