"""Release regressions for nullable data and small-sample dashboard reports."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session


@pytest.mark.parametrize("size", [4, 12])
def test_nullable_features_support_missing_values_without_mutating_data(size):
    frame = pd.DataFrame({
        "integer": pd.Series(([1, 2, None, None] * 3)[:size], dtype="Int64"),
        "boolean": pd.Series(([True, False, None, True] * 3)[:size], dtype="boolean"),
        "y": ([0, 1, 0, 1] * 3)[:size],
    })
    session = Session.ingest(frame).set_roles({"y": "target"})
    report = session.eda()
    if size >= 10:
        assert set(report.bivariate["mutual_information_vs_target"]) == {"integer", "boolean"}
    pd.testing.assert_frame_equal(session.to_pandas(), frame)


def test_categorical_mutual_information_uses_discrete_estimator():
    # Perfect binary correspondence has exactly ln(2) nats of information.
    # A continuous nearest-neighbour estimate on arbitrary codes is incorrect.
    frame = pd.DataFrame({"category": ["a", "b"] * 20, "y": [0, 1] * 20})
    report = Session.ingest(frame).set_roles({"y": "target"}).eda()
    assert report.bivariate["mutual_information_vs_target"]["category"] == pytest.approx(np.log(2))


def test_role_chart_counts_columns_without_declared_roles():
    go = pytest.importorskip("plotly.graph_objects")
    from buildml.dashboard.charts import _fig_roles

    figure = _fig_roles(go, {"overview": {
        "n_columns": 11, "roles": {"row_id": "id", "outcome": "target"},
    }})
    trace = figure["data"][0]
    assert dict(zip(trace["labels"], trace["values"], strict=True)) == {
        "id": 1, "target": 1, "undeclared role": 9,
    }


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_small_regression_statistics_are_unavailable_not_invalid(dtype):
    frame = pd.DataFrame({"x": [1, 2], "y": pd.Series([.5, None], dtype=dtype)})
    report = Session.ingest(frame).set_roles({"y": "target"}).eda()
    assert report.target["summary"] == {
        "type": "regression_target", "mean": .5, "std": None, "skew": None,
    }


@pytest.mark.parametrize("case", ["small", "missing", "constant"])
def test_dashboard_serves_edge_reports_and_exports(case):
    pytest.importorskip("fastapi")
    pytest.importorskip("plotly")
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    frames = {
        "small": pd.DataFrame({"x": [1, 2], "y": pd.Series([.5, None], dtype="Float64")}),
        "missing": pd.DataFrame({"x": [None] * 12, "y": [None] * 12}),
        "constant": pd.DataFrame({"x": [1.] * 12, "y": [.5] * 12}),
    }
    report = Session.ingest(frames[case]).set_roles({"y": "target"}).eda()
    set_state(DashboardState(report, report.to_dict()))
    try:
        with TestClient(create_app()) as client:
            for route in ["/api/meta", "/api/cockpit", "/api/gates", "/api/charts",
                          "/api/domains/target", "/api/domains/academy", "/api/export/html"]:
                response = client.get(route)
                assert response.status_code == 200, (case, route, response.text[:200])
            for section in client.get("/api/meta").json()["csv_sections"]:
                assert client.get(f"/api/export/csv/{section['key']}").status_code == 200
    finally:
        clear_state()
