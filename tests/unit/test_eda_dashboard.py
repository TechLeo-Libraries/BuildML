"""Tests for the local Industry EDA App."""

from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.dashboard.charts import build_chart_catalog
from buildml.dashboard.exports import export_csv, export_pdf, list_csv_sections
from buildml.dashboard.launch import DashboardLaunchError, launch_eda_app
from buildml.dashboard.teaching import build_teaching_studios
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.explain.concepts import CONCEPT_NOTES

pytest.importorskip("fastapi")
pytest.importorskip("plotly")
pytest.importorskip("reportlab")


def _session() -> Session:
    frame = pd.DataFrame(
        {
            "age": [21, 25, 30, 35, 40, 45, None, 55, 60, 22, 28, 33],
            "income": [40, 55, 60, 80, 50, 70, 65, 90, 95, 42, 48, 58],
            "city": ["a", "b", "a", "b", "a", "b", "a", "a", "b", "a", "b", "a"],
            "const": [1] * 12,
            "id_like": list(range(12)),
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "city": "feature",
                "y": "target",
                "id_like": "id",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
    )


def _dirty_classification_session() -> Session:
    frame = pd.DataFrame(
        {
            "age<unsafe>": [20, 22, None, 35, 35, 42] * 8,
            "income": [30, 40, 50, 80, 80, 120] * 8,
            "city": ["north", "south", "north", "", "", "west"] * 8,
            "constant": ["same"] * 48,
            "customer_id": [f"customer-{index}" for index in range(48)],
            "target": [0, 1, 0, 1, 0, 1] * 8,
        }
    )
    session = Session.ingest(frame).set_roles(
        {
            "age<unsafe>": "feature",
            "income": "feature",
            "city": "feature",
            "constant": "feature",
            "customer_id": "id",
            "target": "target",
        }
    )
    session.split(test_size=0.25, stratify=True, random_state=7)
    return session


def _chart_has_trace(figure: dict) -> bool:
    data = figure.get("data") or []
    if not data:
        return False
    # Empty placeholder figures have no meaningful series values.
    for trace in data:
        for key in ("x", "y", "z", "values", "value"):
            values = trace.get(key)
            if values is None:
                continue
            if isinstance(values, list) and values:
                return True
            if not isinstance(values, list):
                return True
    return False


def test_teaching_studios_include_worked_dataset_values() -> None:
    report = _session().eda(include_plots=False, show=False)
    studios = build_teaching_studios(report.to_dict())
    for key in (
        "cockpit",
        "quality",
        "features",
        "relationships",
        "multivariate",
        "target",
        "outliers",
        "visuals",
    ):
        studio = studios[key]
        assert studio["definition"]
        assert studio["why"]
        assert studio["how"]
        assert studio["interpretation"]
        assert studio["pitfalls"]
        assert studio["thresholds"]
        assert studio["assumptions"]
        assert studio["practice_checklist"]
        assert studio["mastery_notes"]
        assert len(studio["definition"]) >= 180
        assert len(studio["practice_checklist"]) >= 3
        assert studio["worked_example"]["values"]
        assert studio["next_action"]["api"]
        assert "research-grade" not in studio["definition"].lower()
        assert "actionable insights" not in studio["why"].lower()

    features = studios["features"]["worked_example"]["values"]
    assert features["numeric_columns"] >= 1
    assert features["categorical_columns"] >= 1
    assert features["example_column"]
    outliers = studios["outliers"]["worked_example"]["values"]
    assert outliers["columns_screened"] >= 1
    assert outliers["top_iqr_rates"]


def test_concept_academy_keys_cover_requested_topics() -> None:
    required = {
        "column-roles",
        "leakage-boundary",
        "data-splitting",
        "missing-data",
        "categorical-encoding",
        "feature-scaling",
        "class-imbalance",
        "baselines",
        "overfitting",
        "probability-calibration",
        "thresholds",
        "feature-importance",
        "dataset-drift",
        "mutual-information",
        "variance-inflation",
        "principal-components",
        "normality-screens",
    }
    assert required.issubset(CONCEPT_NOTES.keys())
    assert "eda_app" in OPERATION_CATALOG
    for note in CONCEPT_NOTES.values():
        assert note.definition
        assert note.intuition
        assert note.formal_idea
        assert len(note.details) >= 8
        assert note.why_it_matters
        assert note.how_buildml_uses
        assert note.interpretation_rules
        assert note.assumptions
        assert note.failure_modes
        assert note.anti_patterns
        assert note.worked_example_pattern
        blob = " ".join(
            [
                note.summary,
                note.definition,
                note.intuition,
                *note.details[:4],
            ]
        ).lower()
        assert "research-grade" not in blob
        assert "actionable insights" not in blob


def test_feature_and_outlier_boards_match_analyzer_schema() -> None:
    report = _dirty_classification_session().eda(
        include_plots=False,
        show=False,
        sample_rows=24,
        max_plots=0,
    )
    payload = report.to_dict()
    catalog = build_chart_catalog(payload)

    for chart_id in (
        "skew_profile",
        "quartile_spread",
        "normality_flags",
        "cardinality_entropy",
        "outlier_rates",
        "outlier_bounds",
        "zscore_outlier_rates",
        "spearman_heatmap",
        "correlation_heatmap",
    ):
        assert chart_id in catalog
        assert _chart_has_trace(catalog[chart_id]), chart_id

    # Cramér / multivariate screens may be empty when pair support is thin.
    assert "cramers_v_bars" in catalog
    assert "multivariate_anomaly" in catalog

    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    set_state(
        DashboardState(
            report=report,
            report_dict=payload,
            title="Dirty Classification Studio",
            session_meta={"has_split": True},
        )
    )
    try:
        client = TestClient(create_app())
        features = client.get("/api/domains/features").json()
        assert features["chart_ids"] == [
            "skew_profile",
            "quartile_spread",
            "normality_flags",
            "cardinality_entropy",
        ]
        assert features["teaching"]["worked_example"]["values"]["numeric_columns"] >= 1
        assert "income" in str(features["teaching"]["worked_example"])

        outliers = client.get("/api/domains/outliers").json()
        assert "outlier_rates" in outliers["chart_ids"]
        assert "outlier_bounds" in outliers["chart_ids"]
        assert outliers["teaching"]["worked_example"]["values"]["top_iqr_rates"]
        for item in outliers["findings"]:
            key = str(item.get("key", ""))
            assert key.startswith("outliers.") or key.startswith("outlier.")
    finally:
        clear_state()


def test_csv_and_pdf_exports(tmp_path: Path) -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    sections = {item["key"] for item in list_csv_sections(report)}
    assert "findings" in sections
    assert "univariate_numeric" in sections
    assert "univariate_categorical" in sections
    assert "outliers" in sections
    assert "mutual_information" in sections or "missing_rates" in sections

    filename, content = export_csv(report, "findings")
    assert filename.endswith(".csv")
    assert "key," in content

    uni_name, uni_csv = export_csv(report, "univariate_numeric")
    assert uni_name.endswith(".csv")
    assert "column," in uni_csv
    assert "skew" in uni_csv

    out_name, out_csv = export_csv(report, "outliers")
    assert out_name.endswith(".csv")
    assert "iqr_outlier_rate" in out_csv

    pdf = export_pdf(report, view="features", title="Test Studio")
    assert pdf[:4] == b"%PDF"
    # Structured briefing (cover + tables + optional chart stills) should exceed a bare dump.
    assert len(pdf) > 1200
    assert pdf.count(b"endobj") >= 8
    assert pdf.count(b"/Page") >= 2  # multi-page structure (cover/TOC/body)
    out = tmp_path / "briefing.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pdf)
    # Chart stills are embedded when kaleido is available.
    pytest.importorskip("kaleido")
    assert b"/Image" in pdf
    assert len(pdf) > 20_000


def test_fastapi_routes_and_launch_smoke() -> None:
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import clear_state

    session = _session()
    report = session.eda(include_plots=False, show=False)
    handle = launch_eda_app(
        report,
        host="127.0.0.1",
        port=8769,
        open_browser=False,
        title="Test EDA Studio",
        session_meta={"has_split": True},
    )
    try:
        assert handle.url.startswith("http://127.0.0.1:8769")
        assert handle.is_running
        clear_state()  # isolate TestClient app factory state
        # Re-bind state for ASGI test client without needing live server port.
        from buildml.dashboard.state import DashboardState, set_state

        set_state(
            DashboardState(
                report=report,
                report_dict=report.to_dict(),
                title="Test EDA Studio",
                session_meta={"has_split": True},
            )
        )
        client = TestClient(create_app())
        home = client.get("/")
        assert home.status_code == 200
        assert "BuildML EDA App" in home.text
        assert 'class="sheet"' in home.text
        assert 'id="sheet-chrome"' in home.text
        assert 'id="domain-nav"' in home.text
        assert "/static/js/app.js" in home.text
        assert "/static/js/learn_ui.js" in home.text
        assert "Offline HTML" in home.text
        assert "PDF briefing" not in home.text
        assert 'id="csv-export"' not in home.text
        assert 'id="html-download"' in home.text
        assert "btn-primary blueprint" in home.text
        assert "Command cockpit" in home.text
        assert "Readiness gates" in home.text
        assert "Concept academy" in home.text

        meta = client.get("/api/meta").json()
        assert meta["domains"]
        assert any(d["key"] == "academy" for d in meta["domains"])
        assert any(d["key"] == "gates" for d in meta["domains"])
        csv_keys = {item["key"] for item in meta["csv_sections"]}
        assert "univariate_numeric" in csv_keys
        assert "outliers" in csv_keys

        cockpit = client.get("/api/cockpit").json()
        assert cockpit["teaching"]["worked_example"]["values"]
        assert "definition" in cockpit["teaching"]
        assert "sheet" in cockpit
        assert cockpit["sheet"]["kpis"]["readiness"]
        assert "register" in cockpit["sheet"]
        assert "ledger" in cockpit["sheet"]
        assert "sequence" in cockpit["sheet"]
        assert "assumptions" in cockpit["sheet"]
        assert cockpit["adapt"]["target_column"] == "y"
        assert cockpit["sheet"]["adapt"]["task"] == "classification"
        assert cockpit["sheet"]["session_sentence"]
        assert "spine_meta" in cockpit["sheet"]
        meta_adapt = client.get("/api/meta").json()["adapt"]
        assert meta_adapt["target_column"] == "y"

        quality = client.get("/api/domains/quality").json()
        assert quality["teaching"]["pitfalls"]
        assert quality["teaching"]["how"]

        features = client.get("/api/domains/features").json()
        assert "skew_profile" in features["chart_ids"]
        outliers = client.get("/api/domains/outliers").json()
        assert "outlier_bounds" in outliers["chart_ids"]

        concepts = client.get("/api/concepts?q=mutual").json()
        assert concepts["count"] >= 1
        detail = client.get("/api/concepts/mutual-information").json()
        assert detail["concept"]["key"] == "mutual-information"
        academy = client.get("/api/domains/academy").json()
        assert academy["domain"]["key"] == "academy"
        assert academy["concepts"]
        assert academy["stages"]
        assert "cited_count" in academy

        gates = client.get("/api/gates").json()
        assert gates["counts"]["total"] >= 40
        assert gates["groups"]
        assert gates["persistence"]["human_decisions"] is False
        assert gates["persistence"]["session_api"] is False
        assert gates["persistence"]["disk"] is False
        assert "ephemeral_notice" in gates
        gates_domain = client.get("/api/domains/gates").json()
        assert gates_domain["domain"]["key"] == "gates"
        assert gates_domain["gates"]["counts"]["total"] == gates["counts"]["total"]

        # No write / save endpoints for gate human decisions.
        assert client.post("/api/gates").status_code in {404, 405}
        assert client.put("/api/gates").status_code in {404, 405}
        assert client.patch("/api/gates").status_code in {404, 405}

        charts = client.get("/api/charts").json()
        assert "severity_map" in charts
        assert "skew_profile" in charts
        assert "outlier_rates" in charts
        assert "data" in charts["severity_map"]
        dark = client.get("/api/charts?theme=dark").json()
        assert "severity_map" in dark
        light_ink = ((charts["severity_map"].get("layout") or {}).get("font") or {}).get("color")
        dark_ink = ((dark["severity_map"].get("layout") or {}).get("font") or {}).get("color")
        assert light_ink != dark_ink
        # Industry steel accents in light catalog
        light_accent = None
        for trace in charts["severity_map"].get("data") or []:
            marker = trace.get("marker") or {}
            color = marker.get("color")
            if isinstance(color, str) and color.startswith("#"):
                light_accent = color
                break
        layout_meta = (charts["severity_map"].get("layout") or {}).get("colorway") or []
        assert "#5980a6" in layout_meta or light_accent == "#5980a6" or any(
            "#5980a6" in str(trace) for trace in (charts["severity_map"].get("data") or [])
        ) or ((charts["severity_map"].get("layout") or {}).get("font") or {}).get("color") == "#1d1f20"

        csv_resp = client.get("/api/export/csv/findings")
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.headers["content-type"]
        uni_resp = client.get("/api/export/csv/univariate_numeric")
        assert uni_resp.status_code == 200
        out_resp = client.get("/api/export/csv/outliers")
        assert out_resp.status_code == 200

        pdf_resp = client.get("/api/export/pdf?view=cockpit")
        assert pdf_resp.status_code == 200
        assert pdf_resp.content[:4] == b"%PDF"

        html_resp = client.get("/api/export/html")
        assert html_resp.status_code == 200
        assert b"__BUILDML_OFFLINE__" in html_resp.content
        assert b"sheet-chrome" in html_resp.content
        assert b"Offline EDA App" in html_resp.content
        assert b"plotly" in html_resp.content.lower()
        assert b'"gates"' in html_resp.content
        assert b'"sheet"' in html_resp.content

        # SPA assets — Industry redesign sheet chrome
        tokens = client.get("/static/css/tokens.css")
        assert tokens.status_code == 200
        assert "#5980a6" in tokens.text
        assert ".blueprint" in tokens.text
        css = client.get("/static/css/app.css")
        assert css.status_code == 200
        assert "kpi-strip" in css.text
        assert "spine" in css.text
        assert "gate-card" in css.text
        assert "academy-index" in css.text
        js = client.get("/static/js/app.js")
        assert js.status_code == 200
        assert "Findings register" in js.text
        assert "renderCockpit" in js.text
        assert "renderGates" in js.text
        assert "renderAcademy" in js.text
        assert "gateSessionMarks" in js.text
        assert "per_column" in js.text
        assert "charts_dark" in js.text or "theme=" in js.text
        assert "offlineApi" in js.text or "__BUILDML_OFFLINE__" in js.text
        assert "#5980a6" in js.text
        assert 'from "./learn_ui.js"' in js.text
        assert 'from "./cockpit_view.js"' in js.text
        gates_js = client.get("/static/js/gates_view.js")
        assert gates_js.status_code == 200
        assert "Mark for this session" in gates_js.text
        assert 'from "./learn_ui.js"' in gates_js.text
        cockpit_js = client.get("/static/js/cockpit_view.js")
        assert cockpit_js.status_code == 200
        assert "openCockpitDrawer" in cockpit_js.text
        assert 'from "./learn_ui.js"' in cockpit_js.text
        assert "data-ledger-jump" in cockpit_js.text
        learn_js = client.get("/static/js/learn_ui.js")
        assert learn_js.status_code == 200
        assert "sectionScaffold" in learn_js.text
        assert "localStorage.setItem(\"buildml-eda-gates\"" not in js.text
        assert "localStorage.setItem(\"buildml-eda-gates\"" not in gates_js.text
        assert cockpit["sheet"]["ledger"][0]["teaching"]["levels"]
        assert "cockpit-drawer" in home.text
    finally:
        handle.stop()
        clear_state()


def test_session_eda_app_method_records_history() -> None:
    session = _session()
    handle = session.eda_app(open_browser=False, port=8770, title="Session Studio")
    try:
        assert session.last_eda is not None
        assert any(item.get("operation_id") == "eda_app" for item in session.history)
        assert handle.url.endswith(":8770/")
    finally:
        handle.stop()


def test_launch_errors_when_port_in_use() -> None:
    report = _session().eda(include_plots=False, show=False)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    try:
        with pytest.raises(DashboardLaunchError) as exc_info:
            launch_eda_app(report, host="127.0.0.1", port=port, open_browser=False)
        message = str(exc_info.value)
        assert str(port) in message
        assert "already in use" in message.lower()
        assert "buildml[dashboard]" in message
        assert "eda_app(port=" in message
    finally:
        sock.close()


def test_missing_dashboard_extra_message_mentions_install() -> None:
    err = MissingExtraError("dashboard", "Industry EDA App")
    assert "pip install 'buildml[dashboard]'" in str(err)


def test_studio_offline_html_and_theme_catalog(tmp_path: Path) -> None:
    from buildml.dashboard.offline import export_studio_html

    report = _session().eda(include_plots=False, show=False).to_dict()
    light = build_chart_catalog(report, theme="light")
    dark = build_chart_catalog(report, theme="dark")
    light_color = light["severity_map"]["layout"]["font"]["color"]
    dark_color = dark["severity_map"]["layout"]["font"]["color"]
    assert light_color != dark_color

    path = tmp_path / "studio.html"
    export_studio_html(report, path, title="Offline Studio")
    html = path.read_text(encoding="utf-8")
    assert "__BUILDML_OFFLINE__" in html
    assert "charts_light" in html
    assert "charts_dark" in html
    assert "sheet-chrome" in html
    assert "Offline EDA App" in html
    # Document shell has no remote script/link tags (plotly is inlined).
    assert "<script src=" not in html
    assert "<link " not in html.split("<style>", 1)[0]
