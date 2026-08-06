"""Cockpit readiness-sheet UX: teaching sidebar, ledger routing, anti-overlap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.cockpit_teaching import (
    LEDGER_GROUP_GLOSSARY,
    enrich_cockpit_sheet,
)
from buildml.dashboard.sheet import build_cockpit_sheet
from buildml.eda.cockpit_style import COCKPIT_CSS
from buildml.eda.html_report import _ledger_block
from buildml.eda.sheet_coverage import build_ledger_groups

pytest.importorskip("fastapi")

_DASHBOARD = Path(__file__).resolve().parents[2] / "buildml" / "dashboard"


def _session() -> Session:
    frame = pd.DataFrame(
        {
            "very_long_feature_name_monthly_spend_usd": [
                10,
                20,
                None,
                40,
                50,
                60,
                70,
                80,
            ],
            "num_b": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
            "const_c": [1] * 8,
            "row_id": list(range(8)),
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "very_long_feature_name_monthly_spend_usd": "feature",
                "num_b": "feature",
                "const_c": "feature",
                "row_id": "id",
                "label": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
    )


def test_cockpit_sheet_carries_teaching_and_glossary() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    sheet = build_cockpit_sheet(report)

    assert sheet["assumptions_purpose"]["purpose"]
    assert sheet["ledger_purpose"]["purpose"]
    assert "exclusions" in sheet["ledger_glossary"] or "frame" in sheet["ledger_glossary"]
    assert sheet["ledger"]
    for group in sheet["ledger"]:
        assert group.get("teaching")
        assert group["teaching"]["worked_example"]["code"]
        assert "levels" in group["teaching"]
        if group["key"] in LEDGER_GROUP_GLOSSARY:
            assert group.get("means")
            assert group.get("why_on_sheet")
    assert sheet["assumptions"]
    assert sheet["assumptions"][0].get("teaching")
    assert sheet["register"]
    assert sheet["register"][0].get("teaching")

    # Adaptive: no demo churn schema.
    blob = str(sheet).lower()
    assert "target_churn" not in blob
    assert "monthly_charges" not in blob


def test_ledger_glossary_covers_known_group_keys() -> None:
    required = {
        "frame",
        "roles",
        "missing",
        "quality-flags",
        "mi",
        "screens",
        "univariate",
        "exclusions",
        "skipped",
    }
    assert required.issubset(set(LEDGER_GROUP_GLOSSARY))
    for key in required:
        assert LEDGER_GROUP_GLOSSARY[key]["means"]
        assert LEDGER_GROUP_GLOSSARY[key]["why_on_sheet"]


def test_enrich_is_idempotent_enough() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    sheet = build_cockpit_sheet(report)
    again = enrich_cockpit_sheet(sheet, report)
    assert again["ledger"][0]["teaching"]["key"] == sheet["ledger"][0]["teaching"]["key"]


def test_app_js_never_routes_ledger_hashes_to_domains() -> None:
    app_js = (_DASHBOARD / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "from \"./cockpit_view.js\"" in app_js
    assert "renderLedgerBody" in app_js
    assert "wireCockpitSheet" in app_js
    assert "startsWith(\"ledger-\")" in app_js
    # Legacy hash anchors must not be emitted as domain hrefs (ignore comments).
    code_lines = [
        line for line in app_js.splitlines() if not line.strip().startswith("//")
    ]
    assert not any('href="#ledger-' in line for line in code_lines)
    assert "data-ledger-jump" in (_DASHBOARD / "static" / "js" / "cockpit_view.js").read_text(
        encoding="utf-8"
    )


def test_cockpit_view_uses_learn_ui_and_drawer() -> None:
    view = (_DASHBOARD / "static" / "js" / "cockpit_view.js").read_text(encoding="utf-8")
    assert 'from "./learn_ui.js"' in view
    assert "openCockpitDrawer" in view
    assert "callout(" in view
    assert "codeBlock(" in view
    assert "calcBlock(" in view
    assert "whatToChange(" in view
    assert "/api/domains" not in view or "never" in view.lower()

    index = (_DASHBOARD / "templates" / "index.html").read_text(encoding="utf-8")
    assert "cockpit_view.js" in index
    assert 'id="cockpit-drawer"' in index

    offline = (_DASHBOARD / "offline.py").read_text(encoding="utf-8")
    assert "cockpit_view.js" in offline
    assert "cockpit-drawer" in offline
    assert "#buildml-cockpit-view" in offline


def test_anti_overlap_css_hardens_led_rows() -> None:
    app_css = (_DASHBOARD / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert "grid-template-columns: minmax(0, 42%) minmax(0, 58%)" in app_css
    assert "text-overflow: ellipsis" in app_css
    assert ".om-led__key" in app_css
    assert ".cell-clip" in app_css
    # Teaching / evidence prose must wrap — never ellipsis-truncate.
    assert ".assumption-card__evidence" in app_css
    assert "white-space: normal !important" in app_css
    assert ".table-wrap" in app_css
    assert "table-layout: fixed" in app_css
    assert ".cell-wrap--prose" in app_css
    assert ".col-chips" in app_css

    assert "minmax(0, 42%)" in COCKPIT_CSS
    assert "text-overflow: ellipsis" in COCKPIT_CSS
    assert ".om-led__key" in COCKPIT_CSS
    assert "bml-table--fit" in COCKPIT_CSS
    assert "bml-cell-wrap" in COCKPIT_CSS
    assert "bml-col-chips" in COCKPIT_CSS


def test_header_offline_html_is_primary_export() -> None:
    index = (_DASHBOARD / "templates" / "index.html").read_text(encoding="utf-8")
    assert "Offline HTML" in index
    assert 'id="html-download"' in index
    assert "PDF briefing" not in index
    assert 'id="csv-export"' not in index
    assert "btn-primary blueprint" in index

    app_js = (_DASHBOARD / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "Offline HTML" in app_js
    assert "PDF briefing" not in app_js
    assert 'href="/api/export/csv/findings"' not in app_js
    assert 'href="/api/export/html"' in app_js
    assert "btn-primary blueprint" in app_js

    view = (_DASHBOARD / "static" / "js" / "cockpit_view.js").read_text(encoding="utf-8")
    assert "assumption-card__prose" in view
    assert "table--register" in app_js or "table--fit" in app_js
    assert "col-chips" in app_js


def test_static_ledger_rows_carry_title_tooltips() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    groups = build_ledger_groups(report, report.get("findings") or [])
    html = _ledger_block(groups)
    assert 'title="' in html
    assert "om-led__key" in html
    assert "om-led__val" in html


def test_api_cockpit_exposes_teaching(tmp_path_factory=None) -> None:
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    report = _session().eda(include_plots=False, show=False)
    set_state(
        DashboardState(
            report=report,
            report_dict=report.to_dict(),
            title="Cockpit UX test",
            session_meta={"has_split": True},
        )
    )
    try:
        client = TestClient(create_app())
        cockpit = client.get("/api/cockpit").json()
        sheet = cockpit["sheet"]
        assert sheet["ledger"][0]["teaching"]["worked_example"]["code"]
        assert sheet["ledger_purpose"]["purpose"]
        # Ledger keys must not be domain boards.
        for group in sheet["ledger"]:
            key = group["key"]
            resp = client.get(f"/api/domains/ledger-{key}")
            assert resp.status_code == 404
            assert "Unknown domain" in resp.json()["detail"]
        # Real domain still works.
        assert client.get("/api/domains/quality").status_code == 200
        js = client.get("/static/js/cockpit_view.js")
        assert js.status_code == 200
        assert "data-ledger-jump" in js.text
    finally:
        clear_state()
