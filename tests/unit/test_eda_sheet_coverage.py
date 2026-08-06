"""Cockpit readiness-sheet payload must expose full EDA coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.charts import charts_for_cockpit_report
from buildml.dashboard.sheet import build_cockpit_sheet
from buildml.eda.html_report import _ledger_groups, _methods_catalog
from buildml.eda.sheet_coverage import (
    build_domain_briefs,
    build_ledger_groups,
    build_methods_catalog,
)

pytest.importorskip("fastapi")


def _rich_session() -> Session:
    frame = pd.DataFrame(
        {
            "num_a": [10, 20, None, 40, 50, 60, 70, 80, 90, 100, 110, 5],
            "num_b": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.2, 0.5],
            "cat_b": ["x", "y", "x", "y", "x", "z", "y", "x", "z", "y", "x", "y"],
            "const_c": [1] * 12,
            "row_id": list(range(12)),
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    roles = {
        "num_a": "feature",
        "num_b": "feature",
        "cat_b": "feature",
        "const_c": "feature",
        "row_id": "id",
        "label": "target",
    }
    return (
        Session.ingest(frame)
        .set_roles(roles)
        .split(test_size=0.25, stratify=True, random_state=0)
    )


def test_cockpit_sheet_full_coverage_payload() -> None:
    report = _rich_session().eda(include_plots=False, show=False).to_dict()
    sheet = build_cockpit_sheet(report)
    coverage = sheet["coverage"]

    assert coverage["register"] == len(report.get("findings") or [])
    assert coverage["register"] >= 1
    assert coverage["assumptions"] >= 1
    assert coverage["ledger_groups"] >= 8
    assert coverage["ledger_items"] >= 30
    assert coverage["sequence"] >= 1
    assert coverage["domain_briefs"] >= 4
    assert coverage["figures"] >= 6
    assert coverage["methods"] >= 8
    assert coverage["methods_ran"] >= 5

    # Deep assumptions (not one-liner only).
    note = sheet["assumptions"][0]
    assert note.get("means")
    assert note.get("matters")
    assert note.get("technical") is not None

    ledger_titles = {group["title"] for group in sheet["ledger"]}
    assert "Frame" in ledger_titles
    assert "Quality flags" in ledger_titles
    assert any("Mutual information" in title for title in ledger_titles)
    assert "Target & screens" in ledger_titles
    assert sheet["ledger_purpose"]["purpose"]
    assert sheet["assumptions_purpose"]["purpose"]
    assert sheet["ledger"][0].get("teaching", {}).get("worked_example", {}).get("code")

    assert sheet["methods"]
    assert sheet["domain_briefs"]
    assert sheet["degraded"] is not None
    assert sheet["chart_ids"] == charts_for_cockpit_report(report)
    assert "severity_map" in sheet["chart_ids"]

    # Dataset-adaptive: no demo churn schema baked in.
    blob = str(sheet).lower()
    assert "target_churn" not in blob
    assert "monthly_charges" not in blob


def test_shared_ledger_matches_static_and_app() -> None:
    report = _rich_session().eda(include_plots=False, show=False).to_dict()
    shared = build_ledger_groups(report, report.get("findings") or [])
    static = _ledger_groups(report, report.get("findings") or [])
    assert [g["title"] for g in shared] == [g["title"] for g in static]
    assert len(shared) >= 8

    methods = build_methods_catalog(report)
    assert methods == _methods_catalog(report)
    assert any(card["status"] == "ran" for card in methods)

    briefs = build_domain_briefs(report)
    keys = {brief["key"] for brief in briefs}
    assert {"quality", "features", "relationships", "target"}.issubset(keys)


def test_cockpit_api_sheet_coverage_rich() -> None:
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    report = _rich_session().eda(include_plots=False, show=False)
    payload = report.to_dict()
    set_state(
        DashboardState(
            report=report,
            report_dict=payload,
            title="Coverage Studio",
            session_meta={"has_split": True},
        )
    )
    try:
        client = TestClient(create_app())
        body = client.get("/api/cockpit").json()
        sheet = body["sheet"]
        assert sheet["coverage"]["ledger_items"] >= 30
        assert len(sheet["domain_briefs"]) >= 4
        assert len(sheet["methods"]) >= 8
        assert len(body["chart_ids"]) == len(sheet["chart_ids"])
        assert len(body["chart_ids"]) >= 6
        assert sheet["spine_meta"]["domains"]
        assert "ran" in sheet["spine_meta"]["methods"]
        assert "groups" in sheet["spine_meta"]["ledger"]
        assert "numbers" in sheet["spine_meta"]["ledger"]
    finally:
        clear_state()
