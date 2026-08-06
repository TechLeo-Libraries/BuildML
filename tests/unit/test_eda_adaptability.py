"""Cross-App dataset adaptability: payloads bind to live report, not demo schema."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.adapt import (
    FORBIDDEN_REQUIRED_COLUMNS,
    assert_no_demo_column_requirements,
    build_adapt_context,
    session_sentence,
    what_to_change,
)
from buildml.dashboard.sheet import build_cockpit_sheet

pytest.importorskip("fastapi")


def _frame_session(
    *,
    target: str = "label",
    extra: dict[str, list] | None = None,
) -> Session:
    data = {
        "num_a": [10, 20, None, 40, 50, 60, 70, 80],
        "cat_b": ["x", "y", "x", "y", "x", "z", "y", "x"],
        "const_c": [1] * 8,
        "row_id": list(range(8)),
        target: [0, 1, 0, 1, 0, 1, 0, 1],
    }
    if extra:
        data.update(extra)
    frame = pd.DataFrame(data)
    roles = {
        "num_a": "feature",
        "cat_b": "feature",
        "const_c": "feature",
        "row_id": "id",
        target: "target",
    }
    return Session.ingest(frame).set_roles(roles).split(
        test_size=0.25, stratify=True, random_state=0
    )


def test_adapt_context_uses_live_target_and_task() -> None:
    report = _frame_session(target="approved").eda(include_plots=False, show=False).to_dict()
    ctx = build_adapt_context(report)
    assert ctx["target_column"] == "approved"
    assert ctx["task"] == "classification"
    assert ctx["has_target"] is True
    assert "approved" in ctx["session_sentence"]
    assert "num_a" in ctx["columns"] or "num_a" in ctx["eligible_features"]
    assert "target_churn" not in ctx["columns"]
    assert ctx["n_rows"] == report["overview"]["n_rows"]
    assert isinstance(ctx["analyzers"], list)
    assert ctx["analyzers"]


def test_adapt_context_without_target() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["u", "v", "u", "v"]})
    session = Session.ingest(frame).set_roles({"a": "feature", "b": "feature"})
    report = session.eda(include_plots=False, show=False).to_dict()
    ctx = build_adapt_context(report)
    assert ctx["target_column"] is None
    assert ctx["has_target"] is False
    assert "no target" in ctx["session_sentence"].lower()
    assert "target" in {row["family"] for row in ctx["analyzers"]}
    skipped = set(ctx["skipped_analyzers"])
    assert "target" in skipped or "drift" in skipped


def test_cockpit_sheet_adaptive_fields_match_report() -> None:
    report = _frame_session(target="outcome").eda(include_plots=False, show=False).to_dict()
    sheet = build_cockpit_sheet(report)
    adapt = sheet["adapt"]
    assert adapt["target_column"] == "outcome"
    assert sheet["session_sentence"] == session_sentence(report)
    assert sheet["spine_meta"]["register"]
    assert "finding" in sheet["spine_meta"]["register"]
    assert isinstance(sheet["what_to_change"], list)
    # Focus columns are drawn from this frame, not a churn template.
    focus = sheet["focus_columns"]
    assert focus["target"] == "outcome"
    blob = str(sheet).lower()
    for forbidden in ("target_churn", "monthly_charges", "tenure"):
        # May appear only if the user's data literally used those names.
        assert forbidden not in blob or forbidden in {
            str(c).lower() for c in (adapt.get("columns") or [])
        }


def test_api_payloads_have_no_required_demo_columns() -> None:
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    report = _frame_session(target="y").eda(include_plots=False, show=False)
    payload = report.to_dict()
    set_state(
        DashboardState(
            report=report,
            report_dict=payload,
            title="Adapt Studio",
            session_meta={"has_split": True},
        )
    )
    try:
        client = TestClient(create_app())
        meta = client.get("/api/meta").json()
        cockpit = client.get("/api/cockpit").json()
        assert meta["adapt"]["target_column"] == "y"
        assert cockpit["adapt"]["target_column"] == "y"
        assert cockpit["sheet"]["adapt"]["task"] == "classification"
        assert cockpit["sheet"]["session_sentence"]
        for name, body in (("meta", meta), ("cockpit", cockpit)):
            hits = assert_no_demo_column_requirements(body)
            assert not hits, f"{name} required demo columns: {hits}"
            # Payloads must not require a churn target key.
            assert "target_churn" not in body.get("adapt", {})
        home = client.get("/")
        assert home.status_code == 200
        assert "/static/js/learn_ui.js" in home.text
        learn = client.get("/static/js/learn_ui.js")
        assert learn.status_code == 200
        assert "sectionScaffold" in learn.text
        assert "callout" in learn.text
        css = client.get("/static/css/app.css")
        assert "Learn UI primitives" in css.text
    finally:
        clear_state()


def test_what_to_change_mentions_live_columns() -> None:
    report = _frame_session().eda(include_plots=False, show=False).to_dict()
    items = what_to_change(report, limit=8)
    assert items
    joined = " ".join(f"{row['change']} {row['why']}" for row in items)
    # Constant / id-like from this frame should surface when flagged.
    assert "const_c" in joined or "row_id" in joined or "session." in joined.lower()


def test_forbidden_required_columns_constant_covers_demo_names() -> None:
    assert "target_churn" in FORBIDDEN_REQUIRED_COLUMNS
    assert "monthly_charges" in FORBIDDEN_REQUIRED_COLUMNS
