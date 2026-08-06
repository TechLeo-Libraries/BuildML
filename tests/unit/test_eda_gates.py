"""Readiness Gates: resolvers, UI-only persistence contract, teaching depth."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.academy import build_academy_payload
from buildml.dashboard.gate_teaching import STATUS_MEANINGS, build_gate_teaching
from buildml.dashboard.gates import GATE_STATUS_LABELS, build_gate_context, build_gates_payload
from buildml.eda.industry_tokens import INDUSTRY_ACCENT, INDUSTRY_ROOT_CSS

pytest.importorskip("fastapi")

_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD = _ROOT / "buildml" / "dashboard"
_FORBIDDEN_DEMO_CLAIMS = (
    "churn dataset",
    "ops-delay",
    "ops delay",
    "this demo always",
    "always use roc_auc for every",
    "telco churn",
)


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


def _regression_session() -> Session:
    frame = pd.DataFrame(
        {
            "sqft": [800.0, 900.5, 1100.2, 1200.1, 1500.0, 1600.7, 1800.3, 2000.4, 2100.0, 2200.5],
            "beds": [1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
            "price": [120.5, 140.2, 180.7, 200.1, 260.4, 280.9, 320.3, 360.6, 390.1, 410.8],
        }
    )
    return (
        Session.ingest(frame)
        .set_roles({"sqft": "feature", "beds": "feature", "price": "target"})
        .split(test_size=0.25, random_state=0)
    )


def test_industry_tokens_shared_with_static_css() -> None:
    from buildml.eda.cockpit_style import COCKPIT_CSS

    assert INDUSTRY_ACCENT == "#5980a6"
    assert "--color-accent: #5980a6" in INDUSTRY_ROOT_CSS
    assert "--color-accent: #5980a6" in COCKPIT_CSS
    tokens = (
        Path(__file__).resolve().parents[2]
        / "buildml"
        / "dashboard"
        / "static"
        / "css"
        / "tokens.css"
    ).read_text(encoding="utf-8")
    assert "#5980a6" in tokens


def test_gates_payload_statuses_and_no_persistence_fields() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    payload = build_gates_payload(report)
    assert payload["counts"]["total"] >= 40
    assert set(GATE_STATUS_LABELS) == {"clear", "open", "human", "na"}
    assert set(payload["counts"]) >= {"clear", "open", "human", "na", "total", "answerable"}
    assert payload["groups"]
    assert payload["persistence"] == {
        "human_decisions": False,
        "session_api": False,
        "disk": False,
        "reason": "ui_only_privacy",
    }
    assert payload["teaching"]["persistence"] is False
    assert payload["teaching"]["adaptive"] is True
    # Payload must not invent a place to store human marks.
    blob = str(payload).lower()
    assert "recorded_decision" not in blob
    assert "saved_decision" not in blob
    for row in payload["rows"]:
        assert row["status"] in GATE_STATUS_LABELS
        assert "evidence" in row and "closes" in row
        assert "session_mark_eligible" in row
        teaching = row["teaching"]
        assert teaching["completeness"]["persistence_claimed"] is False
        assert teaching["completeness"]["adaptive"] is True
        assert teaching["completeness"]["has_worked_example"] is True
        assert "Session" in teaching["worked_example"]["code"]
        assert teaching["worked_example"]["change_these"]
        assert set(STATUS_MEANINGS) <= set(teaching["status_meanings"])
        for needle in _FORBIDDEN_DEMO_CLAIMS:
            assert needle not in str(teaching).lower()


def test_gate_teaching_depth_and_adaptive_examples() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    payload = build_gates_payload(report)
    missing = next(row for row in payload["rows"] if row["id"] == "01.2")
    teaching = missing["teaching"]
    assert "missing" in teaching["beginner"].lower() or "gappy" in teaching["beginner"].lower()
    assert teaching["why_before_modeling"]
    assert teaching["how_derived"]
    assert teaching["levels"]["beginner"]
    assert teaching["levels"]["advanced"]
    assert teaching["calculation"] is not None
    assert "age" in teaching["worked_example"]["code"] or "income" in teaching["worked_example"]["code"]
    assert "y" in teaching["worked_example"]["code"]
    assert teaching["adaptivity"]["target"] == "y"
    assert teaching["adaptivity"]["task"] == "classification"
    assert teaching["next_checks"]

    # Second frame: teaching must adapt to *this* target/columns (not the prior session).
    other = build_gates_payload(_regression_session().eda(include_plots=False, show=False).to_dict())
    other_base = next(row for row in other["rows"] if row["id"] == "04.2")
    assert other_base["teaching"]["adaptivity"]["target"] == "price"
    assert "price" in other_base["teaching"]["worked_example"]["code"]
    assert "sqft" in other_base["teaching"]["worked_example"]["code"]
    assert other_base["teaching"]["adaptivity"]["target"] != missing["teaching"]["adaptivity"]["target"]
    # No redesign demo-dataset slogans in either payload.
    for row in other["rows"]:
        blob = str(row["teaching"]).lower()
        for needle in _FORBIDDEN_DEMO_CLAIMS:
            assert needle not in blob


def test_gate_teaching_builder_marks_are_ephemeral_language() -> None:
    ctx = build_gate_context(_session().eda(include_plots=False, show=False).to_dict())
    teaching = build_gate_teaching(
        gate_id="00.1",
        concept="problem-framing",
        status="human",
        evidence="example evidence",
        closes="one sentence",
        ctx=ctx,
    )
    blob = " ".join(
        [
            teaching["session_mark_note"],
            teaching["status_meanings"]["session_mark"],
            teaching["worked_example"]["code"],
        ]
    ).lower()
    assert "never" in blob or "tab" in blob
    assert "persist" in blob or "save" in blob
    assert teaching["completeness"]["persistence_claimed"] is False


def test_gate_context_reads_report_quality() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    ctx = build_gate_context(report)
    assert ctx["rows"] >= 1
    assert ctx["colCount"] >= 1
    assert "const" in ctx["constants"]
    assert ctx["missingCells"] >= 1
    assert ctx["target"]["name"] == "y"


def test_academy_stages_and_cited_flags() -> None:
    report = _session().eda(include_plots=False, show=False).to_dict()
    academy = build_academy_payload(report)
    assert academy["concept_count"] >= 204
    assert academy.get("catalog_count") == 204
    assert academy.get("extended_count", 0) == 0
    assert academy["stages"]
    assert any(stage["key"] == 0 for stage in academy["stages"])
    assert any(stage["key"] == 6 for stage in academy["stages"])
    assert "cited_count" in academy
    sample = academy["concepts"][0]
    assert "prose" in sample
    assert "read" in sample
    assert "example" in sample
    assert "pitfalls" in sample
    assert "session" in sample
    assert sample.get("curriculum") is True


def test_dashboard_modules_have_no_gate_decision_write_apis() -> None:
    root = Path(__file__).resolve().parents[2] / "buildml" / "dashboard"
    forbidden = (
        "save_gate",
        "persist_gate",
        "record_gate",
        "gate_decision",
        "write_gate",
        "store_gate",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for needle in forbidden:
            assert needle not in lower, f"{path} mentions {needle}"
        # AST: no route decorator that accepts POST/PUT/PATCH on gates.
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for deco in node.decorator_list:
                src = ast.dump(deco)
                if "gates" in src.lower() and any(
                    method in src for method in ("post", "put", "patch", "delete")
                ):
                    raise AssertionError(f"Mutable gates route in {path}: {node.name}")


def test_app_js_keeps_gate_marks_in_memory_only() -> None:
    js_dir = _DASHBOARD / "static" / "js"
    app_js = (js_dir / "app.js").read_text(encoding="utf-8")
    gates_js = (js_dir / "gates_view.js").read_text(encoding="utf-8")
    assert "gateSessionMarks" in app_js
    assert "from \"./gates_view.js\"" in app_js
    assert "renderGatesView" in app_js
    assert "Mark for this session" in gates_js
    assert "openGateDrawer" in gates_js
    assert "from \"./learn_ui.js\"" in gates_js
    assert "localStorage.setItem" in app_js  # theme only
    assert 'localStorage.setItem("buildml-eda-gates"' not in app_js
    assert "localStorage" not in gates_js
    assert "sessionStorage" not in gates_js
    for line in app_js.splitlines():
        if "localStorage.setItem" in line:
            assert "gate" not in line.lower() or "theme" in line.lower()


def test_gates_view_and_css_wired_in_templates() -> None:
    index = (_DASHBOARD / "templates" / "index.html").read_text(encoding="utf-8")
    assert "gates.css" in index
    assert 'id="gate-drawer"' in index
    assert "gates_view.js" not in index or "app.js" in index  # ESM via app import
    assert (_DASHBOARD / "static" / "css" / "gates.css").is_file()
    assert (_DASHBOARD / "gate_teaching.py").is_file()
    offline = (_DASHBOARD / "offline.py").read_text(encoding="utf-8")
    assert "gates_view.js" in offline
    assert "gates.css" in offline
    assert "gate-drawer" in offline
