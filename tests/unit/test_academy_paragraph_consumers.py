"""Academy prose stays as paragraphs through catalog, search and report consumers."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.academy import build_academy_payload
from buildml.dashboard.academy_curriculum import curriculum_by_slug
from buildml.dashboard.serialize import json_safe

SOURCE = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures" / "academy_prose_declarations.json"
    ).read_text(encoding="utf-8")
)
ORACLE = SOURCE["declarations"]
SLUGS = sorted({row["slug"] for row in ORACLE})


@pytest.fixture(scope="module", params=["classification", "regression"])
def observation(request):
    frame = pd.DataFrame(
        {
            "measurement": np.linspace(-2, 8, 48),
            "sensor": np.sin(np.arange(48)),
            "category": ["east", "west"] * 24,
            "outcome": ["yes", "no"] * 24
            if request.param == "classification"
            else np.linspace(12, 40, 48),
        }
    )
    session = Session.ingest(frame).set_roles(
        {"measurement": "feature", "sensor": "feature", "category": "feature", "outcome": "target"}
    )
    report = session.eda(include_plots=False, show=False, sample_rows=16)
    return report, json.loads(json.dumps(json_safe(report.to_dict()), allow_nan=False))


@pytest.fixture(scope="module")
def declarations():
    # Public catalog objects consume the same declarations as the app.
    return curriculum_by_slug()


@pytest.mark.parametrize("slug", SLUGS)
def test_public_lesson_specs_keep_declared_paragraphs(declarations, slug):
    spec = declarations[slug]
    for row in (item for item in ORACLE if item["slug"] == slug):
        value = getattr(spec, row["field"])
        if callable(value):
            value = value({})  # These oracle fields are static declarations.
        assert list(value) == [row["paragraph"]], row["section"]


@pytest.mark.parametrize("row", SOURCE["sequence_controls"], ids=lambda row: row["field"])
def test_existing_paragraph_sequences_keep_order_and_independent_views(declarations, row):
    value = getattr(declarations[row["slug"]], row["field"])
    if callable(value):
        paragraphs = value({})
        assert paragraphs == row["paragraphs"]
        paragraphs.append("Caller-owned note")
        assert value({}) == row["paragraphs"]
    else:
        assert list(value) == row["paragraphs"]


def _assert_paragraphs(payload):
    entries = {entry["slug"]: entry for entry in payload["concepts"]}
    mismatches = []
    for row in ORACLE:
        value = entries[row["slug"]]["sections"]
        for key in row["section"].split("/"):
            value = value[key]
        if value != [row["paragraph"]]:
            mismatches.append(f"{row['slug']}/{row['section']}")
    assert not mismatches, f"{len(mismatches)} broken paragraph fields: {mismatches[:8]}"
    assert len(entries) == payload["concept_count"]
    return entries


@pytest.mark.parametrize("transport", ["direct", "json", "offline", "api"])
def test_all_declared_paragraphs_survive_actual_report_consumers(observation, transport):
    report, raw = observation
    before = copy.deepcopy(raw)
    if transport == "api":
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from buildml.dashboard.app import create_app
        from buildml.dashboard.state import DashboardState, clear_state, set_state

        set_state(DashboardState(report=report, report_dict=raw))
        try:
            with TestClient(create_app()) as client:
                response = client.get("/api/domains/academy")
                assert response.status_code == 200
                payload = response.json()
        finally:
            clear_state()
    elif transport == "offline":
        pytest.importorskip("plotly")
        from buildml.dashboard.offline import build_offline_bundle

        payload = build_offline_bundle(raw)["domains"]["academy"]
    else:
        source = json.loads(json.dumps(raw, allow_nan=False)) if transport == "json" else raw
        payload = build_academy_payload(source)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    _assert_paragraphs(payload)
    assert raw == before


def test_complete_paragraphs_remain_searchable_and_prose_stays_readable(observation):
    _, raw = observation
    payload = build_academy_payload(raw)
    entries = {entry["slug"]: entry for entry in payload["concepts"]}
    missing = [
        f"{row['slug']}/{row['field']}"
        for row in ORACLE
        if row["paragraph"] not in entries[row["slug"]]["search"]
    ]
    assert not missing, f"Search lost {len(missing)} complete paragraphs: {missing[:8]}"
    for row in (item for item in ORACLE if item["field"] in {"plain", "technical"}):
        entry = entries[row["slug"]]
        assert row["paragraph"] in entry["prose"]


def test_standalone_session_artifact_preserves_all_paragraphs(observation, tmp_path):
    pytest.importorskip("plotly")
    from buildml.dashboard.offline import export_studio_html

    _, raw = observation
    before = copy.deepcopy(raw)
    path = export_studio_html(raw, tmp_path / "academy.html")
    text = path.read_text(encoding="utf-8")
    literal = text.split("window.__BUILDML_OFFLINE__ = ", 1)[1]
    bundle, end = json.JSONDecoder().raw_decode(literal)
    assert literal[end:].lstrip().startswith(";")
    _assert_paragraphs(bundle["domains"]["academy"])
    assert bundle["report"] == before == raw
    # A usable saved payload also retains the independent report scope.
    assert bundle["meta"]["overview"]["n_rows"] == 48
    assert bundle["meta"]["overview"]["analysis_rows"] == 16
