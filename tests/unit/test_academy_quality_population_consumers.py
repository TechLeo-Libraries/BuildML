"""Academy consumers preserve full-frame quality, unavailable evidence and zeros."""

import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.academy import build_academy_payload
from buildml.dashboard.academy_curriculum import build_academy_context
from buildml.dashboard.academy_curriculum.note_binder import lesson_from_note
from buildml.dashboard.serialize import json_safe

TRANSPORTS = ["direct", "json", "offline", "api"]
KINDS = ["clean", "flagged", "regression", "ignored"]


@pytest.fixture(scope="module", params=KINDS)
def observation(request):
    kind = request.param
    rng = np.random.default_rng(677)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=120),
            "z": rng.normal(size=120),
            "category": [f"level_{i % (4 if kind == 'clean' else 90)}" for i in range(120)],
            "fixed": 2,
            "near": [0] * 119 + [1],
            "target": np.linspace(-5, 5, 120) if kind == "regression" else ["yes", "no"] * 60,
        }
    )
    if kind != "clean":
        frame.loc[::3, "x"] = np.nan
        frame.loc[::5, "z"] = np.nan
        frame = pd.concat([frame, frame.iloc[:6]], ignore_index=True)
    roles = {name: "ignore" for name in frame.columns}
    if kind != "ignored":
        roles.update(x="feature", z="feature", category="feature", target="target")
    session = Session.ingest(frame).set_roles(roles)
    report = session.eda(
        include_plots=False, show=False, sample_rows=None if kind == "clean" else 16
    )
    raw = json.loads(json.dumps(json_safe(report.to_dict()), allow_nan=False))
    return report, raw, frame, kind


def _consume(report, raw, transport):
    before = copy.deepcopy(raw)
    if transport == "json":
        payload = build_academy_payload(json.loads(json.dumps(raw, allow_nan=False)))
    elif transport == "offline":
        pytest.importorskip("plotly")
        from buildml.dashboard.offline import build_offline_bundle

        bundle = build_offline_bundle(raw)
        payload = bundle["domains"]["academy"]
        assert json.loads(json.dumps(bundle, allow_nan=False))["domains"]["academy"] == payload
    elif transport == "api":
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
                assert client.get("/api/domains/academy").json() == payload
        finally:
            clear_state()
    else:
        payload = build_academy_payload(raw)
    assert raw == before
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    return payload, {entry["slug"]: entry for entry in payload["concepts"]}


def _walkthrough(entry):
    return entry["sections"]["calculation"]["walkthrough"]


def _evidence(entry):
    return entry["sections"]["evidence"]["session"]


def _denominator(text):
    match = re.search(r"(?:^|[ ,])cells\s*[=~]\s*([\d,]+)", text)
    assert match, text
    return int(match[1].replace(",", ""))


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_observed_academy_quality_matches_independent_full_frame_counts(observation, transport):
    report, raw, frame, _ = observation
    payload, lessons = _consume(report, raw, transport)
    missing = int(frame.isna().to_numpy().sum())
    expected = 1.0 - missing / frame.size
    assert payload["context"]["missingCells"] == missing
    assert payload["context"]["completeness"] == pytest.approx(expected)
    assert _denominator(_walkthrough(lessons["missing-data"])) == frame.size
    assert f"{expected:.1%}" in _walkthrough(lessons["missing-data"])
    duplicates = int(frame.duplicated().sum())
    duplicate_evidence = _evidence(lessons["duplicate-records"])
    assert f"{duplicates:,}" in duplicate_evidence
    assert f"{len(frame):,}" in duplicate_evidence
    assert "does not establish key uniqueness" in duplicate_evidence
    if missing:
        assert "Nothing to fill" not in lessons["missing-data"]["sections"]["decide"]
    else:
        assert "No missing cells" in _evidence(lessons["missing-data"])


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize(
    "variant", ["absent", "none", "unknown-count", "legacy-counts", "legacy-score", "missing-scope"]
)
def test_partial_saved_quality_never_invents_clean_data_or_a_sample_denominator(
    observation, transport, variant
):
    report, original, frame, _ = observation
    raw = copy.deepcopy(original)
    if variant == "absent":
        raw.pop("quality")
    elif variant == "none":
        raw["quality"] = None
    elif variant == "unknown-count":
        raw["quality"]["missing_cell_count"] = None
        raw["quality"]["completeness_score"] = None
    elif variant == "legacy-counts":
        raw["quality"].pop("completeness_score")
    elif variant == "legacy-score":
        raw["quality"]["missing_cell_count"] = None
    else:
        raw["quality"]["completeness_score"] = None
        for name in ["n_rows", "n_columns", "columns"]:
            raw["overview"].pop(name, None)
    payload, lessons = _consume(report, raw, transport)
    known_score = variant in {"legacy-counts", "legacy-score"}
    if known_score:
        assert payload["context"]["completeness"] == pytest.approx(
            1.0 - int(frame.isna().to_numpy().sum()) / frame.size
        )
    else:
        assert payload["context"]["completeness"] is None
        assert "unavailable" in _walkthrough(lessons["missing-data"]).lower()
        assert "100.0%" not in _walkthrough(lessons["missing-data"])
    if variant in {"absent", "none", "unknown-count", "legacy-score"}:
        assert payload["context"]["missingCells"] is None
        assert "unavailable" in _evidence(lessons["missing-data"]).lower()
        assert "Nothing to fill" not in lessons["missing-data"]["sections"]["decide"]
    if variant == "legacy-counts":
        assert _denominator(_walkthrough(lessons["missing-data"])) == frame.size


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize(
    "field,slug",
    [
        ("duplicate_row_count", "duplicate-records"),
        ("constant_columns", "constant-and-near-constant"),
        ("quasi_constant_columns", "constant-and-near-constant"),
        ("high_cardinality_columns", "high-cardinality"),
    ],
)
def test_one_unmeasured_quality_screen_stays_unavailable_in_academy(
    observation, transport, field, slug
):
    report, original, _, _ = observation
    raw = copy.deepcopy(original)
    raw["quality"][field] = None
    _, lessons = _consume(report, raw, transport)
    assert "unavailable" in _evidence(lessons[slug]).lower()
    assert "unavailable" in _walkthrough(lessons[slug]).lower()
    if field in {"constant_columns", "quasi_constant_columns"}:
        other = "quasi_constant_columns" if field == "constant_columns" else "constant_columns"
        assert f"columns flagged: {len(raw['quality'][other])}" in _evidence(lessons[slug])


@pytest.mark.parametrize(
    "slug",
    [
        "missing-data",
        "class-imbalance",
        "mutual-information",
        "variance-inflation",
        "data-splitting",
        "cross-validation",
        "dataset-drift",
        "feature-scaling",
    ],
)
def test_public_catalog_lesson_exports_use_the_same_observed_scope(observation, slug, tmp_path):
    _, raw, frame, kind = observation
    before = copy.deepcopy(raw)
    context = build_academy_context(raw)
    lesson = lesson_from_note(slug)
    exported = {
        "slug": lesson.slug,
        "evidence": lesson.session_evidence(context),
        "calculation": lesson.calculation(context),
        "decision": lesson.decide(context),
        "read_steps": lesson.read_steps(context),
    }
    path = tmp_path / "lesson.json"
    path.write_text(json.dumps(exported, allow_nan=False), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == exported
    assert raw == before
    if slug == "missing-data":
        assert _denominator(exported["calculation"]) == frame.size
        expected = 1.0 - int(frame.isna().to_numpy().sum()) / frame.size
        assert f"{expected:.1%}" in exported["calculation"]
    if slug == "feature-scaling" and kind == "ignored":
        assert "p≈0" in exported["calculation"]
    if slug == "class-imbalance" and kind in {"regression", "ignored"}:
        assert "N/A" in exported["evidence"]
    assert exported["evidence"] and exported["calculation"]


@pytest.fixture(scope="module")
def zero_observation():
    frame = pd.DataFrame({"x": [np.nan] * 24, "z": [np.nan] * 24})
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature"})
    report = session.eda(include_plots=False, show=False)
    raw = json.loads(json.dumps(json_safe(report.to_dict()), allow_nan=False))
    assert raw["quality"]["completeness_score"] == 0.0
    return report, raw


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("variant", ["measured", "legacy-score", "unknown"])
def test_true_zero_completeness_and_unknown_quality_remain_distinct(
    zero_observation, transport, variant
):
    report, original = zero_observation
    raw = copy.deepcopy(original)
    if variant != "measured":
        raw["quality"]["missing_cell_count"] = None
    if variant == "unknown":
        raw["quality"]["completeness_score"] = None
    payload, lessons = _consume(report, raw, transport)
    if variant == "unknown":
        assert payload["context"]["completeness"] is None
        assert "unavailable" in _walkthrough(lessons["missing-data"]).lower()
    else:
        assert payload["context"]["completeness"] == 0.0
        assert "0.0%" in _walkthrough(lessons["missing-data"])
    assert "Nothing to fill" not in lessons["missing-data"]["sections"]["decide"]


@pytest.mark.parametrize("observation", ["flagged", "ignored"], indirect=True)
@pytest.mark.parametrize("variant", ["observed", "none", "legacy-counts", "legacy-score"])
def test_saved_studio_html_carries_exact_academy_quality_evidence(observation, variant, tmp_path):
    pytest.importorskip("plotly")
    from buildml.dashboard.offline import export_studio_html

    report, original, _, _ = observation
    raw = copy.deepcopy(original)
    if variant == "none":
        raw["quality"] = None
    elif variant == "legacy-counts":
        raw["quality"].pop("completeness_score")
    elif variant == "legacy-score":
        raw["quality"]["missing_cell_count"] = None
    expected, _ = _consume(report, raw, "direct")
    before = copy.deepcopy(raw)
    path = export_studio_html(raw, tmp_path / "studio.html")
    text = path.read_text(encoding="utf-8")
    marker = "window.__BUILDML_OFFLINE__ = "
    embedded, _ = json.JSONDecoder().raw_decode(text[text.index(marker) + len(marker) :])
    actual = embedded["domains"]["academy"]
    assert actual["context"] == expected["context"]
    assert actual["concepts"] == expected["concepts"]
    assert raw == before


def test_quality_scope_metadata_and_explicit_zero_features_survive_payload(observation):
    _, raw, frame, kind = observation
    context = build_academy_context(raw)
    payload = build_academy_payload(raw)
    assert payload["context"]["qualityRows"] == len(frame)
    assert payload["context"]["qualityColumns"] == len(frame.columns)
    assert payload["context"]["rows"] == raw["overview"]["analysis_rows"]
    if kind == "ignored":
        assert context["features"] == []
        assert payload["context"]["eligible"] == 0
