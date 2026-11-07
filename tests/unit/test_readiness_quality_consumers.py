"""Public readiness consumers distinguish unmeasured quality from observed zeros."""

import copy
import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.gates import build_gates_payload
from buildml.dashboard.serialize import json_safe


@pytest.fixture(scope="module", params=["clean", "flagged", "sampled"])
def observation(request):
    kind = request.param
    n = 80
    rng = np.random.default_rng(731)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "entity": [f"entity_{i % 10}" for i in range(n)],
            "category": [f"level_{i % (64 if kind != 'clean' else 24)}" for i in range(n)],
            "target": ["yes", "no"] * (n // 2),
        }
    )
    if kind != "clean":
        frame["constant"] = 2
        frame["near_constant"] = [0] * (n - 1) + [1]
        frame.loc[::4, "x"] = np.nan
        frame = pd.concat([frame, frame.iloc[:6]], ignore_index=True)
    session = Session.ingest(frame).set_roles(
        {
            "x": "feature",
            "z": "feature",
            "category": "feature",
            "target": "target",
        }
    )
    report = session.eda(
        include_plots=False, show=False, sample_rows=24 if kind == "sampled" else None
    )
    return report, frame, kind


def _consume(report, transport):
    raw = report.to_dict()
    before = copy.deepcopy(raw)
    if transport == "json":
        result = build_gates_payload(json.loads(json.dumps(json_safe(raw), allow_nan=False)))
    elif transport == "offline":
        pytest.importorskip("plotly")
        from buildml.dashboard.offline import build_offline_bundle

        bundle = build_offline_bundle(raw)
        result = bundle["gates"]
        assert bundle["domains"]["gates"]["gates"] == result
        assert json.loads(json.dumps(bundle, allow_nan=False))["gates"] == result
    elif transport == "api":
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from buildml.dashboard.app import create_app
        from buildml.dashboard.state import DashboardState, clear_state, set_state

        set_state(DashboardState(report=report, report_dict=raw))
        try:
            with TestClient(create_app()) as client:
                response = client.get("/api/gates")
                assert response.status_code == 200
                result = response.json()
                domain = client.get("/api/domains/gates")
                assert domain.status_code == 200
                assert domain.json()["gates"] == result
                assert client.post("/api/gates", json={"01.4": "clear"}).status_code == 405
                assert client.get("/api/gates").json() == result
        finally:
            clear_state()
    else:
        result = build_gates_payload(raw)
    assert raw == before
    counts = Counter(row["status"] for row in result["rows"])
    assert all(result["counts"][key] == counts[key] for key in ("clear", "open", "human", "na"))
    assert result["counts"]["total"] == sum(counts.values())
    assert result["counts"]["answerable"] == sum(v for k, v in counts.items() if k != "na")
    expected = round(100 * counts["clear"] / max(1, result["counts"]["answerable"]))
    assert result["settled_pct"] == f"{expected}%"
    assert result["persistence"]["human_decisions"] is False
    return {row["id"]: row for row in result["rows"]}


@pytest.mark.parametrize("transport", ["memory", "json", "offline", "api"])
def test_observed_quality_uses_full_population_and_does_not_assert_key_uniqueness(
    observation, transport
):
    report, frame, kind = observation
    quality = report.quality
    missing = sum(
        value is None or pd.isna(value)
        for row in frame.itertuples(index=False, name=None)
        for value in row
    )
    # A scalar row comparison supplies an oracle independent of the EDA counters.
    normalized = list(frame.fillna("<missing>").itertuples(index=False, name=None))
    duplicates = sum(row in normalized[:i] for i, row in enumerate(normalized))
    assert quality["missing_cell_count"] == missing
    assert quality["duplicate_row_count"] == duplicates
    assert frame.entity.nunique() < len(frame)  # No key uniqueness screen was requested.
    rows = _consume(report, transport)
    assert rows["01.2"]["status"] == ("open" if missing else "clear")
    assert rows["01.3"]["status"] == ("open" if missing else "na")
    assert rows["01.4"]["status"] == ("open" if duplicates else "clear")
    assert "no repeated keys" not in rows["01.4"]["evidence"].lower()
    assert "key uniqueness" in rows["01.4"]["evidence"].lower()
    assert f"{len(frame):,}" in rows["01.4"]["evidence"]
    if kind == "sampled":
        assert report.overview["analysis_rows"] < len(frame)
    constants = [name for name in frame if len(set(frame[name].dropna())) <= 1]
    assert rows["01.5"]["status"] == ("open" if constants else "clear")
    high = [
        name
        for name in ["category", "entity"]
        if len(set(frame[name])) > min(1000, max(50, int(0.5 * len(frame))))
    ]
    assert rows["01.6"]["status"] == ("open" if high else "clear")
    assert "20" not in rows["01.6"]["evidence"]
    assert quality["high_cardinality_counts"] == {name: len(set(frame[name])) for name in high}
    if high:
        width = sum(len(set(frame[name])) for name in high)
        calc = rows["01.6"]["teaching"]["calculation"]
        assert calc["inputs"]["estimated_dummy_columns"] == width
        assert calc["result"] == f"{width:,}"


@pytest.mark.parametrize("transport", ["memory", "json"])
@pytest.mark.parametrize("absence", ["omitted", "null"])
@pytest.mark.parametrize(
    "gate,fields",
    [
        ("01.2", ["missing_cell_count"]),
        ("01.3", ["missing_cell_count", "missing_rate_by_column"]),
        ("01.4", ["duplicate_row_count"]),
        ("01.5", ["constant_columns", "quasi_constant_columns"]),
        ("01.6", ["high_cardinality_columns"]),
    ],
)
def test_partial_quality_cannot_settle_an_unmeasured_screen(
    observation, transport, absence, gate, fields
):
    report, _, _ = observation
    quality = copy.deepcopy(report.quality)
    for name in fields:
        if absence == "omitted":
            quality.pop(name)
        else:
            quality[name] = None
    rows = _consume(replace(report, quality=quality), transport)
    row = rows[gate]
    assert row["status"] == "open"
    assert "unavailable" in row["evidence"].lower()
    assert row["session_mark_eligible"] is True
    if gate == "01.2":
        calculation = row["teaching"]["calculation"]
        assert calculation is None or "unavailable" in str(calculation["result"]).lower()
        assert row["teaching"]["adaptivity"]["missing_cells"] is None


@pytest.mark.parametrize("transport", ["json", "offline", "api"])
@pytest.mark.parametrize("quality", [{}, None])
def test_unavailable_quality_survives_public_report_consumers(observation, transport, quality):
    report, _, _ = observation
    rows = _consume(replace(report, quality=quality), transport)
    for gate in ["01.2", "01.3", "01.4", "01.5", "01.6"]:
        assert rows[gate]["status"] == "open"
        assert "unavailable" in rows[gate]["evidence"].lower()
    full = _consume(report, transport)
    assert full["01.2"]["status"] == (
        "clear" if report.quality["missing_cell_count"] == 0 else "open"
    )


@pytest.mark.parametrize("transport", ["memory", "json"])
@pytest.mark.parametrize("missing_field", ["constant_columns", "quasi_constant_columns"])
def test_one_missing_constant_screen_does_not_clear_the_pair(observation, transport, missing_field):
    report, _, _ = observation
    quality = copy.deepcopy(report.quality)
    quality["constant_columns"] = []
    quality["quasi_constant_columns"] = []
    quality.pop(missing_field)
    row = _consume(replace(report, quality=quality), transport)["01.5"]
    assert row["status"] == "open"
    assert "unavailable" in row["evidence"].lower()


@pytest.mark.parametrize("transport", ["memory", "json", "api"])
def test_total_missing_count_without_column_rates_does_not_invent_a_zero_rate(
    observation, transport
):
    report, frame, _ = observation
    quality = copy.deepcopy(report.quality)
    quality.pop("missing_rate_by_column")
    rows = _consume(replace(report, quality=quality), transport)
    cells = int(frame.isna().sum().sum())
    calculation = rows["01.2"]["teaching"]["calculation"]
    if cells:
        assert rows["01.3"]["status"] == "open"
        assert calculation["inputs"]["missing_cells"] == cells
        assert calculation["inputs"]["rows"] == len(frame)
        assert calculation["inputs"]["columns"] == len(frame.columns)
        assert calculation["result"] == f"{100 * cells / frame.size:.1f}%"
        assert "densest" not in calculation["reading"]
    else:
        assert rows["01.3"]["status"] == "na"
        assert calculation["result"] == "0%"


@pytest.mark.parametrize("transport", ["memory", "json"])
def test_saved_report_without_full_cardinality_counts_discloses_sample_limit(
    observation, transport
):
    report, frame, kind = observation
    quality = copy.deepcopy(report.quality)
    quality.pop("high_cardinality_counts", None)
    rows = _consume(replace(report, quality=quality), transport)
    if kind == "sampled":
        assert rows["01.6"]["status"] == "open"
        assert "unavailable" in rows["01.6"]["evidence"].lower()
        assert rows["01.6"]["teaching"]["calculation"] is None
    elif kind == "flagged":
        expected = len(set(frame.category))
        assert (
            rows["01.6"]["teaching"]["calculation"]["inputs"]["estimated_dummy_columns"] == expected
        )
    else:
        assert rows["01.6"]["status"] == "clear"


@pytest.mark.parametrize("transport", ["memory", "json"])
def test_observed_zero_missingness_uses_full_frame_denominator_when_sampled(transport):
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 48), "y": ["yes", "no"] * 24})
    report = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .eda(include_plots=False, show=False, sample_rows=12)
    )
    assert report.overview["analysis_rows"] == 12
    row = _consume(report, transport)["01.2"]
    assert row["status"] == "clear"
    calc = row["teaching"]["calculation"]
    assert calc["inputs"]["rows"] == len(frame)
    assert calc["inputs"]["columns"] == len(frame.columns)
    assert calc["inputs"]["missing_cells"] == 0
    assert calc["result"] == "0%"
