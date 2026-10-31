"""Readiness gates retain independently counted evidence across saved report forms."""

import copy
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.gates import build_gate_context, build_gates_payload
from buildml.dashboard.serialize import json_safe


@pytest.fixture(
    scope="module",
    params=[
        ("classification", False),
        ("classification", True),
        ("regression", False),
        ("regression", True),
    ],
)
def evidence(request):
    task, messy = request.param
    rng = np.random.default_rng(687)
    x, z = rng.normal(size=(2, 48))
    y = 1.6 * x + 0.4 * z + rng.normal(0, 0.2, 48)
    if task == "classification":
        y = np.where(y > 0.5, "positive", "negative")
    frame = pd.DataFrame(
        {"x": x, "z": z, "category": [f"kind_{i % 17}" for i in range(48)], "target": y}
    )
    if messy:
        frame.loc[::5, "x"] = np.nan
        frame.loc[::7, "category"] = None
        frame = pd.concat([frame, frame.iloc[:2]], ignore_index=True)
    s = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "category": "feature", "target": "target"}
    )
    report = s.eda(include_plots=False, show=False).to_dict()
    return report, frame, task


@pytest.mark.parametrize("replay", ["memory", "json", "legacy_scalar_vif", "legacy_nested_vif"])
def test_readiness_evidence_counts_and_human_boundaries_survive_report_replay(evidence, replay):
    source, frame, task = evidence
    report = copy.deepcopy(source)
    baseline = build_gate_context(source)
    if replay.startswith("legacy"):
        report["multivariate"]["vif"] = {
            row["name"]: (row["vif"] if replay.endswith("scalar_vif") else {"vif": row["vif"]})
            for row in baseline["vif"]
        }
        report["quality"]["high_cardinality_columns"] = [
            {"name": row["name"], "distinct": row["distinct"]} for row in baseline["highCard"]
        ]
        report["bivariate"]["top_abs_pearson_pairs"] = [
            {"left": row["a"], "right": row["b"], "correlation": row["r"]}
            for row in baseline["corrPairs"]
        ]
        report["bivariate"]["mutual_information_vs_target"] = {
            row.get("feature", row.get("column")): row.get("mi", 0) for row in baseline["mi"]
        }
    if replay != "memory":
        report = json.loads(json.dumps(json_safe(report), allow_nan=False))
    before = copy.deepcopy(report)
    context = build_gate_context(report)
    assert context["rows"] == context["rowsTotal"] == len(frame)
    assert context["missingCells"] == int(frame.isna().sum().sum())
    assert context["duplicates"]["rows"] == int(frame.duplicated().sum())
    expected_missing = {c: float(frame[c].isna().mean()) for c in frame if frame[c].isna().any()}
    assert {row["name"]: row["missingRate"] for row in context["missing"]} == pytest.approx(
        expected_missing
    )
    for row in context["profiledNumeric"]:
        assert row["min"] == pytest.approx(frame[row["name"]].min())
        assert row["max"] == pytest.approx(frame[row["name"]].max())
    if task == "classification":
        assert {
            row["label"]: row["count"] for row in context["target"]["classes"]
        } == frame.target.value_counts().to_dict()
        assert context["target"]["labeledRows"] == len(frame)
    else:
        assert context["target"]["stats"]["median"] == pytest.approx(frame.target.median())
    for key in ("vif", "corrPairs", "highCard"):
        assert context[key] == baseline[key]
    payload = build_gates_payload(report)
    statuses = Counter(row["status"] for row in payload["rows"])
    for status in ("clear", "open", "human", "na"):
        assert payload["counts"][status] == statuses[status]
    assert payload["counts"]["total"] == sum(statuses.values())
    rows = {row["id"]: row for row in payload["rows"]}
    assert rows["00.1"]["status"] == "human"
    assert rows["05.4"]["status"] == "open"
    assert "observational" in rows["05.4"]["evidence"]
    assert payload["persistence"]["human_decisions"] is False
    assert report == before
