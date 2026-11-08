"""Temporal dashboard evidence follows the population that EDA profiled."""

import numpy as np
import pandas as pd

from buildml import Session
from buildml.dashboard.gates import build_gate_context, build_gates_payload


def _temporal_gate(report):
    return next(row for row in build_gates_payload(report)["rows"] if row["id"] == "03.5")


def test_temporal_gate_reports_full_train_range_despite_row_sampling():
    event = pd.date_range("2024-03-09", periods=25, freq="6h", tz="America/New_York")
    order = np.random.default_rng(17).permutation(24)
    order = [*order, 24]
    frame = pd.DataFrame(
        {
            "logged_at": pd.date_range("2023-01-01", periods=25, freq="D"),
            "event_at": event[order],
            "measurement": np.arange(25, dtype=float),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"event_at": "time", "measurement": "feature"})
        .inject_split(train_indices=list(range(24)), test_indices=[24])
    )
    report = session.eda(partition="train", sample_rows=5).to_dict()
    assert report["overview"]["n_rows"] == 24
    assert report["overview"]["analysis_rows"] == 5
    expected_min, expected_max = event[0].isoformat(), event[23].isoformat()
    context = build_gate_context(report)["timeCol"]
    assert context == {"name": "event_at", "min": expected_min, "max": expected_max, "gaps": None}
    evidence = _temporal_gate(report)["evidence"]
    assert expected_min in evidence and expected_max in evidence
    assert event[24].isoformat() not in evidence


def test_temporal_gate_reports_unavailable_range_for_unobserved_or_legacy_data():
    frame = pd.DataFrame({"event_at": pd.Series([pd.NaT] * 6, dtype="datetime64[ns]")})
    report = Session.ingest(frame).set_roles({"event_at": "time"}).eda().to_dict()
    assert build_gate_context(report)["timeCol"] == {
        "name": "event_at",
        "min": None,
        "max": None,
        "gaps": None,
    }
    assert "no observed date range" in _temporal_gate(report)["evidence"]

    report["overview"].pop("datetime_ranges", None)
    assert "no observed date range" in _temporal_gate(report)["evidence"]
