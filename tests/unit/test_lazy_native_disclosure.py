"""Lazy-native / engine status disclosure in walkthrough and Teaching Studio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.dashboard.teaching import build_teaching_studios
from buildml.ingest.detect import available_engines


def _frame(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) for i in range(n)],
            "y": ([0, 1] * (n // 2)),
        }
    )


def test_walkthrough_engine_status_pandas() -> None:
    session = Session.ingest(_frame()).set_roles({"a": "feature", "b": "feature", "y": "target"})
    report = session.walkthrough()
    status = report.engine_status
    assert status["engine"] == "pandas"
    assert status["has_native"] is False
    assert status["has_lazy_native"] is False
    assert any("Pandas-backed" in note for note in status["disclosures"])
    payload = report.to_dict()
    assert "engine_status" in payload
    assert payload["engine_status"]["has_lazy_native"] is False


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_walkthrough_and_eda_disclose_lazy_native(tmp_path: Path) -> None:
    path = tmp_path / "lazy.csv"
    _frame().to_csv(path, index=False)
    session = Session.ingest(path, engine="polars", mode="lazy").set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    walk = session.walkthrough()
    assert walk.engine_status["has_lazy_native"] is True
    assert any("collect-on-promote" in note for note in walk.engine_status["disclosures"])
    html = walk.export_html(tmp_path / "walk.html").read_text(encoding="utf-8")
    assert "has_lazy_native" in html
    assert "Engine and lazy-native status" in html

    eda = session.eda(include_plots=False)
    overview = eda.to_dict()["overview"]
    assert overview["has_lazy_native"] is True
    assert overview["has_native"] is True
    assert overview["engine_disclosures"]
    assert any(
        "collect-on-promote" in note.lower() for note in overview["engine_disclosures"]
    )

    studios = build_teaching_studios(eda.to_dict())
    cockpit = studios["cockpit"]
    assert cockpit["worked_example"]["values"]["has_lazy_native"] is True
    assert any("has_lazy_native" in line for line in cockpit["interpretation"])
    assert "engine-choice" in cockpit["concepts"]
