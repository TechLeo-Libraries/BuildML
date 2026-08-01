"""Native Polars/DuckDB handles on Dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import prepare_design_frame
from buildml.ingest.detect import available_engines


def _wide(n: int = 30) -> pd.DataFrame:
    data = {"y": ([0, 1] * (n // 2)), "signal": list(range(n))}
    for i in range(6):
        data[f"noise_{i}"] = [float(i + 1)] * n
    return pd.DataFrame(data)


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_with_engine_attaches_polars_native_and_projects() -> None:
    session = Session.ingest(_wide()).with_engine("polars")
    assert session.dataset.has_native
    assert session.dataset.engine == EngineName.POLARS
    projected = session.dataset.project(["signal", "y"])
    assert projected.has_native
    assert projected.columns == ["signal", "y"]
    assert projected.n_rows == 30
    sampled = projected.sample(n=5, random_state=0)
    assert len(sampled) == 5
    assert list(sampled.columns) == ["signal", "y"]


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_prepare_design_uses_attached_native_handle() -> None:
    session = (
        Session.ingest(_wide())
        .set_roles({"signal": "feature", "y": "target"})
        .with_engine("polars")
    )
    result = prepare_design_frame(session.dataset, ["signal", "y"], sample_rows=8, random_state=1)
    assert result.used_native_handle is True
    assert result.engine == "polars"
    assert result.n_rows_materialized == 8
    assert any("native handle" in tip for tip in result.disclosures)


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_native_filter_and_project() -> None:
    session = Session.ingest(_wide(20)).with_engine("duckdb")
    assert session.dataset.has_native
    mask = [True, False] * 10
    filtered = session.dataset.filter_rows(mask)
    assert filtered.n_rows == 10
    assert filtered.has_native
    projected = filtered.project(["signal"])
    assert projected.columns == ["signal"]
    frame = projected.to_pandas()
    assert list(frame.columns) == ["signal"]
    assert len(frame) == 10


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_to_engine_returns_attached_native_without_roundtrip() -> None:
    session = Session.ingest(_wide(12)).with_engine("polars")
    native = session.dataset.native
    again = session.dataset.to_engine("polars")
    assert again is native
