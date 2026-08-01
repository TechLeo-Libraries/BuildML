"""DuckDB Arrow/IPC native attach without Pandas Feather bridge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import get_engine
from buildml.ingest.detect import available_engines
from buildml.ingest.native_load import load_native_path


def _write_feather(path: Path, n: int = 8) -> Path:
    frame = pd.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) for i in range(n)],
            "y": ([0, 1] * (n // 2)),
        }
    )
    frame.to_feather(path)
    return path


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_arrow_ingest_uses_pyarrow_bridge(tmp_path: Path) -> None:
    path = _write_feather(tmp_path / "table.feather")
    native, schema, details = load_native_path(
        path, engine=EngineName.DUCKDB, format_name="arrow"
    )
    assert details.get("pandas_first") is False
    assert details.get("pandas_bridge") is False
    assert details.get("arrow_bridge") in {"pyarrow_feather", "pyarrow_ipc"}
    assert details.get("n_rows") == 8
    assert [f.name for f in schema.fields] == ["a", "b", "y"]

    session = Session.ingest(path, engine="duckdb")
    assert session.dataset.has_native
    assert session.dataset.engine == EngineName.DUCKDB
    assert session.dataset.n_rows == 8
    native_details = (session.ingest_report.details or {}).get("native_load", {})
    assert native_details.get("arrow_bridge") in {"pyarrow_feather", "pyarrow_ipc"}
    assert native_details.get("pandas_bridge") is False
    out = session.to_pandas()
    assert list(out.columns) == ["a", "b", "y"]
    assert len(out) == 8
    del native


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_engine_from_pandas_prefers_arrow() -> None:
    engine = get_engine("duckdb")
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    rel = engine.from_pandas(frame)
    assert getattr(engine, "_bridge", None) == "arrow"
    out = engine.to_pandas(rel)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), frame)
    mask = [True, False, True]
    filtered = engine.filter_rows(rel, mask)
    filtered_pd = engine.to_pandas(filtered)
    assert len(filtered_pd) == 2
    # Prefer SQL mask pushdown; Arrow filter remains a fallback only.
    assert getattr(engine, "_bridge", None) in {"sql", "arrow"}
    assert getattr(engine, "_last_pushdown", None) in {
        "sql_mask",
        "arrow_filter_fallback",
    }


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_ipc_stream_path(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.table({"a": [1, 2, 3], "y": [0, 1, 0]})
    path = tmp_path / "stream.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    native, _schema, details = load_native_path(
        path, engine=EngineName.DUCKDB, format_name="arrow"
    )
    assert details["arrow_bridge"] in {"pyarrow_feather", "pyarrow_ipc"}
    assert details["n_rows"] == 3
    engine = get_engine("duckdb")
    assert engine.n_rows(native) == 3
