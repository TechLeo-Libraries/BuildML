"""Native columnar previews pass row budgets to the reader before materialization."""
from __future__ import annotations

import pytest

from buildml.core.errors import IngestError
from buildml.core.types import EngineName
from buildml.ingest.native_load import load_native_path


@pytest.mark.parametrize("format_name", ["parquet", "arrow"])
@pytest.mark.parametrize("lazy", [False, True])
def test_columnar_preview_passes_row_limit_to_real_reader(format_name, lazy, tmp_path, monkeypatch):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"row": list(range(40)), "text": ["orchard", None] * 20})
    path = tmp_path / ("data.parquet" if format_name == "parquet" else "data.arrow")
    getattr(frame, "write_parquet" if format_name == "parquet" else "write_ipc")(path)
    method = "read_parquet" if format_name == "parquet" else "read_ipc"
    original_reader = getattr(pl, method)
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(pl, method, read)
    table, schema, details = load_native_path(path, engine=EngineName.POLARS,
                                             format_name=format_name, nrows=5, lazy=lazy)
    assert calls == [{"n_rows": 5}]
    assert isinstance(table, pl.DataFrame)
    assert table.to_dicts() == frame.head(5).to_dicts()
    assert [field.name for field in schema.fields] == ["row", "text"]
    assert details["n_rows"] == 5
    assert details["pandas_first"] is False
    assert details["lazy_handle"] is False


@pytest.mark.parametrize("format_name", ["parquet", "arrow"])
@pytest.mark.parametrize("lazy", [False, True])
def test_unlimited_columnar_load_preserves_requested_native_handle(format_name, lazy, tmp_path):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"row": [3, 1, 2], "text": ["pear", None, "apple"]})
    path = tmp_path / ("data.parquet" if format_name == "parquet" else "data.arrow")
    getattr(frame, "write_parquet" if format_name == "parquet" else "write_ipc")(path)
    table, _, details = load_native_path(path, engine=EngineName.POLARS,
                                         format_name=format_name, lazy=lazy)
    assert isinstance(table, pl.LazyFrame) is lazy
    assert details["lazy_handle"] is lazy
    assert details["lazy_scan"] is lazy
    actual = table.collect() if lazy else table
    assert actual.to_dicts() == frame.to_dicts()


@pytest.mark.parametrize("format_name", ["parquet", "arrow"])
def test_corrupt_columnar_input_is_an_ingest_error(format_name, tmp_path):
    pytest.importorskip("polars")
    path = tmp_path / "broken.columnar"
    path.write_bytes(b"invalid file contents")
    with pytest.raises(IngestError, match="Polars native load failed"):
        load_native_path(path, engine=EngineName.POLARS, format_name=format_name)
