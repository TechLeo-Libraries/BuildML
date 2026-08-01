"""DuckDB connection ownership and reuse on Dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import get_engine
from buildml.data.engines.duckdb_engine import DuckDBEngine, DuckDBTable
from buildml.ingest.detect import available_engines


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_get_engine_does_not_open_connection_per_call() -> None:
    a = get_engine("duckdb")
    b = get_engine("duckdb")
    assert a is b
    assert isinstance(a, DuckDBEngine)
    assert getattr(a, "_con", None) is None


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_dataset_reuses_duckdb_connection_across_ops() -> None:
    session = Session.ingest(
        pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]}),
        engine="duckdb",
    )
    ds = session.dataset
    assert isinstance(ds.native, DuckDBTable)
    assert ds._owns_native_connection is True
    con = ds.native.connection

    projected = ds.project(["a", "y"])
    assert isinstance(projected.native, DuckDBTable)
    assert projected.native.connection is con
    assert projected._owns_native_connection is False

    filtered = ds.filter_expr('"a" > 1')
    assert isinstance(filtered.native, DuckDBTable)
    assert filtered.native.connection is con
    assert filtered._owns_native_connection is False

    ds.sync_native()
    assert isinstance(ds.native, DuckDBTable)
    assert ds.native.connection is con
    assert ds._owns_native_connection is True

    # Derived close must not tear down the owner connection.
    projected.close_native()
    assert con is not None
    out = ds.to_pandas()
    assert list(out["a"]) == [1, 2, 3, 4]

    ds.close_native()
    assert ds.native is None
    assert ds._owns_native_connection is False


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_close_native_is_safe_to_repeat() -> None:
    session = Session.ingest(pd.DataFrame({"a": [1], "y": [0]}), engine="duckdb")
    session.dataset.close_native()
    session.dataset.close_native()
    assert session.dataset.native is None


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_dataset_context_manager_closes_owned_connection() -> None:
    from buildml.data.engines.duckdb_engine import DuckDBTable

    session = Session.ingest(pd.DataFrame({"a": [1, 2], "y": [0, 1]}), engine="duckdb")
    ds = session.dataset
    assert isinstance(ds.native, DuckDBTable)
    con = ds.native.connection
    with ds:
        assert ds.has_native
        assert ds.native.connection is con
    assert ds.native is None
    assert ds._owns_native_connection is False


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_session_context_manager_closes_owned_connection() -> None:
    with Session.ingest(pd.DataFrame({"a": [1], "y": [0]}), engine="duckdb") as session:
        assert session.dataset.has_native
        assert session.dataset._owns_native_connection is True
    assert session.dataset.native is None
    assert session.dataset._owns_native_connection is False


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_dataset_context_manager_closes_on_exception() -> None:
    session = Session.ingest(pd.DataFrame({"a": [1], "y": [0]}), engine="duckdb")
    ds = session.dataset
    with pytest.raises(RuntimeError, match="boom"):
        with ds:
            raise RuntimeError("boom")
    assert ds.native is None
