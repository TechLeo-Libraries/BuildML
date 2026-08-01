"""DuckDB filter/sample/projection without full Arrow materialization."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import get_engine
from buildml.ingest.detect import available_engines


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_filter_rows_uses_sql_mask_pushdown() -> None:
    engine = get_engine("duckdb")
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    rel = engine.from_pandas(frame)
    filtered = engine.filter_rows(rel, [True, False, True, False, True])
    assert engine._last_pushdown == "sql_mask"
    assert engine._bridge == "sql"
    out = engine.to_pandas(filtered)
    assert list(out["a"]) == [1, 3, 5]


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_filter_expr_and_sample_pushdown() -> None:
    engine = get_engine("duckdb")
    frame = pd.DataFrame({"a": list(range(20)), "b": [float(i) for i in range(20)]})
    rel = engine.from_pandas(frame)
    narrowed = engine.filter_expr(rel, '"a" >= 10')
    assert engine._last_pushdown == "filter_expr"
    sampled = engine.sample_rows(narrowed, 5, random_state=None)
    assert engine._last_pushdown in {"using_sample", "order_random_limit"}
    out = engine.to_pandas(sampled)
    assert len(out) == 5
    assert out["a"].min() >= 10


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_seeded_sample_is_relation_native() -> None:
    engine = get_engine("duckdb")
    rel = engine.from_pandas(pd.DataFrame({"a": list(range(30)), "b": list(range(30))}))
    sampled = engine.sample_rows(rel, 7, random_state=11)
    assert engine._last_pushdown == "hash_order_limit"
    assert len(engine.to_pandas(sampled)) == 7


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_dataset_filter_expr_keeps_native_stale() -> None:
    session = Session.ingest(
        pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]}),
        engine="duckdb",
    )
    filtered = session.dataset.filter_expr('"a" > 2')
    assert filtered.has_native
    assert filtered.pandas_stale is True
    out = filtered.to_pandas()
    assert list(out["a"]) == [3, 4]
