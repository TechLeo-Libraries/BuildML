"""Polars filter_expr parity with DuckDB (optional polars)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import get_engine
from buildml.ingest.detect import available_engines


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_filter_expr_keeps_lazyframe(tmp_path: Path) -> None:
    import polars as pl

    path = tmp_path / "src.csv"
    pd.DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0], "y": [0, 1, 0, 1]}).to_csv(
        path, index=False
    )
    session = Session.ingest(path, engine="polars", mode="lazy")
    assert session.dataset.has_lazy_native

    filtered = session.dataset.filter_expr("a > 2")
    assert filtered.has_native
    assert filtered.pandas_stale is True
    assert isinstance(filtered.native, pl.LazyFrame)
    out = filtered.to_pandas()
    assert list(out["a"]) == [3, 4]


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_engine_filter_expr_eager() -> None:
    engine = get_engine("polars")
    frame = engine.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    narrowed = engine.filter_expr(frame, "a >= 2")
    out = engine.to_pandas(narrowed)
    assert list(out["a"]) == [2, 3]
