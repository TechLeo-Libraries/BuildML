"""Polars LazyFrame native handle + collect-on-promote."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import DataMode, EngineName
from buildml.data.engines import get_engine
from buildml.ingest.detect import available_engines


def _write_csv(path: Path, n: int = 10) -> Path:
    pd.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) * 0.25 for i in range(n)],
            "y": ([0, 1] * (n // 2)),
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_lazy_path_keeps_lazyframe_until_promote(tmp_path: Path) -> None:
    import polars as pl

    path = _write_csv(tmp_path / "lazy.csv")
    session = Session.ingest(path, engine="polars", mode="lazy")
    assert session.dataset.engine == EngineName.POLARS
    assert session.dataset.has_native
    assert session.dataset.has_lazy_native
    assert session.dataset.mode == DataMode.LAZY
    assert session.dataset.pandas_stale is True
    assert isinstance(session.dataset.native, pl.LazyFrame)
    warnings = session.ingest_report.warnings if session.ingest_report else []
    assert any("LazyFrame" in w and "not out-of-core" in w for w in warnings)

    projected = session.dataset.project(["a", "y"])
    assert projected.has_lazy_native
    assert isinstance(projected.native, pl.LazyFrame)

    frame = session.to_pandas()
    assert len(frame) == 10
    assert list(frame.columns) == ["a", "b", "y"]
    assert session.dataset.pandas_stale is False
    # Promotion collects; native may remain LazyFrame until sync, but frame is live.
    assert isinstance(session.dataset.frame, pd.DataFrame)


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_engine_collects_lazyframe_on_to_pandas() -> None:
    import polars as pl

    engine = get_engine("polars")
    lazy = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).lazy()
    assert engine.is_lazy_handle(lazy)
    assert engine.n_rows(lazy) == 3
    assert engine.columns(lazy) == ["a", "b"]
    projected = engine.select_columns(lazy, ["a"])
    assert engine.is_lazy_handle(projected)
    out = engine.to_pandas(projected)
    assert list(out.columns) == ["a"]
    assert len(out) == 3
