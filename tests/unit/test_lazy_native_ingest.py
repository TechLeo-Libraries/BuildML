"""Native-first / deferred-Pandas ingest paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.core.types import DataMode, EngineName
from buildml.ingest.detect import MEMORY_SOFT_LIMIT, available_engines


def _write_csv(path: Path, n: int = 8) -> Path:
    pd.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) * 0.5 for i in range(n)],
            "y": ([0, 1] * (n // 2)),
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_path_ingest_avoids_pandas_first(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "tiny.csv")
    session = Session.ingest(path, engine="polars", mode="lazy")
    assert session.dataset.engine == EngineName.POLARS
    assert session.dataset.has_native
    assert session.dataset.mode == DataMode.LAZY
    assert session.dataset.pandas_stale is True
    assert session.ingest_report is not None
    assert session.ingest_report.details.get("native_load", {}).get("pandas_first") is False
    # Promotion happens at the Pandas boundary.
    frame = session.to_pandas()
    assert len(frame) == 8
    assert list(frame.columns) == ["a", "b", "y"]
    assert session.dataset.pandas_stale is False


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_path_ingest_native(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "duck.csv", n=6)
    session = Session.ingest(path, engine="duckdb")
    assert session.dataset.has_native
    assert session.dataset.engine == EngineName.DUCKDB
    assert session.dataset.n_rows == 6
    projected = session.dataset.project(["a", "y"])
    assert projected.has_native
    assert projected.to_pandas().shape == (6, 2)


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_large_path_uses_native_when_engine_available(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "big.csv", n=4)
    session = Session.ingest(
        path,
        engine="polars",
        mode="lazy",
        mock_byte_estimate=MEMORY_SOFT_LIMIT + 1,
    )
    assert session.dataset.has_native
    warnings = session.ingest_report.warnings if session.ingest_report else []
    assert any("Native-first" in w for w in warnings)


def test_polars_ingest_missing_extra_message(tmp_path: Path) -> None:
    if EngineName.POLARS in available_engines():
        pytest.skip("polars installed in this environment")
    path = _write_csv(tmp_path / "x.csv")
    with pytest.raises(MissingExtraError, match="buildml\\[polars\\]"):
        Session.ingest(path, engine="polars")
