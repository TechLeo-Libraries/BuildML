"""Checkpoint native sidecar round-trips for optional engines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import DataMode, EngineName
from buildml.ingest.detect import available_engines


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) * 0.1 for i in range(n)],
            "y": ([0, 1] * (n // 2)),
        }
    )


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_checkpoint_sidecar_roundtrip_polars_lazy(tmp_path: Path) -> None:
    import polars as pl

    path = tmp_path / "lazy.csv"
    _frame().to_csv(path, index=False)
    session = (
        Session.ingest(path, engine="polars", mode="lazy")
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    assert session.dataset.has_lazy_native
    ckpt = tmp_path / "ckpt_lazy"
    session.checkpoint_save(ckpt)

    assert (ckpt / "data" / "frame.parquet").exists()
    assert (ckpt / "data" / "native_sidecar.parquet").exists()
    meta = (ckpt / "meta.json").read_text(encoding="utf-8")
    assert "native_sidecar" in meta
    assert "lazy_intent" in meta

    restored = Session.checkpoint_load(ckpt)
    assert restored.dataset.engine == EngineName.POLARS
    assert restored.dataset.mode == DataMode.LAZY
    assert restored.dataset.has_native
    assert restored.dataset.has_lazy_native
    assert isinstance(restored.dataset.native, pl.LazyFrame)
    assert restored.reattach_result is not None
    assert any("sidecar" in m and "scan_parquet" in m for m in restored.reattach_result.messages)
    assert restored.reattach_result.details.get("has_lazy_native") is True
    out = restored.dataset.project(["a", "y"]).to_pandas()
    assert out.shape[1] == 2
    assert len(out) == 12


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_checkpoint_sidecar_roundtrip_duckdb(tmp_path: Path) -> None:
    session = Session.ingest(_frame(8), engine="duckdb").set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    ckpt = tmp_path / "ckpt_duck"
    session.checkpoint_save(ckpt)
    assert (ckpt / "data" / "native_sidecar.parquet").exists()

    restored = Session.checkpoint_load(ckpt)
    assert restored.dataset.engine == EngineName.DUCKDB
    assert restored.dataset.has_native
    assert restored.reattach_result is not None
    assert any("sidecar" in m for m in restored.reattach_result.messages)
    assert restored.dataset.n_rows == 8
    projected = restored.dataset.project(["a", "y"])
    assert projected.has_native
    assert projected.to_pandas().shape == (8, 2)


def test_checkpoint_without_sidecar_stays_backward_compatible(tmp_path: Path) -> None:
    """Older frame.parquet-only bundles still load (no sidecar required)."""
    session = Session.ingest(_frame(6)).set_roles({"a": "feature", "y": "target"})
    ckpt = tmp_path / "ckpt_pd"
    session.checkpoint_save(ckpt)
    assert not (ckpt / "data" / "native_sidecar.parquet").exists()
    restored = Session.checkpoint_load(ckpt)
    assert restored.dataset.engine == EngineName.PANDAS
    assert not restored.dataset.has_native
