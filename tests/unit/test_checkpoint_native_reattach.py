"""Checkpoint load restores native engine handles when possible."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.ingest.detect import available_engines


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_checkpoint_roundtrip_reattaches_polars_native(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "y": [0, 1] * 4,
        }
    )
    session = (
        Session.ingest(frame, engine="polars")
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    assert session.dataset.has_native
    path = tmp_path / "ckpt_polars"
    session.checkpoint_save(path)

    restored = Session.checkpoint_load(path)
    assert restored.reattach_result is not None
    assert restored.reattach_result.status == "resume"
    assert restored.dataset.engine == EngineName.POLARS
    assert restored.dataset.has_native
    assert any("Restored polars native handle" in m for m in restored.reattach_result.messages)
    assert any("sidecar" in m for m in restored.reattach_result.messages)
    assert restored.reattach_result.details.get("has_native") is True
    assert (path / "data" / "native_sidecar.parquet").exists()
    projected = restored.dataset.project(["a", "y"])
    assert projected.has_native
    assert projected.to_pandas().shape == (8, 2)


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_checkpoint_roundtrip_reattaches_duckdb_native(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    session = Session.ingest(frame, engine="duckdb").set_roles({"a": "feature", "y": "target"})
    path = tmp_path / "ckpt_duck"
    session.checkpoint_save(path)
    restored = Session.checkpoint_load(path)
    assert restored.dataset.engine == EngineName.DUCKDB
    assert restored.dataset.has_native
    assert any("Restored duckdb native handle" in m for m in restored.reattach_result.messages)  # type: ignore[union-attr]
    assert any("sidecar" in m for m in restored.reattach_result.messages)  # type: ignore[union-attr]
    assert (path / "data" / "native_sidecar.parquet").exists()


def test_checkpoint_pandas_engine_skips_native_reattach(tmp_path: Path) -> None:
    session = Session.ingest(pd.DataFrame({"a": [1, 2], "y": [0, 1]}))
    path = tmp_path / "ckpt_pd"
    session.checkpoint_save(path)
    restored = Session.checkpoint_load(path)
    assert restored.dataset.engine == EngineName.PANDAS
    assert not restored.dataset.has_native
    assert restored.reattach_result is not None
    assert not any("native handle" in m for m in restored.reattach_result.messages)
