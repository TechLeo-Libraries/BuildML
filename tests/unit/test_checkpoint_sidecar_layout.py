"""Checkpoint native sidecar compression and partitioned layout."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.checkpoint import bundle as bundle_mod
from buildml.core.types import EngineName
from buildml.ingest.detect import available_engines


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) * 0.1 for i in range(n)],
            "y": ([0, 1] * (n // 2 + 1))[:n],
        }
    )


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_sidecar_single_file_records_compression(tmp_path: Path) -> None:
    session = Session.ingest(_frame(10), engine="polars").set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    ckpt = tmp_path / "ckpt_single"
    session.checkpoint_save(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    sidecar = meta["native_sidecar"]
    assert sidecar["layout"] == "single"
    assert sidecar["compression"] == "zstd"
    assert (ckpt / "data" / "native_sidecar.parquet").exists()

    restored = Session.checkpoint_load(ckpt, trusted=True)
    assert restored.dataset.n_rows == 10
    assert restored.dataset.has_native
    assert any("compression=zstd" in m for m in restored.reattach_result.messages)  # type: ignore[union-attr]


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_sidecar_partitioned_layout_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bundle_mod, "SIDECAR_PARTITION_ROW_THRESHOLD", 8)
    monkeypatch.setattr(bundle_mod, "SIDECAR_ROWS_PER_PARTITION", 5)
    session = Session.ingest(_frame(12), engine="polars").set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    ckpt = tmp_path / "ckpt_parts"
    session.checkpoint_save(ckpt)

    part_dir = ckpt / "data" / "native_sidecar"
    assert part_dir.is_dir()
    parts = sorted(part_dir.glob("*.parquet"))
    assert len(parts) == 3
    assert not (ckpt / "data" / "native_sidecar.parquet").exists()

    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    sidecar = meta["native_sidecar"]
    assert sidecar["layout"] == "partitioned"
    assert sidecar["n_partitions"] == 3
    assert sidecar["compression"] == "zstd"
    assert sidecar["relative_path"] == "data/native_sidecar"

    manifest = json.loads((ckpt / "MANIFEST.json").read_text(encoding="utf-8"))
    part_hashes = [k for k in manifest["hashes"] if k.startswith("data/native_sidecar/")]
    assert len(part_hashes) == 3

    restored = Session.checkpoint_load(ckpt, trusted=True)
    assert restored.dataset.n_rows == 12
    assert restored.dataset.has_native
    assert any("layout=partitioned" in m for m in restored.reattach_result.messages)  # type: ignore[union-attr]
    out = restored.dataset.project(["a", "y"]).to_pandas()
    assert out.shape == (12, 2)


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_legacy_single_sidecar_still_loads(tmp_path: Path) -> None:
    """Older meta without layout/compression still reattaches from single file."""
    session = Session.ingest(_frame(6), engine="duckdb")
    ckpt = tmp_path / "legacy"
    session.checkpoint_save(ckpt)
    meta_path = ckpt / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Simulate older sidecar metadata shape.
    meta["native_sidecar"] = {
        "relative_path": "data/native_sidecar.parquet",
        "format": "parquet",
        "engine": "duckdb",
        "lazy_intent": False,
        "limits": "legacy",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    session.dataset.close_native()

    restored = Session.checkpoint_load(ckpt, trusted=True)
    assert restored.dataset.engine == EngineName.DUCKDB
    assert restored.dataset.has_native
    assert restored.dataset.n_rows == 6
    restored.dataset.close_native()


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_sidecar_layout_force_single(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundle_mod, "SIDECAR_PARTITION_ROW_THRESHOLD", 5)
    session = Session.ingest(_frame(12), engine="polars")
    ckpt = tmp_path / "force_single"
    session.checkpoint_save(
        ckpt,
        sidecar_layout="single",
        sidecar_partition_rows=3,
        sidecar_compression="zstd",
    )
    assert (ckpt / "data" / "native_sidecar.parquet").exists()
    assert not (ckpt / "data" / "native_sidecar").exists()
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    assert meta["native_sidecar"]["layout"] == "single"
    assert meta["native_sidecar"]["compression"] == "zstd"


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_sidecar_layout_force_partitioned(tmp_path: Path) -> None:
    session = Session.ingest(_frame(10), engine="polars")
    ckpt = tmp_path / "force_parts"
    session.checkpoint_save(
        ckpt,
        sidecar_layout="partitioned",
        sidecar_partition_rows=4,
        sidecar_compression="zstd",
    )
    part_dir = ckpt / "data" / "native_sidecar"
    assert part_dir.is_dir()
    parts = sorted(part_dir.glob("*.parquet"))
    assert len(parts) == 3
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    assert meta["native_sidecar"]["layout"] == "partitioned"
    assert meta["native_sidecar"]["rows_per_partition"] == 4
    assert meta["native_sidecar"]["n_partitions"] == 3

    restored = Session.checkpoint_load(ckpt, trusted=True)
    assert restored.dataset.n_rows == 10
    assert restored.dataset.has_native


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_sidecar_invalid_layout_raises(tmp_path: Path) -> None:
    from buildml.core.errors import ValidationError

    session = Session.ingest(_frame(6), engine="polars")
    with pytest.raises(ValidationError, match="sidecar_layout"):
        session.checkpoint_save(tmp_path / "bad", sidecar_layout="hive")
