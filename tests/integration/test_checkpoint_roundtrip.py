from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def test_checkpoint_save_load_restores_roles_and_splits(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8], "y": [0, 1, 0, 1, 0, 1, 0, 1]})
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    path = tmp_path / "ckpt"
    session.checkpoint_save(path)

    assert (path / "MANIFEST.json").exists()
    assert (path / "data" / "frame.parquet").exists()

    restored = Session.checkpoint_load(path, trusted=True)
    assert restored.reattach_result is not None
    assert restored.reattach_result.status == "resume"
    assert restored.dataset.roles["y"].value == "target"
    assert restored.split_plan is not None
    assert restored.split_plan.train_indices == session.split_plan.train_indices  # type: ignore[union-attr]
    assert restored.metadata()["dataset"]["n_rows"] == 8


def test_data_only_reattach_is_fresh_ingest(tmp_path: Path) -> None:
    session = Session.ingest(pd.DataFrame({"a": [1, 2], "y": [0, 1]}))
    path = tmp_path / "ckpt"
    session.checkpoint_save(path)

    restored = Session.checkpoint_load(path, data_only=True, trusted=True)
    assert restored.reattach_result is not None
    assert restored.reattach_result.status == "fresh_ingest"
    assert restored.split_plan is None
    assert restored.dataset.roles == {}


def test_removed_column_blocks_reattach(tmp_path: Path) -> None:
    session = Session.ingest(pd.DataFrame({"a": [1, 2], "b": [3, 4], "y": [0, 1]}))
    session.set_roles({"a": "feature", "y": "target"})
    path = tmp_path / "ckpt"
    session.checkpoint_save(path)

    # Simulate external edit that drops a column that existed at checkpoint time.
    broken = pd.read_parquet(path / "data" / "frame.parquet").drop(columns=["a"])
    broken.to_parquet(path / "data" / "frame.parquet", index=False)

    with pytest.raises(ValidationError, match="required column"):
        Session.checkpoint_load(path, trusted=True)


def test_row_change_invalidates_splits(tmp_path: Path) -> None:
    session = (
        Session.ingest(pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]}))
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.25, random_state=1)
    )
    path = tmp_path / "ckpt"
    session.checkpoint_save(path)

    altered = pd.read_parquet(path / "data" / "frame.parquet")
    altered = pd.concat([altered, pd.DataFrame({"a": [99], "y": [1]})], ignore_index=True)
    altered.to_parquet(path / "data" / "frame.parquet", index=False)

    restored = Session.checkpoint_load(path, trusted=True)
    assert restored.reattach_result is not None
    assert restored.reattach_result.status == "splits_invalidated"
    assert restored.split_plan is None


def test_manifest_contains_version_and_hashes(tmp_path: Path) -> None:
    import json

    import buildml

    session = Session.ingest(pd.DataFrame({"a": [1], "y": [0]}))
    path = tmp_path / "ckpt"
    session.checkpoint_save(path)
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["buildml_version"] == buildml.__version__
    assert "data/frame.parquet" in manifest["hashes"]
    assert len(manifest["hashes"]["meta.json"]) == 64
