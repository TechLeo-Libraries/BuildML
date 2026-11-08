"""Checkpoint resume may only read members within the selected artifact."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture
def frames(tmp_path):
    canonical = pd.DataFrame({"x": range(12), "y": range(12, 24)})
    outside = tmp_path / "independent.parquet"
    canonical.assign(x=canonical.x + 100).to_parquet(outside, index=False)
    return canonical, outside


def _rewrite_metadata(bundle, metadata):
    meta_path = bundle / "meta.json"
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest["hashes"]["meta.json"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _check_canonical_only(bundle, canonical):
    restored = Session.checkpoint_load(bundle, data_only=True)
    pd.testing.assert_frame_equal(restored.dataset._ensure_pandas(), canonical)


def _symlink(member, outside):
    try:
        member.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Cannot create symlinks on this filesystem: {exc}")


@pytest.mark.parametrize("absolute", [False, True])
def test_manifest_cannot_hash_an_external_file(tmp_path, frames, absolute):
    canonical, outside = frames
    bundle = Session.ingest(canonical).checkpoint_save(tmp_path / "bundle")
    path = bundle / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    member = str(outside) if absolute else f"../{outside.name}"
    manifest["hashes"][member] = hashlib.sha256(outside.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="member|checkpoint"):
        Session.checkpoint_load(bundle)
    _check_canonical_only(bundle, canonical)


def test_manifest_symlink_cannot_hash_an_external_file(tmp_path, frames):
    canonical, outside = frames
    bundle = Session.ingest(canonical).checkpoint_save(tmp_path / "bundle")
    link = bundle / "linked.parquet"
    _symlink(link, outside)
    path = bundle / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["hashes"][link.name] = hashlib.sha256(outside.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="member|checkpoint"):
        Session.checkpoint_load(bundle)
    _check_canonical_only(bundle, canonical)


def test_data_only_does_not_follow_a_canonical_file_symlink_outside_bundle(tmp_path, frames):
    canonical, outside = frames
    bundle = Session.ingest(canonical).checkpoint_save(tmp_path / "bundle")
    data = bundle / "data" / "frame.parquet"
    data.unlink()
    _symlink(data, outside)
    with pytest.raises(ValidationError, match="member|checkpoint"):
        Session.checkpoint_load(bundle, data_only=True)


def test_explicit_bare_parquet_path_can_be_loaded_without_a_bundle(frames):
    canonical, outside = frames
    pd.testing.assert_frame_equal(
        Session.checkpoint_load(outside, data_only=True).dataset._ensure_pandas(),
        canonical.assign(x=canonical.x + 100),
    )


def test_data_only_preserves_relative_source_labels(tmp_path, frames, monkeypatch):
    canonical, outside = frames
    bundle = Session.ingest(canonical).checkpoint_save(tmp_path / "bundle")
    monkeypatch.chdir(tmp_path)

    for requested, expected in (
        (Path(bundle.name), Path(bundle.name) / "data" / "frame.parquet"),
        (Path(outside.name), Path(outside.name)),
    ):
        restored = Session.checkpoint_load(requested, data_only=True)
        assert restored.dataset.source == str(expected)


@pytest.mark.parametrize("engine", ["duckdb", "polars"])
@pytest.mark.parametrize("absolute", [False, True])
def test_native_metadata_cannot_replace_the_canonical_rows(tmp_path, frames, engine, absolute):
    pytest.importorskip(engine)
    canonical, outside = frames
    session = Session.ingest(canonical, engine=engine)
    bundle = session.checkpoint_save(tmp_path / "bundle", sidecar_layout="single")
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert not meta["has_plans"]
    meta["native_sidecar"]["relative_path"] = str(outside) if absolute else f"../{outside.name}"
    _rewrite_metadata(bundle, meta)
    with pytest.raises(ValidationError, match="sidecar|member|checkpoint"):
        Session.checkpoint_load(bundle)
    _check_canonical_only(bundle, canonical)


@pytest.mark.parametrize("layout", ["single", "partitioned"])
def test_legacy_native_layout_without_member_names_restores_canonical_rows(
    tmp_path, frames, layout
):
    pytest.importorskip("duckdb")
    canonical, _ = frames
    session = Session.ingest(canonical, engine="duckdb")
    bundle = session.checkpoint_save(
        tmp_path / "bundle",
        sidecar_layout=layout,
        sidecar_partition_rows=4,
    )
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    sidecar = meta["native_sidecar"]
    del sidecar["relative_path"]
    sidecar.pop("parts", None)
    _rewrite_metadata(bundle, meta)
    loaded = Session.checkpoint_load(bundle)
    assert loaded.dataset.has_native
    pd.testing.assert_frame_equal(loaded.dataset._ensure_pandas(), canonical)


def test_legacy_partition_cannot_link_to_external_rows(tmp_path, frames):
    pytest.importorskip("duckdb")
    canonical, outside = frames
    session = Session.ingest(canonical, engine="duckdb")
    bundle = session.checkpoint_save(
        tmp_path / "bundle",
        sidecar_layout="partitioned",
        sidecar_partition_rows=4,
    )
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    sidecar = meta["native_sidecar"]
    member = bundle / sidecar["relative_path"] / sidecar["parts"][0]
    member.unlink()
    _symlink(member, outside)
    sidecar.pop("parts")
    _rewrite_metadata(bundle, meta)
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["hashes"][member.relative_to(bundle).as_posix()]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="member|checkpoint"):
        Session.checkpoint_load(bundle)
    _check_canonical_only(bundle, canonical)
