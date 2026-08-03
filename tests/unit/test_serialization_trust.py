"""Trust gate for pickle/joblib deserialization + integrity helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.anomaly.checkpoint import load_anomaly_bundle, save_anomaly_bundle
from buildml.anomaly.results import AnomalyPlan
from buildml.checkpoint.bundle import load_checkpoint
from buildml.core.errors import ValidationError
from buildml.core.serialization import (
    assert_local_load_path,
    joblib_load_trusted,
    read_json_sidecar,
    require_trusted_deserialize,
    sha256_file,
    verify_sha256,
)
from buildml.pipeline.bundle import load_pipeline_bundle


def _tiny_session() -> Session:
    frame = pd.DataFrame(
        {"x": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], "y": [0, 1, 0, 1, 0, 1]}
    )
    session = Session.ingest(frame).set_roles({"y": "target"})
    session.split(test_size=0.34, stratify=True)
    session.impute()
    session.fit(LogisticRegression())
    return session


def test_require_trusted_deserialize_blocks_by_default(tmp_path: Path) -> None:
    target = tmp_path / "payload.joblib"
    target.write_text("not-a-real-joblib", encoding="utf-8")
    with pytest.raises(ValidationError, match="trusted=True"):
        require_trusted_deserialize(trusted=False, artifact="test", path=target)


def test_assert_local_load_path_refuses_uris() -> None:
    with pytest.raises(ValidationError, match="URI"):
        assert_local_load_path("https://evil.example/model.joblib")
    with pytest.raises(ValidationError, match="file URI"):
        assert_local_load_path("file:///tmp/model.joblib")
    with pytest.raises(ValidationError, match="s3"):
        assert_local_load_path("s3://bucket/model.joblib")


def test_joblib_load_trusted_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "obj.joblib"
    joblib.dump({"ok": 1}, path)
    with pytest.raises(ValidationError):
        joblib_load_trusted(path, trusted=False)
    loaded = joblib_load_trusted(path, trusted=True, artifact="unit test")
    assert loaded == {"ok": 1}


def test_joblib_load_trusted_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "obj.joblib"
    joblib.dump({"ok": 1}, path)
    with pytest.raises(ValidationError, match="Integrity check failed"):
        joblib_load_trusted(path, trusted=True, expected_sha256="0" * 64)
    digest = sha256_file(path)
    loaded = joblib_load_trusted(path, trusted=True, expected_sha256=digest)
    assert loaded == {"ok": 1}


def test_verify_sha256_noop_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "obj.bin"
    path.write_bytes(b"abc")
    verify_sha256(path, None)
    verify_sha256(path, "")


def test_read_json_sidecar(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text('{"format": "x", "n": 1}', encoding="utf-8")
    payload = read_json_sidecar(meta)
    assert payload["format"] == "x"
    with pytest.raises(ValidationError, match="Missing"):
        read_json_sidecar(tmp_path / "missing.json")


def test_load_checkpoint_plans_require_trusted(tmp_path: Path) -> None:
    session = _tiny_session()
    ckpt = tmp_path / "ckpt"
    session.checkpoint_save(ckpt)
    assert (ckpt / "plans.joblib").is_file()
    data_only = load_checkpoint(ckpt, data_only=True, trusted=False)
    assert data_only.plans == {}
    with pytest.raises(ValidationError, match="trusted=True"):
        load_checkpoint(ckpt, trusted=False)
    restored = load_checkpoint(ckpt, trusted=True)
    assert restored.dataset is not None


def test_checkpoint_manifest_hash_mismatch_refuses(tmp_path: Path) -> None:
    session = _tiny_session()
    ckpt = tmp_path / "ckpt-hash"
    session.checkpoint_save(ckpt)
    manifest_path = ckpt / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_key = next(iter(manifest["hashes"]))
    manifest["hashes"][first_key] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="Integrity check failed"):
        load_checkpoint(ckpt, trusted=True)


def test_session_checkpoint_load_forwards_trusted(tmp_path: Path) -> None:
    session = _tiny_session()
    ckpt = tmp_path / "sess-ckpt"
    session.checkpoint_save(ckpt)
    with pytest.raises(ValidationError, match="trusted=True"):
        Session.checkpoint_load(ckpt)
    restored = Session.checkpoint_load(ckpt, trusted=True)
    assert restored.dataset is not None
    plain = Session.checkpoint_load(ckpt, data_only=True)
    assert plain.dataset is not None


def test_pipeline_bundle_requires_trusted(tmp_path: Path) -> None:
    session = _tiny_session()
    pipe = tmp_path / "pipe"
    session.save_pipeline(pipe)
    with pytest.raises(ValidationError, match="trusted=True"):
        load_pipeline_bundle(pipe)
    bundle = load_pipeline_bundle(pipe, trusted=True)
    assert bundle.fit_result is not None


def test_pipeline_bundle_hash_mismatch_refuses(tmp_path: Path) -> None:
    session = _tiny_session()
    pipe = tmp_path / "pipe-hash"
    session.save_pipeline(pipe)
    meta_path = pipe / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "payload_hashes" in meta
    meta["payload_hashes"]["model.joblib"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValidationError, match="Integrity check failed"):
        load_pipeline_bundle(pipe, trusted=True)


def test_anomaly_bundle_requires_trusted(tmp_path: Path) -> None:
    plan = AnomalyPlan(
        method="isolation_forest",
        mode="unsupervised",
        backend="sklearn",
        columns=("a", "b"),
        n_train_rows=10,
        n_fit_rows=10,
        threshold_policy="contamination",
        threshold_=0.5,
        contamination=0.1,
        train_alert_rate_=0.1,
        train_score_stats_={"min": 0.0, "max": 1.0},
        flag_column="is_anomaly",
        score_column="anomaly_score",
        estimator_=object(),
    )
    path = save_anomaly_bundle(tmp_path / "anomaly", plan)
    meta = read_json_sidecar(path / "meta.json")
    assert "payload_sha256" in meta
    with pytest.raises(ValidationError, match="trusted=True"):
        load_anomaly_bundle(path)
    loaded = load_anomaly_bundle(path, trusted=True)
    assert loaded.method == "isolation_forest"
    (path / "anomaly_plan.joblib").write_bytes(b"not-the-original")
    with pytest.raises(ValidationError, match="Integrity check failed"):
        load_anomaly_bundle(path, trusted=True)
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 1.1]})
    path2 = save_anomaly_bundle(tmp_path / "anomaly2", plan)
    with pytest.raises(ValidationError, match="trusted=True"):
        Session.ingest(frame).load_anomaly_bundle(path2)
    Session.ingest(frame).load_anomaly_bundle(path2, trusted=True)
