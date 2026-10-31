"""Replacement checkpoints preserve one data/split/plan snapshot for consumers."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.core.errors import ValidationError

BACKINGS = ["pandas", "polars-memory", "polars-lazy", "duckdb-memory", "duckdb-lazy"]
NATIVE = BACKINGS[1:]


def _session(tmp_path, backing, *, generation=2, plans="none"):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    i = np.arange(48, dtype=float)
    frame = pd.DataFrame({"row_id": [f"{n:04}" for n in range(len(i))],
                          "x": i * .7 + generation * 30,
                          "z": np.sin(i / 3) + generation * 4,
                          "y": i * .3 + np.cos(i / 2) + generation})
    if plans != "none":
        frame.loc[frame.index[::7], "x"] = np.nan
    if backing.endswith("lazy"):
        source = tmp_path / f"source-{generation}.parquet"
        frame.to_parquet(source, index=False)
        session = Session.ingest(source, engine=engine, mode="lazy")
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles({"row_id": "id", "x": "feature", "z": "feature", "y": "target"})
    session.split(test_size=.25, validation_size=.25, random_state=19 + generation)
    expected = frame.copy()
    train = list(session.split_plan.train_indices)
    if plans != "none":
        median = float(np.nanmedian(frame.iloc[train]["x"]))
        session.impute(columns=["x"], strategy="median")
        expected["x"] = expected["x"].fillna(median)
        assert session.impute_plan.statistics_["x"] == pytest.approx(median)
    if plans in {"standard", "minmax"}:
        values = expected[["x", "z"]].to_numpy()
        training = values[train]
        if plans == "standard":
            center, divisor = training.mean(axis=0), training.std(axis=0)
        else:
            center, divisor = training.min(axis=0), np.ptp(training, axis=0)
        session.scale(columns=["x", "z"], method=plans)
        expected[["x", "z"]] = (values - center) / divisor
    return session, expected


def _consumer(session, expected, split):
    """Compare persisted rows, split membership and OLS against scalar inputs."""
    pd.testing.assert_frame_equal(session.to_pandas(), expected, check_dtype=False,
                                  rtol=2e-12, atol=2e-12)
    assert session.reattach_result.status == "resume"
    assert session.split_plan.to_dict() == split
    assert session.fit_result is None  # A checkpoint carries no fitted estimator.
    plans = hashlib.sha256(pickle.dumps((session.impute_plan, session.scale_plan))).hexdigest()
    train = list(session.split_plan.train_indices)
    design = np.column_stack([np.ones(len(expected)), expected[["x", "z"]].to_numpy()])
    beta = np.linalg.lstsq(design[train], expected.iloc[train]["y"], rcond=None)[0]
    session.fit(LinearRegression(), task="regression")
    for partition in ["validation", "test"]:
        indices = list(getattr(session.split_plan, f"{partition}_indices"))
        prediction = design[indices] @ beta
        actual = session.fit_result.estimator.predict(session.partition(partition)[["x", "z"]])
        np.testing.assert_allclose(actual, prediction, rtol=2e-11, atol=2e-11)
        truth = expected.iloc[indices]["y"].to_numpy()
        residual = truth - prediction
        metrics = session.evaluate(partition=partition).metrics
        assert metrics["mae"] == pytest.approx(np.abs(residual).mean(), abs=2e-11)
        assert metrics["r2"] == pytest.approx(1 - sum(residual ** 2) / sum((truth-truth.mean()) ** 2))
        assert session.partition(partition)["row_id"].tolist() == expected.iloc[indices]["row_id"].tolist()
    assert plans == hashlib.sha256(pickle.dumps((session.impute_plan, session.scale_plan))).hexdigest()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("layout", ["single", "partitioned"])
@pytest.mark.parametrize("transition", ["clear", "replace", "subset", "new"])
@pytest.mark.parametrize("restore", ["load", "reattach"])
def test_replacement_is_one_snapshot(backing, layout, transition, restore, tmp_path):
    old, _ = _session(tmp_path, backing, generation=1,
                      plans="none" if transition == "new" else "standard")
    new_plan = {"clear": "none", "replace": "minmax", "subset": "impute", "new": "standard"}[transition]
    latest, expected = _session(tmp_path, backing, plans=new_plan)
    path = tmp_path / "checkpoint"
    try:
        old.checkpoint_save(path, sidecar_layout="partitioned", sidecar_partition_rows=5)
        latest.checkpoint_save(path, sidecar_layout=layout, sidecar_partition_rows=11)
        assert (path / "plans.joblib").exists() == (new_plan != "none")
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert meta["has_plans"] == (new_plan != "none")
        assert ("plans.joblib" in manifest["hashes"]) == (new_plan != "none")
        for name, digest in manifest["hashes"].items():
            assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
        restored = Session.checkpoint_load(path, trusted=new_plan != "none") if restore == "load" else old.reattach(path, trusted=new_plan != "none")
        try:
            assert (restored.impute_plan is not None) == (latest.impute_plan is not None)
            assert (restored.scale_plan is not None) == (latest.scale_plan is not None)
            if latest.impute_plan is not None:
                assert restored.impute_plan.statistics_ == latest.impute_plan.statistics_
            if latest.scale_plan is not None:
                assert restored.scale_plan.to_dict() == latest.scale_plan.to_dict()
            _consumer(restored, expected, latest.split_plan.to_dict())
        finally:
            restored.close_native()
    finally:
        old.close_native()
        latest.close_native()


@pytest.mark.parametrize("backing", NATIVE)
@pytest.mark.parametrize("rows_per_part", [1, 7, 48])
@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_partitioned_native_rows_drive_same_holdouts(backing, rows_per_part, codec, tmp_path):
    original, expected = _session(tmp_path, backing)
    path = tmp_path / "parts"
    try:
        original.checkpoint_save(path, sidecar_layout="partitioned",
                                 sidecar_partition_rows=rows_per_part, sidecar_compression=codec)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        info = meta["native_sidecar"]
        assert info["n_partitions"] == (48 + rows_per_part - 1) // rows_per_part
        assert info["compression"] == codec
        restored = Session.checkpoint_load(path)
        try:
            assert restored.dataset.has_native
            assert restored.reattach_result.details["native_sidecar"]["parts"] == info["parts"]
            _consumer(restored, expected, original.split_plan.to_dict())
        finally:
            restored.close_native()
    finally:
        original.close_native()


@pytest.mark.parametrize("backing", NATIVE)
@pytest.mark.parametrize("change", ["duplicate_part", "extra_rows", "missing_part"])
def test_partition_membership_change_refuses_resume_but_data_only_is_usable(backing, change, tmp_path):
    original, expected = _session(tmp_path, backing)
    path = tmp_path / "parts"
    receiver, _ = _session(tmp_path, "pandas", generation=3, plans="standard")
    receiver_frame = receiver.to_pandas()
    state = (receiver.split_plan.to_dict(), len(receiver.history), receiver.impute_plan, receiver.scale_plan)
    try:
        original.checkpoint_save(path, sidecar_layout="partitioned", sidecar_partition_rows=7)
        parts = sorted((path / "data/native_sidecar").glob("*.parquet"))
        if change == "duplicate_part":
            shutil.copyfile(parts[0], parts[0].parent / "unrecorded.parquet")
        elif change == "extra_rows":
            expected.iloc[:3].to_parquet(parts[0].parent / "extra.parquet", index=False)
        else:
            parts[0].unlink()
        with pytest.raises(ValidationError, match="partitions"):
            receiver.reattach(path, trusted=True)
        assert receiver.split_plan.to_dict() == state[0] and len(receiver.history) == state[1]
        assert receiver.impute_plan is state[2] and receiver.scale_plan is state[3]
        pd.testing.assert_frame_equal(receiver.to_pandas(), receiver_frame)
        safe = Session.checkpoint_load(path, data_only=True)
        try:
            pd.testing.assert_frame_equal(safe.to_pandas(), expected, check_dtype=False)
            assert safe.split_plan is None and safe.impute_plan is None
            assert safe.reattach_result.status == "fresh_ingest"
        finally:
            safe.close_native()
    finally:
        original.close_native()
        receiver.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("restore", ["load", "reattach"])
def test_old_replacement_with_inactive_plan_file_does_not_load_it(backing, restore, tmp_path):
    """Recreate the old writer's leftover payload through two public saves."""
    old, _ = _session(tmp_path, backing, generation=1, plans="standard")
    latest, expected = _session(tmp_path, backing)
    path = tmp_path / "old-replacement"
    try:
        old.checkpoint_save(path)
        saved_plan = (path / "plans.joblib").read_bytes()
        latest.checkpoint_save(path)
        # Retain the old writer's stale file; its metadata/manifest mark it inactive.
        (path / "plans.joblib").write_bytes(saved_plan)
        restored = Session.checkpoint_load(path) if restore == "load" else old.reattach(path)
        try:
            assert restored.impute_plan is None and restored.scale_plan is None
            _consumer(restored, expected, latest.split_plan.to_dict())
        finally:
            restored.close_native()
    finally:
        old.close_native()
        latest.close_native()
