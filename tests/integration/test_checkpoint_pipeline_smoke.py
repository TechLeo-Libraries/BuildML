"""CI smoke: checkpoint and pipeline bundle side-by-side round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.pipeline.bundle import BUNDLE_FORMAT, PLANS_FORMAT, unpack_plans_payload
from buildml.preprocess import apply_preprocess_plans


def test_checkpoint_and_pipeline_side_by_side_smoke(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "age": [21, None, 35, 40, 29, 33, 52, 47, 38, 44, 31, 27],
            "income": [40, 55, 60, 80, 50, 70, 90, 65, 72, 58, 61, 49],
            "approved": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "approved": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .impute(strategy="median")
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=400), task="classification")
    )
    before = session.evaluate(partition="test").metrics

    ckpt = tmp_path / "checkpoint"
    pipe = tmp_path / "pipeline"
    session.checkpoint_save(ckpt)
    session.save_pipeline(pipe, evaluate_partition="test", title="Smoke model")

    assert (ckpt / "data" / "frame.parquet").exists()
    assert (ckpt / "plans.joblib").exists()
    assert not (ckpt / "model.joblib").exists()
    assert (pipe / "model.joblib").exists()
    assert (pipe / "plans.joblib").exists()
    assert not (pipe / "data" / "frame.parquet").exists()

    meta = json.loads((pipe / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == BUNDLE_FORMAT
    assert meta["plans_format"] == PLANS_FORMAT

    import joblib

    plans_payload = joblib.load(pipe / "plans.joblib")
    assert plans_payload["format"] == PLANS_FORMAT
    plans, fmt = unpack_plans_payload(plans_payload)
    assert fmt == PLANS_FORMAT
    assert plans["impute_plan"] is not None

    # Legacy flat plans.joblib still migrates.
    legacy_plans, legacy_fmt = unpack_plans_payload(
        {
            "impute_plan": plans["impute_plan"],
            "scale_plan": plans["scale_plan"],
        }
    )
    assert legacy_fmt.startswith("buildml.plans.")
    assert legacy_plans["impute_plan"] is not None

    restored_ckpt = Session.checkpoint_load(ckpt, trusted=True)
    assert restored_ckpt.fit_result is None
    assert restored_ckpt.impute_plan is not None
    assert restored_ckpt.scale_plan is not None

    restored_pipe = Session.ingest(frame).load_pipeline(pipe, trusted=True)
    assert restored_pipe.fit_result is not None
    assert restored_pipe.model_card is not None
    assert restored_pipe.impute_plan is not None

    holdout = frame.iloc[list(session.split_plan.test_indices)].reset_index(drop=True)  # type: ignore[union-attr]
    applied = apply_preprocess_plans(
        holdout,
        {
            "impute_plan": restored_pipe.impute_plan,
            "scale_plan": restored_pipe.scale_plan,
        },
        roles={"age": "feature", "income": "feature", "approved": "target"},
    )
    assert "impute" in applied.applied
    assert "scale" in applied.applied

    scored = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "approved": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .load_pipeline(pipe, trusted=True)
    )
    scored.apply_preprocess_plans()
    after = scored.evaluate(partition="test").metrics
    assert after["f1_weighted"] == pytest.approx(before["f1_weighted"])
