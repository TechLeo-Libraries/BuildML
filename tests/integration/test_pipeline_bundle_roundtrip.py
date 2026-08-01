"""Pipeline bundle + model card persistence roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.preprocess.binning import transform_binning
from buildml.preprocess.impute import transform_simple_imputer
from buildml.preprocess.outliers import apply_outlier_plan
from buildml.preprocess.scale import transform_scaler
from buildml.preprocess.select import transform_feature_selector


def test_save_load_pipeline_roundtrip(tmp_path: Path) -> None:
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
        .fit(LogisticRegression(max_iter=500), task="classification")
    )
    before = session.evaluate(partition="test").metrics

    bundle_dir = tmp_path / "pipeline"
    session.save_pipeline(bundle_dir, evaluate_partition="test", title="Approval model")
    assert (bundle_dir / "model.joblib").exists()
    assert (bundle_dir / "plans.joblib").exists()
    assert (bundle_dir / "meta.json").exists()
    assert (bundle_dir / "model_card.json").exists()
    assert (bundle_dir / "model_card.md").exists()
    assert session.model_card is not None
    assert session.model_card.title == "Approval model"
    assert "test" in session.model_card.metrics
    assert session.model_card.preprocess_summary["impute"] is not None
    assert "impute" in session.model_card.lineage["plans_present"]
    assert "scale" in session.model_card.lineage["plans_present"]
    md = (bundle_dir / "model_card.md").read_text(encoding="utf-8")
    assert "Plans present:" in md

    restored = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "approved": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .impute(strategy="median")
        .scale(method="standard")
        .load_pipeline(bundle_dir)
    )
    assert restored.fit_result is not None
    assert restored.impute_plan is not None
    assert restored.scale_plan is not None
    assert restored.model_card is not None
    after = restored.evaluate(partition="test").metrics
    assert after["f1_weighted"] == pytest.approx(before["f1_weighted"])


def test_pipeline_persists_outlier_binning_select_date_plans(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 48
    frame = pd.DataFrame(
        {
            "num": np.concatenate([rng.normal(0, 1, n - 2), [40.0, -40.0]]),
            "noise": rng.normal(size=n),
            "const": np.zeros(n),
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": ([0, 1] * (n // 2)),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "num": "feature",
                "noise": "feature",
                "const": "feature",
                "when": "time",
                "y": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
        .extract_dates(columns=["when"], drop_original=True)
        .handle_outliers(columns=["num"], method="iqr", action="cap")
        .impute(strategy="median")
        .bin(columns=["num"], strategy="quantile", n_bins=4)
        .select_features(strategy="variance", threshold=0.0)
        .fit(LogisticRegression(max_iter=400), task="classification")
    )
    before = session.evaluate(partition="test").metrics

    bundle = tmp_path / "full_pipe"
    session.save_pipeline(bundle, evaluate_partition=None, title="Depth bundle")
    card_md = (bundle / "model_card.md").read_text(encoding="utf-8")
    for label in ("outliers", "binning", "feature_select", "dates"):
        assert label in card_md
        assert label in session.model_card.lineage["plans_present"]  # type: ignore[index]

    restored = Session.ingest(frame).load_pipeline(bundle)
    assert restored.outlier_plan is not None
    assert restored.binning_plan is not None
    assert restored.feature_select_plan is not None
    assert restored.date_plan is not None
    assert restored.impute_plan is not None
    assert restored.fit_result is not None

    # Rebuild the same feature contract from raw rows using restored plans.
    rebuilt = (
        Session.ingest(frame)
        .set_roles(
            {
                "num": "feature",
                "noise": "feature",
                "const": "feature",
                "when": "time",
                "y": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
        .extract_dates(columns=["when"], drop_original=True)
    )
    dataset, split_plan, _, _ = apply_outlier_plan(
        rebuilt.dataset,
        rebuilt.split_plan,  # type: ignore[arg-type]
        restored.outlier_plan,
    )
    rebuilt._dataset = dataset
    rebuilt._split_plan = split_plan
    rebuilt._dataset = transform_simple_imputer(rebuilt.dataset, restored.impute_plan)
    rebuilt._dataset, _ = transform_binning(rebuilt.dataset, restored.binning_plan)
    rebuilt._dataset, _ = transform_feature_selector(
        rebuilt.dataset, restored.feature_select_plan
    )
    rebuilt._fit_result = restored.fit_result
    after = rebuilt.evaluate(partition="test").metrics
    assert after["f1_weighted"] == pytest.approx(before["f1_weighted"])


def test_pipeline_bundle_distinct_from_checkpoint(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": list(range(20)), "y": [i % 2 for i in range(20)]})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .impute(strategy="median")
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    ckpt = tmp_path / "ckpt"
    pipe = tmp_path / "pipe"
    session.checkpoint_save(ckpt)
    session.save_pipeline(pipe, evaluate_partition=None)
    assert (ckpt / "data" / "frame.parquet").exists()
    assert (ckpt / "plans.joblib").exists()
    assert not (pipe / "data" / "frame.parquet").exists()
    assert (pipe / "model_card.json").exists()
    assert not (ckpt / "model_card.json").exists()
    assert session.model_card is not None
    assert session.model_card.lineage.get("contains_checkpoint") is False
    assert "checkpoint_compatibility" in session.model_card.lineage
    assert "complementary" in session.model_card.lineage["checkpoint_compatibility"].lower()

    restored_ckpt = Session.checkpoint_load(ckpt)
    assert restored_ckpt.impute_plan is not None
    assert restored_ckpt.fit_result is None


def test_checkpoint_plan_roundtrip_applies_transforms(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .impute(strategy="median")
        .scale(method="standard")
    )
    expected = session.to_pandas().copy()
    path = tmp_path / "ckpt_plans"
    session.checkpoint_save(path)

    restored = Session.checkpoint_load(path)
    assert restored.impute_plan is not None
    assert restored.scale_plan is not None
    # Data already transformed in the checkpoint frame.
    pd.testing.assert_frame_equal(restored.to_pandas(), expected)

    # Plans can re-apply to a fresh ingest of the original rows.
    fresh = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    reapplied = transform_simple_imputer(fresh.dataset, restored.impute_plan)
    reapplied = transform_scaler(reapplied, restored.scale_plan)
    pd.testing.assert_frame_equal(
        reapplied.frame.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
