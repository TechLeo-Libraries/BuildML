"""Score-time preprocess plan reapplication round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import apply_preprocess_plans


def test_apply_preprocess_plans_roundtrip_held_out(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "income": rng.normal(60, 15, n),
            "city": rng.choice(["a", "b", "c"], size=n),
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": ([0, 1] * (n // 2)),
        }
    )
    frame.loc[0, "age"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "city": "feature",
                "when": "time",
                "y": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
        .extract_dates(columns=["when"], drop_original=True)
        .handle_outliers(columns=["income"], method="iqr", action="cap")
        .impute(strategy="median")
        .encode(method="onehot", columns=["city"])
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=400), task="classification")
    )
    expected_test = session.partition("test").copy()
    before = session.evaluate(partition="test").metrics

    pipe = tmp_path / "pipe"
    session.save_pipeline(pipe, evaluate_partition=None)
    restored = Session.ingest(frame).load_pipeline(pipe)
    assert restored.impute_plan is not None
    assert restored.scale_plan is not None
    assert restored.encode_plan is not None
    assert restored.date_plan is not None
    assert restored.outlier_plan is not None

    holdout = frame.iloc[list(session.split_plan.test_indices)].reset_index(drop=True)  # type: ignore[union-attr]
    result = apply_preprocess_plans(
        holdout,
        {
            "date_plan": restored.date_plan,
            "outlier_plan": restored.outlier_plan,
            "impute_plan": restored.impute_plan,
            "encode_plan": restored.encode_plan,
            "scale_plan": restored.scale_plan,
        },
        roles={
            "age": "feature",
            "income": "feature",
            "city": "feature",
            "when": "time",
            "y": "target",
        },
    )
    assert "resample" not in result.applied
    assert "dates" in result.applied
    assert "impute" in result.applied
    assert "scale" in result.applied
    # Compare feature columns used by the estimator.
    feature_cols = list(restored.fit_result.feature_columns)  # type: ignore[union-attr]
    got = result.dataset.frame[feature_cols].reset_index(drop=True)
    want = expected_test[feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, atol=1e-8)

    # Session helper path using restored plans on a fresh ingest of all rows.
    fresh = (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "city": "feature",
                "when": "time",
                "y": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
        .load_pipeline(pipe)
    )
    applied = fresh.apply_preprocess_plans()
    assert "impute" in applied.applied
    after = fresh.evaluate(partition="test").metrics
    assert after["f1_weighted"] == pytest.approx(before["f1_weighted"])


def test_apply_preprocess_plans_missing_columns_and_resample_lineage() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .impute(strategy="median")
        .scale(method="standard")
    )
    bad = pd.DataFrame({"z": [1.0, 2.0]})
    with pytest.raises(ValidationError, match="missing from score frame"):
        apply_preprocess_plans(
            bad,
            {"impute_plan": session.impute_plan, "scale_plan": session.scale_plan},
        )

    # Resample lineage is skipped, not applied.
    from buildml.preprocess.imbalance import ResamplePlan

    lineage = ResamplePlan(
        sampler="random_oversample",
        n_train_before=3,
        n_train_after=4,
        class_counts_before={"0": 2, "1": 1},
        class_counts_after={"0": 2, "1": 2},
    )
    result = apply_preprocess_plans(
        frame,
        {
            "impute_plan": session.impute_plan,
            "scale_plan": session.scale_plan,
            "resample_plan": lineage,
        },
        roles={"a": "feature", "y": "target"},
    )
    assert "resample" in result.skipped
    assert any("lineage-only" in note for note in result.warnings)
