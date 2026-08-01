"""Leakage, correctness, and edge-case tests for deeper preprocess APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import LeakageError


def _base_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num": np.concatenate([rng.normal(0, 1, n - 2), [50.0, -50.0]]),
            "cat": (["a", "b", "c", "rare"] * ((n // 4) + 1))[:n],
            "y": ([0, 1] * ((n // 2) + 1))[:n],
        }
    )


def test_handle_outliers_requires_split_and_caps_with_train_fences() -> None:
    frame = _base_frame()
    session = Session.ingest(frame).set_roles({"num": "feature", "cat": "feature", "y": "target"})
    with pytest.raises(LeakageError):
        session.handle_outliers(columns=["num"], action="cap")

    session.split(test_size=0.25, stratify=True, random_state=0)
    train = session.partition("train")
    q1 = float(train["num"].quantile(0.25))
    q3 = float(train["num"].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    session.handle_outliers(columns=["num"], method="iqr", action="cap")
    assert session.outlier_plan is not None
    assert session.outlier_plan.lower_["num"] == pytest.approx(lower)
    assert session.outlier_plan.upper_["num"] == pytest.approx(upper)
    values = session.dataset.frame["num"]
    assert float(values.min()) >= lower - 1e-9
    assert float(values.max()) <= upper + 1e-9
    assert session.last_preprocess is not None
    assert session.last_preprocess.operation == "handle_outliers"
    outlier_records = [
        record
        for record in session.history
        if record.get("operation_id") == "handle_outliers"
    ]
    assert outlier_records
    assert outlier_records[-1]["schema_version"] == 2
    assert outlier_records[-1]["result_summary"]["findings"]


def test_handle_outliers_detect_does_not_mutate_and_drop_rebuilds_split() -> None:
    frame = _base_frame()
    session = (
        Session.ingest(frame)
        .set_roles({"num": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
    )
    before = session.to_pandas().copy()
    session.handle_outliers(columns=["num"], action="detect")
    assert session.to_pandas().equals(before)

    n_before = len(session.to_pandas())
    session.handle_outliers(columns=["num"], action="drop")
    assert session.outlier_plan is not None
    assert session.outlier_plan.n_dropped >= 1
    assert len(session.to_pandas()) == n_before - session.outlier_plan.n_dropped
    assert session.split_plan is not None
    session.split_plan.assert_disjoint()
    assert set(session.split_plan.train_indices).isdisjoint(session.split_plan.test_indices)


def test_bin_uses_train_edges_only() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 6.0],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    with pytest.raises(LeakageError):
        Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).bin(columns=["x"])

    session.bin(columns=["x"], strategy="quantile", n_bins=4, encode_as="ordinal")
    assert "x" not in session.dataset.columns
    assert "x_bin" in session.dataset.columns
    assert session.binning_plan is not None
    assert session.binning_plan.edges_["x"][0] == float("-inf")
    assert session.binning_plan.edges_["x"][-1] == float("inf")
    assert session.last_preprocess is not None
    assert "binning.applied" in {f.key for f in session.last_preprocess.findings}


def test_encode_infrequent_pools_rare_train_levels() -> None:
    frame = pd.DataFrame(
        {
            "city": ["a"] * 20 + ["b"] * 20 + ["c"] * 2 + ["d"] * 2,
            "x": list(range(44)),
            "y": [0, 1] * 22,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"city": "feature", "x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    session.encode(columns=["city"], method="infrequent", min_frequency=3)
    assert session.encode_plan is not None
    assert session.encode_plan.method == "infrequent"
    rare = set(session.encode_plan.infrequent_maps_["city"])
    assert "c" in rare and "d" in rare
    cols = session.dataset.columns
    assert any("__infrequent__" in name or "infrequent" in name for name in cols)


def test_encode_target_is_oof_on_train_and_requires_split() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c", "a", "b", "c", "a", "b", "c"] * 3,
            "x": list(range(36)),
            "y": [0, 1, 0, 1, 1, 0] * 6,
        }
    )
    bare = Session.ingest(frame).set_roles({"group": "feature", "x": "feature", "y": "target"})
    with pytest.raises(LeakageError):
        bare.encode(columns=["group"], method="target")

    session = bare.split(test_size=0.25, stratify=True, random_state=0)
    session.encode(columns=["group"], method="target", n_folds=3, random_state=0)
    assert "group_target" in session.dataset.columns
    assert "group" not in session.dataset.columns
    assert session.encode_plan is not None
    assert session.encode_plan.method == "target"
    assert session.last_preprocess is not None
    assert any(
        "out-of-fold" in warning.lower() or "OOF" in warning
        for warning in session.last_preprocess.warnings
    ) or any("out-of-fold" in lim.lower() for lim in session.last_preprocess.limitations)

    # Holdout encodings must use the stored global map (finite values).
    holdout_idx = list(session.split_plan.test_indices)  # type: ignore[union-attr]
    values = session.to_pandas().iloc[holdout_idx]["group_target"]
    assert values.notna().all()


def test_select_features_variance_and_univariate_train_only() -> None:
    frame = pd.DataFrame(
        {
            "keep": np.linspace(0, 1, 40),
            "const": [1.0] * 40,
            "noise": np.random.default_rng(1).normal(size=40),
            "y": [0] * 20 + [1] * 20,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "keep": "feature",
                "const": "feature",
                "noise": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(LeakageError):
        Session.ingest(frame).set_roles(
            {"keep": "feature", "const": "feature", "noise": "feature", "y": "target"}
        ).select_features(strategy="variance")

    session.select_features(strategy="variance", threshold=0.0)
    assert "const" not in session.dataset.columns
    assert "y" in session.dataset.columns
    assert session.feature_select_plan is not None
    assert "const" in session.feature_select_plan.dropped_features_

    session.select_features(strategy="univariate", k=1, score_func="f_classif")
    assert len(session.feature_select_plan.selected_features_) == 1
    assert session.last_preprocess is not None
    assert session.last_preprocess.recommendations


def test_select_features_model_based_keeps_at_least_one_feature() -> None:
    frame = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 30),
            "b": np.linspace(1, 2, 30),
            "y": [0, 1] * 15,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.3, stratify=True, random_state=0)
        .select_features(
            strategy="model",
            estimator=LogisticRegression(max_iter=400),
        )
    )
    assert session.feature_select_plan is not None
    assert len(session.feature_select_plan.selected_features_) >= 1
    assert "y" in session.dataset.columns
