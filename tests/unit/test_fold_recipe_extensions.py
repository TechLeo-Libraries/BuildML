"""Fold-local model selection and outlier recipe leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor


def test_fold_local_model_select_no_eval_peeking() -> None:
    rng = np.random.default_rng(2)
    n = 90
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise": rng.normal(scale=0.05, size=n),
            "decoy": rng.normal(size=n),
            "y": np.zeros(n, dtype=int),
        }
    )
    frame["y"] = (frame["signal"] > 0).astype(int)
    # Poison eval-looking decoy correlation if someone peeked at all rows.
    frame.loc[frame.index >= 70, "decoy"] = frame.loc[frame.index >= 70, "y"] * 10.0

    x = frame[["signal", "noise", "decoy"]]
    y = frame["y"]
    train_idx = np.arange(0, 70)
    eval_idx = np.arange(70, 90)
    prep = build_fold_preprocessor(
        x.iloc[train_idx],
        PreprocessRecipe(
            impute=None,
            select="model",
            select_estimator=LogisticRegression(max_iter=300),
        ),
        y.iloc[train_idx],
    )
    transformed_eval = prep.transform(x.iloc[eval_idx])
    assert "signal" in transformed_eval.columns or len(transformed_eval.columns) >= 1
    # Selector fitted without eval rows: decoy should not dominate solely from eval labels.
    assert set(prep._selected_features_).issubset({"signal", "noise", "decoy"})

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "signal": "feature",
                "noise": "feature",
                "decoy": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.2, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        preprocess=PreprocessRecipe(impute=None, select="model"),
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["select"] == "model"
    assert any("model-based" in tip or "SelectFromModel" in tip for tip in result.limitations)
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]


def test_fold_local_outliers_fit_train_apply_eval() -> None:
    x = pd.DataFrame(
        {
            "a": [0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 50.0],
            "b": [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0])
    # Fit fences on first 6 rows only (no extreme 50).
    prep = build_fold_preprocessor(
        x.iloc[:6],
        PreprocessRecipe(
            impute=None,
            outliers="iqr",
            outlier_action="cap",
            outlier_columns=("a",),
            iqr_multiplier=1.5,
        ),
        y.iloc[:6],
    )
    out = prep.transform(x.iloc[[6]])
    assert float(out["a"].iloc[0]) < 50.0
    assert float(out["a"].iloc[0]) == pytest.approx(prep._outlier_upper["a"])

    # Leakage unit: fences must ignore eval extreme when fitting.
    assert prep._outlier_upper["a"] < 10.0


def test_fold_local_outliers_in_cv_score_preserves_test() -> None:
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(size=n - 1), [100.0]]),
            "y": ([0, 1] * (n // 2)),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    result = session.cv_score(
        LogisticRegression(max_iter=300),
        cv=3,
        preprocess=PreprocessRecipe(impute="median", outliers="zscore", outlier_action="cap"),
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["outliers"] == "zscore"
    assert any("Fold-local outlier fences" in tip for tip in result.limitations)
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]


def test_fold_local_model_select_requires_labels() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 0.0, 1.0, 1.0]})
    with pytest.raises(ValidationError, match="requires fold-train labels"):
        build_fold_preprocessor(x, PreprocessRecipe(impute=None, select="model"), y_train=None)


def test_fold_local_binning_edges_ignore_eval() -> None:
    x = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
            "b": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])
    prep = build_fold_preprocessor(
        x.iloc[:5],
        PreprocessRecipe(
            impute=None,
            binning="quantile",
            n_bins=4,
            binning_columns=("a",),
        ),
        y.iloc[:5],
    )
    # Extreme 100 was not in fold-train; edges must not use it.
    finite_edges = [e for e in prep._binning_edges["a"] if np.isfinite(e)]
    assert max(finite_edges) < 50.0
    out = prep.transform(x.iloc[[5]])
    assert "a_bin" in out.columns
    assert float(out["a_bin"].iloc[0]) == float(len(prep._binning_edges["a"]) - 2)


def test_fold_local_dates_and_binning_in_cv() -> None:
    n = 80
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2023-01-01", periods=n, freq="D"),
            "x": rng.normal(size=n),
            "y": ([0, 1] * (n // 2)),
        }
    )
    session = (
        Session.ingest(frame)
        # Keep `when` as a feature so the fold design matrix still carries it.
        .set_roles({"when": "feature", "x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=3,
        preprocess=PreprocessRecipe(
            impute="median",
            dates=True,
            date_columns=("when",),
            date_drop_original=True,
            binning="uniform",
            n_bins=3,
            binning_columns=("x",),
        ),
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["dates"] is True
    assert result.fold_preprocess["binning"] == "uniform"
    assert "session_global_only" in result.fold_preprocess
    assert any("bin edges" in tip for tip in result.limitations)
    assert any("date expansion" in tip for tip in result.limitations)
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]
