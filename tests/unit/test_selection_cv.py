"""Leakage-safe CV and search behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.preprocess import PreprocessRecipe
from buildml.preprocess.fold import FoldLocalPreprocessor, build_fold_preprocessor


def _cls_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.3 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_cv_score_stays_in_train_and_reports_mean_std() -> None:
    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        task="classification",
        cv=4,
        cv_strategy="stratified",
    )
    assert result.population == "train"
    assert "test" in result.held_out_partitions
    assert result.n_splits == 4
    assert "f1_weighted" in result.mean_metrics
    assert "f1_weighted" in result.std_metrics
    assert result.interpretation
    assert result.recommendations
    # Fold sizes must come from train only.
    train_n = len(session.split_plan.train_indices)  # type: ignore[union-attr]
    for fold in result.folds:
        assert fold.n_train + fold.n_eval == train_n
    assert session.last_cv is result
    # Test membership never changes during CV.
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]


def test_cv_score_requires_split() -> None:
    session = Session.ingest(_cls_frame(40)).set_roles(
        {"x1": "feature", "x2": "feature", "y": "target"}
    )
    with pytest.raises(LeakageError, match="full data"):
        session.cv_score(LogisticRegression(max_iter=200), cv=3)


def test_fold_local_preprocess_records_recipe() -> None:
    frame = _cls_frame(70)
    frame.loc[::7, "x1"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, stratify=True, random_state=1)
    )
    recipe = PreprocessRecipe(impute="median", scale="standard")
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=3,
        preprocess=recipe,
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["impute"] == "median"
    assert any("Fold-local PreprocessRecipe" in tip for tip in result.limitations)


def test_session_preprocess_limitation_flagged() -> None:
    session = (
        Session.ingest(_cls_frame(60))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
        .impute(strategy="median")
        .scale(method="standard")
    )
    result = session.cv_score(LogisticRegression(max_iter=300), cv=3)
    assert any(
        "Session-global" in tip or "full train partition" in tip for tip in result.limitations
    )


def test_grid_search_ranks_and_refits() -> None:
    session = (
        Session.ingest(_cls_frame(90))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=3)
    )
    search = session.grid_search(
        DecisionTreeClassifier(random_state=0),
        param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
        cv=3,
        ranking_metric="f1_weighted",
        refit=True,
    )
    assert len(search.trials) == 4
    assert search.best_params
    assert search.best_score is not None
    assert session.fit_result is not None
    assert session.last_search is search
    assert search.interpretation
    # Winner should be usable on held-out validation.
    metrics = session.evaluate(partition="validation").metrics
    assert "f1_weighted" in metrics


def test_randomized_search_budget() -> None:
    session = (
        Session.ingest(_cls_frame(70))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=4)
    )
    search = session.randomized_search(
        DecisionTreeClassifier(random_state=0),
        param_distributions={"max_depth": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 4]},
        n_iter=3,
        cv=3,
        refit=False,
    )
    assert len(search.trials) == 3
    assert session.fit_result is None


def test_group_cv_rejects_too_few_groups() -> None:
    frame = _cls_frame(40)
    frame["g"] = [0, 1] * 20
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target", "g": "group"})
        .split(test_size=0.25, random_state=0)
    )
    with pytest.raises(ValidationError, match="distinct groups"):
        session.cv_score(LogisticRegression(max_iter=200), cv=5, cv_strategy="group")


def test_group_and_time_cv_strategies() -> None:
    frame = _cls_frame(80)
    frame["g"] = [i % 10 for i in range(80)]
    frame["t"] = pd.date_range("2024-01-01", periods=80, freq="D")
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "y": "target",
                "g": "group",
                "t": "time",
            }
        )
        .group_split(test_size=0.2, random_state=0)
    )
    group_cv = session.cv_score(
        LogisticRegression(max_iter=300),
        cv=4,
        cv_strategy="group",
    )
    assert group_cv.cv_strategy == "group"
    assert group_cv.n_splits == 4

    time_session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "y": "target",
                "g": "group",
                "t": "time",
            }
        )
        .time_split(test_size=0.2)
    )
    time_cv = time_session.cv_score(
        LogisticRegression(max_iter=300),
        cv=3,
        cv_strategy="time",
    )
    assert time_cv.cv_strategy == "time"
    assert time_cv.n_splits == 3
    assert any("time-series" in tip.lower() for tip in time_cv.limitations)


def test_fold_local_infrequent_encoding_uses_fold_train_only() -> None:
    """Rare maps must come from fold-train counts; eval-only levels are unknown."""
    n = 40
    frame = pd.DataFrame(
        {
            "city": (["a"] * 16 + ["b"] * 16 + ["rare_eval"] * 8),
            "x": np.linspace(0, 1, n),
            "y": [0, 1] * (n // 2),
        }
    )
    x = frame[["city", "x"]]
    y = frame["y"]
    # Force a fold where rare_eval appears only in eval positions.
    train_pos = list(range(32))
    eval_pos = list(range(32, 40))
    recipe = PreprocessRecipe(
        impute=None,
        encode="infrequent",
        encode_columns=("city",),
        min_frequency=3,
    )
    prep = build_fold_preprocessor(x.iloc[train_pos], recipe, y.iloc[train_pos])
    assert isinstance(prep, FoldLocalPreprocessor)
    assert "rare_eval" not in set(prep._infrequent_maps.get("city", []))
    # Eval-only level collapses via handle_unknown after infrequent maps from train.
    encoded_eval = prep.transform(x.iloc[eval_pos])
    assert encoded_eval.shape[0] == 8
    # Train rare maps should list levels below threshold among train rows only.
    train_counts = x.iloc[train_pos]["city"].astype(str).value_counts()
    for level, count in train_counts.items():
        if count < 3:
            assert str(level) in prep._infrequent_maps["city"]


def test_fold_local_target_encoding_no_eval_label_leakage() -> None:
    """Fold-eval labels must not influence category means."""
    n = 36
    # Category "leak" appears in both partitions; eval rows all have y=1,
    # train rows for "leak" all have y=0. Leakage would pull the mean toward 1.
    cities = ["a"] * 12 + ["b"] * 12 + ["leak"] * 12
    y = [0] * 12 + [1] * 12 + [0] * 6 + [1] * 6
    frame = pd.DataFrame({"city": cities, "x": np.arange(n, dtype=float), "y": y})
    x = frame[["city", "x"]]
    y_s = frame["y"]
    train_pos = list(range(0, 12)) + list(range(12, 24)) + list(range(24, 30))
    eval_pos = list(range(30, 36))
    recipe = PreprocessRecipe(
        impute=None,
        encode="target",
        encode_columns=("city",),
        target_smoothing=1.0,
    )
    prep = build_fold_preprocessor(x.iloc[train_pos], recipe, y_s.iloc[train_pos])
    train_mean_leak = float(np.mean(y_s.iloc[train_pos][x.iloc[train_pos]["city"] == "leak"]))
    assert train_mean_leak == pytest.approx(0.0)
    encoded = prep.transform(x.iloc[eval_pos])
    # Encoded leak values must stay near the fold-train mean (~0), not eval mean (1).
    leak_values = encoded.loc[x.iloc[eval_pos]["city"].to_numpy() == "leak", "city_target"]
    assert float(leak_values.mean()) < 0.35

    session = (
        Session.ingest(frame)
        .set_roles({"city": "feature", "x": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=3,
        preprocess=recipe,
    )
    assert any("fold-train labels only" in tip for tip in result.limitations)
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]


def test_fold_local_feature_selection_no_test_peeking() -> None:
    rng = np.random.default_rng(1)
    n = 80
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise1": rng.normal(scale=0.01, size=n),
            "noise2": rng.normal(scale=0.01, size=n),
            "const": np.zeros(n),
            "y": (rng.normal(size=n) > 0).astype(int),
        }
    )
    # Make signal predict y on train-ish rows.
    frame["y"] = (frame["signal"] > 0).astype(int)
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "signal": "feature",
                "noise1": "feature",
                "noise2": "feature",
                "const": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    recipe = PreprocessRecipe(
        impute=None,
        select="variance",
        select_threshold=0.0,
    )
    result = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        preprocess=recipe,
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["select"] == "variance"
    assert any("feature selection fits on fold-train" in tip for tip in result.limitations)
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]

    # Direct unit check: selector fitted without eval rows.
    train = session.partition("train")
    x_train = train[["signal", "noise1", "noise2", "const"]]
    y_train = train["y"]
    # Simulate one fold by holding out the last 10 train rows.
    prep = build_fold_preprocessor(
        x_train.iloc[:-10],
        PreprocessRecipe(impute=None, select="variance", select_threshold=1e-8),
        y_train.iloc[:-10],
    )
    assert "const" not in prep._selected_features_


def test_fold_local_univariate_select_requires_labels() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 0.0, 1.0, 1.0]})
    recipe = PreprocessRecipe(impute=None, select="univariate", select_k=1)
    with pytest.raises(ValidationError, match="requires fold-train labels"):
        build_fold_preprocessor(x, recipe, y_train=None)
