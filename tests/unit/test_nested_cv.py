"""Outer/inner nested CV leakage and structure tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe


def _cls_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.4 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_nested_cv_outer_inner_separation_and_no_test_peek() -> None:
    session = (
        Session.ingest(_cls_frame(140))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    valid_idx = set(session.split_plan.validation_indices)  # type: ignore[union-attr]
    train_n = len(session.split_plan.train_indices)  # type: ignore[union-attr]

    result = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        param_grid={"max_depth": [2, 3], "min_samples_leaf": [1, 4]},
        outer_cv=3,
        inner_cv=2,
        cv_strategy="stratified",
        scoring_metric="f1_weighted",
    )
    assert result.population == "train"
    assert "test" in result.held_out_partitions
    assert "validation" in result.held_out_partitions
    assert result.n_outer_splits == 3
    assert result.search_method == "grid"
    assert result.scoring_metric in result.mean_metrics
    assert result.scoring_metric in result.std_metrics
    assert result.inner_selection_summary["n_outer_folds"] == 3
    assert result.inner_selection_summary["param_stability"] in {"high", "medium", "low"}
    assert result.limitations
    assert result.recommendations
    assert any("Outer-eval rows never enter inner" in tip for tip in result.limitations)
    assert any("inner cv means" in tip.lower() for tip in result.limitations)

    for fold in result.outer_folds:
        assert fold.n_train + fold.n_eval == train_n
        assert fold.best_params
        assert fold.inner_n_trials == 4
        assert "f1_weighted" in fold.metrics

    # Session holdouts untouched.
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]
    assert set(session.split_plan.validation_indices) == valid_idx  # type: ignore[union-attr]
    assert session.last_nested_cv is result
    assert session.fit_result is None


def test_nested_cv_requires_a_search_space() -> None:
    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
    )
    with pytest.raises(ValidationError, match="requires an estimator and/or recipe search space"):
        session.nested_cv_score(LogisticRegression(max_iter=200), outer_cv=2, inner_cv=2)
    with pytest.raises(ValidationError, match="at most one of param_grid"):
        session.nested_cv_score(
            LogisticRegression(max_iter=200),
            param_grid={"C": [0.5, 1.0]},
            param_distributions={"C": [0.5, 1.0]},
            outer_cv=2,
            inner_cv=2,
        )


def test_nested_cv_recipe_knobs_recorded_without_test_peek() -> None:
    rng = np.random.default_rng(11)
    n = 140
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "d": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {"a": "feature", "b": "feature", "c": "feature", "d": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    result = session.nested_cv_score(
        LogisticRegression(max_iter=400),
        param_grid={"C": [0.5, 2.0]},
        recipe_grid={"select_k": [1, 2]},
        outer_cv=2,
        inner_cv=2,
        preprocess=PreprocessRecipe(
            impute=None,
            scale="standard",
            select="univariate",
            select_score_func="f_classif",
            select_k=2,
        ),
    )
    assert result.n_outer_splits == 2
    assert "selected_recipe_knobs_by_fold" in result.inner_selection_summary
    for fold in result.outer_folds:
        assert "select_k" in fold.best_recipe_knobs
        assert fold.best_recipe_knobs["select_k"] in {1, 2}
        assert fold.best_params
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]
    assert session.fit_result is None


def test_nested_cv_with_fold_local_preprocess() -> None:
    frame = _cls_frame(100)
    frame.loc[::8, "x1"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, stratify=True, random_state=2)
    )
    result = session.nested_cv_score(
        LogisticRegression(max_iter=400),
        param_grid={"C": [0.5, 2.0]},
        outer_cv=2,
        inner_cv=2,
        preprocess=PreprocessRecipe(impute="median", scale="standard"),
    )
    assert result.fold_preprocess is not None
    assert result.fold_preprocess["impute"] == "median"
    assert any("Fold-local" in tip for tip in result.limitations)


def test_nested_cv_randomized_inner() -> None:
    session = (
        Session.ingest(_cls_frame(90))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, stratify=True, random_state=3)
    )
    result = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        param_distributions={"max_depth": [2, 3, 4], "min_samples_leaf": [1, 2, 5]},
        n_iter=3,
        outer_cv=2,
        inner_cv=2,
        random_state=0,
    )
    assert result.search_method == "randomized"
    assert all(fold.inner_n_trials == 3 for fold in result.outer_folds)
