"""Nested CV with Optuna inner search (optional extra)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.preprocess import PreprocessRecipe


def _cls_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.3 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def test_nested_cv_optuna_missing_extra_message() -> None:
    if _has_optuna():
        pytest.skip("optuna installed in this environment")
    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(MissingExtraError, match="buildml\\[optuna\\]"):
        session.nested_cv_score(
            DecisionTreeClassifier(random_state=0),
            inner_search="optuna",
            param_space={"max_depth": {"type": "int", "low": 2, "high": 4}},
            n_trials=2,
            outer_cv=2,
            inner_cv=2,
        )


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_nested_cv_optuna_inner_records_params_and_holds_out_test() -> None:
    session = (
        Session.ingest(_cls_frame(120))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
    )
    test_idx = set(session.split_plan.test_indices)  # type: ignore[union-attr]
    valid_idx = set(session.split_plan.validation_indices)  # type: ignore[union-attr]

    result = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        inner_search="optuna",
        param_space={
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "min_samples_leaf": [1, 2, 4],
        },
        n_trials=3,
        outer_cv=2,
        inner_cv=2,
        random_state=0,
        scoring_metric="f1_weighted",
    )
    assert result.search_method == "optuna"
    assert result.n_outer_splits == 2
    assert "test" in result.held_out_partitions
    assert "validation" in result.held_out_partitions
    for fold in result.outer_folds:
        assert fold.best_params
        assert "max_depth" in fold.best_params
        assert fold.inner_n_trials == 3
        assert "f1_weighted" in fold.metrics
    assert set(session.split_plan.test_indices) == test_idx  # type: ignore[union-attr]
    assert set(session.split_plan.validation_indices) == valid_idx  # type: ignore[union-attr]
    assert session.fit_result is None


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_nested_cv_optuna_recipe_space_recorded() -> None:
    session = (
        Session.ingest(_cls_frame(110))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, stratify=True, random_state=1)
    )
    result = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        param_space={"max_depth": {"type": "int", "low": 2, "high": 3}},
        recipe_space={"select_k": {"type": "int", "low": 1, "high": 2}},
        n_trials=2,
        outer_cv=2,
        inner_cv=2,
        preprocess=PreprocessRecipe(scale="standard", select="variance", select_k=2),
        random_state=1,
    )
    assert result.search_method == "optuna"
    for fold in result.outer_folds:
        assert "select_k" in fold.best_recipe_knobs
        assert fold.best_params


def test_nested_cv_optuna_rejects_mixed_spaces() -> None:
    session = (
        Session.ingest(_cls_frame(60))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
    )
    with pytest.raises(ValidationError, match="either Optuna spaces"):
        session.nested_cv_score(
            DecisionTreeClassifier(random_state=0),
            param_grid={"max_depth": [2, 3]},
            param_space={"max_depth": {"type": "int", "low": 2, "high": 3}},
            outer_cv=2,
            inner_cv=2,
        )
