"""Optuna search path (optional extra)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.preprocess.fold import PreprocessRecipe


def _cls_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),
            "y": ([0, 1] * (n // 2)),
        }
    )


def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def test_optuna_missing_extra_message() -> None:
    if _has_optuna():
        pytest.skip("optuna installed in this environment")
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(MissingExtraError, match="buildml\\[optuna\\]"):
        session.optuna_search(
            LogisticRegression(max_iter=200),
            param_space={"C": {"type": "float", "low": 0.1, "high": 2.0, "log": True}},
            n_trials=2,
            cv=2,
            refit=False,
        )


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_optuna_search_ranks_and_respects_train_only() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    result = session.optuna_search(
        DecisionTreeClassifier(random_state=0),
        param_space={
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "min_samples_leaf": [1, 2, 4],
        },
        n_trials=4,
        cv=3,
        random_state=0,
        refit=True,
    )
    assert result.method == "optuna"
    assert len(result.trials) == 4
    assert result.best_params
    assert result.best_score is not None
    assert session.fit_result is not None
    assert result.best_cv is not None
    assert "test" in result.best_cv.held_out_partitions


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_optuna_recipe_space_with_preprocess() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
    )
    recipe = PreprocessRecipe(scale="standard", select="variance", select_k=2)
    result = session.optuna_search(
        LogisticRegression(max_iter=300),
        param_space={"C": {"type": "float", "low": 0.5, "high": 2.0}},
        recipe_space={"select_k": {"type": "int", "low": 1, "high": 3}},
        preprocess=recipe,
        n_trials=3,
        cv=2,
        random_state=2,
        refit=False,
    )
    assert result.best_recipe_knobs
    assert "select_k" in result.best_recipe_knobs
