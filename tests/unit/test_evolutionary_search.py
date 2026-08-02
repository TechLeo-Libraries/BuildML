"""Evolutionary GA hyperparameter search (in-tree NumPy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.preprocess.fold import PreprocessRecipe


def _cls_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),
            "y": ([0, 1] * (n // 2)),
        }
    )


def test_evolutionary_search_ranks_and_respects_train_only() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    result = session.evolutionary_search(
        DecisionTreeClassifier(random_state=0),
        param_space={
            "max_depth": {"type": "int", "low": 2, "high": 5},
            "min_samples_leaf": [1, 2, 4],
        },
        population_size=6,
        n_generations=3,
        elite_size=1,
        max_evaluations=12,
        cv=3,
        random_state=0,
        refit=True,
    )
    assert result.method == "evolutionary"
    assert result.best_params
    assert result.best_score is not None
    assert len(result.trials) >= 1
    assert len(result.trials) <= 12
    assert session.fit_result is not None
    assert session.last_search is result
    assert result.best_cv is not None
    assert "test" in result.best_cv.held_out_partitions
    assert isinstance(result.study, dict)
    assert result.study.get("kind") == "evolutionary"
    assert result.study.get("generation_best")
    assert result.study.get("n_evaluations", 0) >= 1


def test_evolutionary_search_reproducible() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
    )
    kwargs = dict(
        estimator=DecisionTreeClassifier(random_state=0),
        param_space={"max_depth": {"type": "int", "low": 2, "high": 4}},
        population_size=5,
        n_generations=2,
        elite_size=1,
        max_evaluations=8,
        cv=2,
        random_state=7,
        refit=False,
    )
    a = session.evolutionary_search(**kwargs)
    b = session.evolutionary_search(**kwargs)
    assert a.best_params == b.best_params
    assert a.best_score == pytest.approx(b.best_score)


def test_evolutionary_recipe_space_with_preprocess() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
    )
    recipe = PreprocessRecipe(scale="standard", select="variance", select_k=2)
    result = session.evolutionary_search(
        LogisticRegression(max_iter=300),
        param_space={"C": {"type": "float", "low": 0.5, "high": 2.0, "log": True}},
        recipe_space={"select_k": {"type": "int", "low": 1, "high": 3}},
        preprocess=recipe,
        population_size=4,
        n_generations=2,
        elite_size=1,
        max_evaluations=6,
        cv=2,
        random_state=2,
        refit=False,
    )
    assert result.best_recipe_knobs
    assert "select_k" in result.best_recipe_knobs


def test_evolutionary_refuses_session_global_preprocess() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .impute(strategy="median")
        .scale(method="standard")
    )
    with pytest.raises(LeakageError, match="allow_session_global_preprocess=True|already"):
        session.evolutionary_search(
            DecisionTreeClassifier(random_state=0),
            param_space={"max_depth": {"type": "int", "low": 2, "high": 4}},
            population_size=4,
            n_generations=2,
            elite_size=1,
            max_evaluations=4,
            cv=2,
            refit=False,
        )


def test_evolutionary_rejects_callable_space() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="declare-style dict"):
        session.evolutionary_search(
            DecisionTreeClassifier(random_state=0),
            param_space=lambda trial: {"max_depth": 3},  # type: ignore[arg-type]
            population_size=4,
            n_generations=1,
            elite_size=1,
            refit=False,
        )


def test_evolutionary_nested_cv_inner_search() -> None:
    session = (
        Session.ingest(_cls_frame(n=90))
        .set_roles({"x1": "feature", "x2": "feature", "x3": "feature", "y": "target"})
        .split(test_size=0.2, stratify=True, random_state=0)
    )
    nested = session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        param_space={
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "min_samples_leaf": [1, 2],
        },
        inner_search="evolutionary",
        n_trials=8,
        population_size=4,
        n_generations=2,
        outer_cv=2,
        inner_cv=2,
        random_state=0,
    )
    assert nested.search_method == "evolutionary"
    assert nested.mean_metrics
    assert nested.outer_folds
    assert all(fold.best_params for fold in nested.outer_folds)


def test_evolutionary_ai_tool_registered() -> None:
    from buildml.ai.tools import registered_tool_names

    assert "evolutionary_search" in registered_tool_names()
