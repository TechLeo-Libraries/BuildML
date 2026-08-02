"""R5 AutoML industry depth: backends, catalog, export, evolutionary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.automl.catalog import automl_capability_matrix, backend_available, list_automl_methods
from buildml.automl.extras import (
    autogluon_available,
    flaml_available,
    gradient_boosting_extras_available,
    optuna_available,
)
from buildml.automl.search import export_comparison_metrics
from buildml.automl.types import AutoMLBudget
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.preprocess.fold import PreprocessRecipe


def _clf_frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = (0.9 * x1 - 0.5 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


def _ready(**split_kwargs: object) -> Session:
    kwargs = {"test_size": 0.25, "validation_size": 0.2, "random_state": 0, "stratify": True}
    kwargs.update(split_kwargs)
    return (
        Session.ingest(_clf_frame())
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(**kwargs)
    )


def test_capability_matrix_native_always_available() -> None:
    matrix = automl_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert "randomized" in list_automl_methods(backend="native")
    assert backend_available("native") is True


def test_evolutionary_method_runs_without_extra() -> None:
    session = _ready()
    result = session.run_automl(
        backend="native",
        method="evolutionary",
        n_trials=6,
        cv=2,
        families=("logistic", "random_forest"),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute="median", scale="standard"),
        budget=AutoMLBudget(max_trials=6),
        random_state=0,
    )
    assert result.method == "evolutionary"
    assert len(result.trials) >= 1


def test_export_comparison_metrics(tmp_path: Path) -> None:
    session = _ready()
    result = session.run_automl(
        n_trials=6,
        cv=2,
        families=("logistic", "random_forest"),
        include_recipe_search=True,
        budget=AutoMLBudget(max_trials=6, max_recipe_strategies=3),
        random_state=0,
    )
    assert len(result.trials) >= 1
    out = export_comparison_metrics(result, tmp_path / "compare.json")
    assert out.is_file()
    assert "trials" in out.read_text(encoding="utf-8")


def test_stacking_ensemble_mode() -> None:
    session = _ready()
    result = session.run_automl(
        n_trials=8,
        cv=3,
        families=("logistic", "random_forest", "decision_tree"),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute="median", scale="standard", encode="onehot"),
        include_ensembles=True,
        ensemble_mode="stacking",
        max_ensemble_bases=3,
        budget=AutoMLBudget(max_trials=8, max_ensemble_trials=2),
        random_state=1,
    )
    kinds = {t.kind for t in result.trials}
    assert "single" in kinds
    assert "stacking" in kinds or result.best_kind == "single"


@pytest.mark.skipif(not optuna_available(), reason="buildml[automl] not installed")
def test_optuna_backend_with_budget() -> None:
    session = _ready()
    result = session.run_automl(
        backend="optuna",
        n_trials=4,
        cv=2,
        families=("logistic", "random_forest"),
        budget=AutoMLBudget(
            max_trials=4,
            enable_pruning=True,
            max_time_seconds=120.0,
        ),
        random_state=0,
    )
    assert any("Optuna" in d or "optuna" in d.lower() for d in result.disclosures)


@pytest.mark.skipif(not gradient_boosting_extras_available(), reason="GBDT extras missing")
def test_industry_families_in_native_catalog() -> None:
    session = _ready()
    result = session.run_automl(
        n_trials=4,
        cv=2,
        include_industry_families=True,
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute="median", scale="standard"),
        random_state=0,
    )
    searched = set(result.families_searched)
    assert searched.intersection({"lightgbm", "xgboost", "catboost"})


@pytest.mark.skipif(not flaml_available(), reason="buildml[automl-industry] FLAML missing")
def test_flaml_backend_train_only() -> None:
    session = _ready()
    result = session.run_automl(
        backend="flaml",
        selection="validation",
        time_budget=30.0,
        random_state=0,
    )
    assert result.best_family == "flaml"
    assert any("flaml" in d.lower() for d in result.disclosures)
    test = session.evaluate_automl(partition="test")
    assert test.n_rows > 0


@pytest.mark.skipif(not autogluon_available(), reason="buildml[automl-industry] AutoGluon missing")
def test_autogluon_backend_train_only() -> None:
    session = _ready()
    result = session.run_automl(
        backend="autogluon",
        selection="validation",
        time_budget=45.0,
        random_state=0,
    )
    assert result.best_family == "autogluon"
    test = session.evaluate_automl(partition="test")
    assert test.n_rows > 0


def test_flaml_nested_rejected_without_extra() -> None:
    if flaml_available():
        pytest.skip("FLAML installed — use integration env for nested rejection")
    session = _ready()
    with pytest.raises((MissingExtraError, ValidationError)):
        session.run_automl(backend="flaml", selection="nested", time_budget=10.0)
