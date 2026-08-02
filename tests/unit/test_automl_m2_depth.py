"""Deeper AutoML checks: nested selection, explain, walkthrough, audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml import Session
from buildml.automl.types import AutoMLBudget
from buildml.preprocess.fold import PreprocessRecipe


def _frame(n: int = 140, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (1.1 * x1 - 0.3 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_nested_selection_records_outer_estimate() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    )
    result = session.run_automl(
        method="randomized",
        selection="nested",
        n_trials=6,
        cv=2,
        outer_cv=2,
        families=("logistic", "random_forest"),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None, scale="standard"),
        budget=AutoMLBudget(max_trials=6),
        random_state=0,
    )
    assert result.selection == "nested"
    assert session.automl_plan is not None
    # Outer estimate may be None only if all folds failed; expect values here.
    assert result.outer_score_mean is not None
    assert session.automl_plan.outer_score_mean is not None


def test_explain_before_run_automl() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    before = session.explain("run_automl", moment="before")
    assert before.operation == "run_automl"
    assert before.prerequisite_status.get("split") is True


def test_walkthrough_and_audit_include_automl() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    session.run_automl(
        n_trials=4,
        cv=2,
        families=("logistic",),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None, scale=None),
        random_state=0,
    )
    walk = session.walkthrough()
    assert walk.automl_status.get("has_automl_plan") is True
    assert walk.automl_status.get("best_family") == "logistic"

    summary = session.summarize_history()
    ops = {r.get("operation_id") or r.get("action") for r in session.history}
    assert "run_automl" in ops
    assert summary.n_operations >= 1


def test_regression_families() -> None:
    rng = np.random.default_rng(4)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.25, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    result = session.run_automl(
        task="regression",
        n_trials=6,
        cv=3,
        families=("ridge", "random_forest"),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None, scale="standard"),
        random_state=0,
    )
    assert result.task == "regression"
    assert result.best_family in {"ridge", "random_forest"}
    metrics = session.evaluate_automl(partition="test")
    assert "r2" in metrics.metrics or "mae" in metrics.metrics
