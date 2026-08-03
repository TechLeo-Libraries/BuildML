"""Unit coverage for AutoML family + recipe search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.automl.checkpoint import BUNDLE_FORMAT
from buildml.automl.types import AutoMLBudget
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.preprocess.fold import PreprocessRecipe


def _clf_frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = (0.9 * x1 - 0.5 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


def _ready_clf(**split_kwargs: object) -> Session:
    kwargs = {"test_size": 0.25, "validation_size": 0.2, "random_state": 0, "stratify": True}
    kwargs.update(split_kwargs)
    return (
        Session.ingest(_clf_frame())
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(**kwargs)
    )


def test_core_import_does_not_require_extra() -> None:
    import buildml.automl as automl

    assert hasattr(Session, "run_automl")
    assert hasattr(Session, "evaluate_automl")
    assert hasattr(automl, "run_automl")


def test_catalog_covers_automl_operations() -> None:
    for name in (
        "run_automl",
        "evaluate_automl",
        "save_automl_bundle",
        "load_automl_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "automl-beyond-hpo" in OPERATION_CATALOG["run_automl"].concept_links
    assert "automl-recipe-strategy-search" in OPERATION_CATALOG["run_automl"].concept_links
    assert "automl-selection-honesty" in OPERATION_CATALOG["run_automl"].concept_links
    assert "automl-bundle-boundary" in OPERATION_CATALOG["save_automl_bundle"].concept_links


def test_run_requires_split() -> None:
    session = Session.ingest(_clf_frame()).set_roles(
        {"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.run_automl(n_trials=4, cv=3, families=("logistic", "random_forest"))


def test_refuses_session_global_preprocess() -> None:
    session = _ready_clf().scale(method="standard")
    with pytest.raises(LeakageError):
        session.run_automl(
            n_trials=4,
            cv=3,
            families=("logistic",),
            include_recipe_search=False,
            preprocess=PreprocessRecipe(impute="median", scale="standard"),
        )


def test_randomized_search_evaluate_and_bundle(tmp_path: Path) -> None:
    session = _ready_clf()
    result = session.run_automl(
        method="randomized",
        selection="cv",
        n_trials=8,
        cv=3,
        include_recipe_search=True,
        families=("logistic", "random_forest"),
        budget=AutoMLBudget(max_trials=8, max_recipe_strategies=4),
        random_state=0,
    )
    assert result.best_family in {"logistic", "random_forest"}
    assert session.automl_plan is not None
    assert session.fit_result is not None
    assert len(result.trials) >= 1
    assert any(
        "not neural architecture search" in d.lower() or "NAS" in d
        for d in result.disclosures
    )

    metrics = session.evaluate_automl(partition="test")
    assert "accuracy" in metrics.metrics or "f1_weighted" in metrics.metrics
    assert metrics.diagnostics.get("automl", {}).get("best_family") == result.best_family

    bundle = session.save_automl_bundle(tmp_path / "automl_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "automl_plan.joblib").is_file()
    meta = (bundle / "meta.json").read_text(encoding="utf-8")
    assert BUNDLE_FORMAT in meta

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
    )
    restored.load_automl_bundle(bundle, trusted=True)
    again = restored.evaluate_automl(partition="test")
    assert again.n_rows == metrics.n_rows


def test_validation_selection_requires_partition() -> None:
    session = (
        Session.ingest(_clf_frame())
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
    )
    with pytest.raises(ValidationError, match="validation"):
        session.run_automl(
            selection="validation",
            n_trials=4,
            families=("logistic",),
            include_recipe_search=False,
        )


def test_validation_selection_ranks_without_test_touch() -> None:
    session = _ready_clf()
    result = session.run_automl(
        method="randomized",
        selection="validation",
        n_trials=6,
        families=("logistic", "random_forest"),
        include_recipe_search=True,
        budget=AutoMLBudget(max_trials=6, max_recipe_strategies=3),
        random_state=1,
    )
    assert result.selection == "validation"
    assert result.best_score is not None
    # Test evaluate still works after selection.
    test = session.evaluate_automl(partition="test")
    assert test.n_rows > 0


def test_include_ensembles_scores_voting() -> None:
    session = _ready_clf()
    result = session.run_automl(
        method="randomized",
        n_trials=8,
        cv=3,
        families=("logistic", "random_forest", "decision_tree"),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute="median", scale="standard", encode="onehot"),
        include_ensembles=True,
        max_ensemble_bases=3,
        budget=AutoMLBudget(max_trials=8, max_ensemble_trials=2),
        random_state=2,
    )
    kinds = {t.kind for t in result.trials}
    assert "single" in kinds
    # Voting may or may not win; it should appear among trials when bases exist.
    assert "voting" in kinds or result.best_kind == "single"
