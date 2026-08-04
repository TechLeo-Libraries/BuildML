"""HIGH-depth AutoML reporting: leaderboard fields, nested disclosure, catalog honesty."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from buildml import Session
from buildml.automl import automl_capability_matrix, export_comparison_metrics
from buildml.automl.types import AutoMLBudget
from buildml.preprocess.fold import PreprocessRecipe


def _frame(n: int = 130, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (1.0 * x1 - 0.35 * x2 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_leaderboard_rich_fields_and_default_cv_disclosure() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    )
    result = session.automl.run(
        method="randomized",
        selection="cv",
        n_trials=6,
        cv=2,
        families=("logistic", "random_forest"),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None, scale="standard"),
        budget=AutoMLBudget(max_trials=6),
        random_state=0,
    )
    assert result.selection == "cv"
    board = result.leaderboard()
    assert not board.empty
    for col in (
        "rank",
        "family",
        "recipe_strategy",
        "mean_score",
        "gap_to_best",
        "selection",
        "outer_score_mean",
        "nested_cv_disclosed",
        "ranking_metric",
    ):
        assert col in board.columns
    assert board["selection"].iloc[0] == "cv"
    assert bool(board["nested_cv_disclosed"].iloc[0]) is False
    assert int(board["rank"].iloc[0]) == 1
    joined = " ".join(result.disclosures + result.recommendations).lower()
    assert "selection='cv'" in joined or "default selection" in joined
    assert "nested" in joined


def test_nested_leaderboard_marks_outer_estimate() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=1)
    )
    result = session.automl.run(
        method="randomized",
        selection="nested",
        n_trials=4,
        cv=2,
        outer_cv=2,
        families=("logistic",),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None, scale="standard"),
        budget=AutoMLBudget(max_trials=4),
        random_state=1,
    )
    board = result.leaderboard(top_n=3)
    assert bool(board["nested_cv_disclosed"].iloc[0]) is True
    assert result.outer_score_mean is not None
    assert board["outer_score_mean"].iloc[0] == result.outer_score_mean


def test_export_comparison_includes_leaderboard(tmp_path) -> None:
    session = (
        Session.ingest(_frame(100))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
    )
    result = session.automl.run(
        n_trials=4,
        cv=2,
        families=("logistic",),
        include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None, scale=None),
        budget=AutoMLBudget(max_trials=4),
        random_state=2,
    )
    path = export_comparison_metrics(result, tmp_path / "trials.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "leaderboard" in payload
    assert payload["nested_cv_disclosed"] is False
    assert "default_selection_note" in payload
    assert "nested" in payload["default_selection_note"].lower()


def test_catalog_selection_modes_and_industry_probe_honesty() -> None:
    matrix = automl_capability_matrix()
    assert matrix["default_selection"] == "cv"
    assert matrix["selection_modes"]["nested"]["prominent"] is True
    assert "leaderboard_fields" in matrix["reporting"]
    honesty = matrix["industry_import_honesty"].lower()
    assert "subprocess" in honesty
    assert "find_spec" in honesty
