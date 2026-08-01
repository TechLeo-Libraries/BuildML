"""Classical alpha-gate smoke: ingest through predict_from_pipeline (core-only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.preprocess import PreprocessRecipe


def _frame(n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * (n // 2))
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, size=n).round(1),
            "income": rng.normal(60, 15, size=n).round(1),
            "approved": labels,
        }
    )


def test_classical_alpha_gate_smoke(tmp_path: Path) -> None:
    frame = _frame()
    # Introduce a few train-visible nulls after ingest for impute coverage.
    frame.loc[[0, 3, 8], "age"] = np.nan

    session = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "approved": "target"})
    )
    eda = session.eda(include_plots=False)
    assert eda is not None

    session = session.split(test_size=0.25, stratify=True, random_state=0)
    assert session.split_plan is not None
    test_idx = list(session.split_plan.test_indices)

    recipe = PreprocessRecipe(impute="median", scale="standard")
    cv = session.cv_score(
        LogisticRegression(max_iter=400),
        cv=3,
        preprocess=recipe,
    )
    assert cv.n_splits == 3
    assert cv.scoring_metric in cv.mean_metrics
    assert set(session.split_plan.test_indices) == set(test_idx)

    search = session.grid_search(
        DecisionTreeClassifier(random_state=0),
        param_grid={"max_depth": [2, 3], "min_samples_leaf": [1, 2]},
        cv=3,
        preprocess=recipe,
        refit=False,
    )
    assert search.best_params
    assert "max_depth" in search.best_params

    session = (
        session.impute(strategy="median")
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=400), task="classification")
    )
    evaluation = session.evaluate(partition="test")
    assert "f1_weighted" in evaluation.metrics
    before = evaluation.metrics["f1_weighted"]

    ckpt = tmp_path / "ckpt"
    pipe = tmp_path / "pipe"
    session.checkpoint_save(ckpt)
    session.save_pipeline(pipe, evaluate_partition="test", title="Alpha smoke")

    restored = Session.checkpoint_load(ckpt)
    assert restored.split_plan is not None
    assert restored.impute_plan is not None
    assert restored.fit_result is None

    holdout = frame.iloc[test_idx].reset_index(drop=True)
    scored = Session().predict_from_pipeline(
        pipe,
        holdout,
        roles={"age": "feature", "income": "feature", "approved": "target"},
    )
    assert scored.n_rows == len(holdout)
    assert len(scored.predictions) == len(holdout)

    reloaded = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "approved": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .load_pipeline(pipe)
    )
    reloaded.apply_preprocess_plans()
    after = reloaded.evaluate(partition="test").metrics["f1_weighted"]
    assert after == pytest.approx(before)
