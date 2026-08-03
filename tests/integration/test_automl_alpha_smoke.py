"""AutoML alpha-gate smoke: search → eval → bundle → pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.automl.types import AutoMLBudget


def test_automl_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    n = 180
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = (0.85 * x1 - 0.45 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )

    result = session.run_automl(
        method="randomized",
        selection="cv",
        n_trials=10,
        cv=3,
        include_recipe_search=True,
        include_ensembles=True,
        families=("logistic", "random_forest", "gradient_boosting"),
        budget=AutoMLBudget(max_trials=10, max_recipe_strategies=5, max_ensemble_trials=2),
        random_state=0,
    )
    assert result.best_family
    assert session.automl_plan is not None

    val = session.evaluate_automl(partition="validation")
    assert val.n_rows > 0
    test = session.evaluate(partition="test")
    assert "accuracy" in test.metrics or "f1_weighted" in test.metrics

    before = session.explain("run_automl", moment="before")
    assert before.operation == "run_automl"

    automl_bundle = session.save_automl_bundle(tmp_path / "automl_bundle")
    assert (automl_bundle / "meta.json").is_file()

    pipeline = session.save_pipeline(tmp_path / "automl_pipeline", evaluate_partition="test")
    assert (pipeline / "meta.json").is_file()

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    restored.load_automl_bundle(automl_bundle, trusted=True)
    again = restored.evaluate_automl(partition="test")
    assert again.n_rows == test.n_rows
