"""Ensemble alpha-gate smoke: prep → vote/stack/blend → eval → bundle → pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session


def test_ensemble_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 160
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (0.9 * x1 - 0.4 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    bases = {
        "lr": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=40, random_state=0),
    }

    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )

    vote = session.fit_voting(bases, voting="soft", task="classification")
    assert vote.strategy == "voting"
    vote_eval = session.evaluate_ensemble(partition="validation")
    assert vote_eval.n_rows > 0

    stack = session.fit_stacking(bases, cv=3, task="classification")
    assert stack.strategy == "stacking"
    assert session.ensemble_plan is not None
    stack_eval = session.evaluate(partition="test")
    assert "accuracy" in stack_eval.metrics or "f1_weighted" in stack_eval.metrics

    blend = session.fit_blending(
        bases, holdout_fraction=0.2, random_state=0, task="classification"
    )
    assert blend.strategy == "blending"
    blend_eval = session.evaluate_ensemble(partition="test")
    assert blend_eval.diagnostics.get("ensemble", {}).get("strategy") == "blending"

    before = session.explain("fit_stacking", moment="before")
    assert before.operation == "fit_stacking"
    assert before.prerequisite_status.get("split") is True

    ens_bundle = session.save_ensemble_bundle(tmp_path / "ensemble_bundle")
    assert (ens_bundle / "meta.json").is_file()

    pipeline = session.save_pipeline(tmp_path / "ensemble_pipeline", evaluate_partition="test")
    assert (pipeline / "meta.json").is_file()

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    restored.load_ensemble_bundle(ens_bundle, trusted=True)
    again = restored.evaluate_ensemble(partition="test")
    assert again.n_rows == blend_eval.n_rows
