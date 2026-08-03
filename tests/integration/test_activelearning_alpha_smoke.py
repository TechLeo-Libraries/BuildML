"""Active-learning alpha-gate smoke: split → mask train → fit → query → label → eval → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe


def test_activelearning_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    x0 = rng.normal([-1.0, -1.0], 0.6, size=(150, 2))
    x1 = rng.normal([1.3, 1.1], 0.6, size=(150, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 150 + [1] * 150
    truth = frame["label"].copy()

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    blank = rng.choice(train_idx, size=int(0.8 * len(train_idx)), replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )

    fit = session.fit_active_learner(
        strategy="margin",
        batch_size=5,
        label_budget=15,
    )
    assert fit.n_unlabeled_pool > 0
    assert session.activelearning_plan is not None

    q = session.suggest_query(batch_size=5)
    assert len(q.indices) == 5
    # Simulated oracle for the integration smoke only.
    labels = [int(truth.loc[i]) for i in q.indices]
    labeled = session.label_rows(indices=q.indices, labels=labels)
    assert labeled.n_newly_labeled == 5
    assert session.activelearning_plan.n_queries_used == 5

    ev = session.evaluate_active_learning(partition="validation")
    assert ev.partition == "validation"
    assert "accuracy" in ev.metrics
    assert session.activelearning_eval_result is not None

    before = session.explain("suggest_query", moment="before")
    assert before.prerequisite_status.get("activelearning-plan") is True

    bundle = session.save_active_learning_bundle(tmp_path / "al_bundle")
    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_active_learning_bundle(bundle, trusted=True)
    assert restored.activelearning_plan is not None
    assert restored.activelearning_plan.n_queries_used == 5
    again = restored.suggest_query(batch_size=3)
    assert len(again.indices) <= 3
