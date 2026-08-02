"""Deeper active-learning coverage: low-level API, budget, explain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from buildml.activelearning.evaluate import evaluate_active_learning
from buildml.activelearning.fit import fit_active_learner
from buildml.activelearning.label import label_rows
from buildml.activelearning.query import suggest_query


def _frame(n: int = 160, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.8, size=(n // 2, 2))
    x1 = rng.normal(2.5, 0.8, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask(session: Session, fraction: float = 0.75) -> tuple[Session, pd.Series]:
    rng = np.random.default_rng(4)
    full = session.to_pandas().copy()
    truth = full["y"].copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session, truth


def test_low_level_fit_query_label_evaluate(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, truth = _mask(session)
    plan, fit = fit_active_learner(
        session.dataset,
        session.split_plan,
        strategy="margin",
        prefer_reduce_components=False,
        label_budget=10,
    )
    assert fit.n_unlabeled_pool > 0
    q = suggest_query(session.dataset, plan, session.split_plan, batch_size=4)
    assert len(q.indices) == 4
    labels = [int(truth.loc[i]) for i in q.indices]
    new_ds, new_plan, lab, fit2 = label_rows(
        session.dataset,
        plan,
        session.split_plan,
        indices=q.indices,
        labels=labels,
        refit=True,
    )
    assert lab.n_newly_labeled == 4
    assert fit2 is not None
    ev = evaluate_active_learning(
        new_ds, new_plan, session.split_plan, partition="test"
    )
    assert ev.metrics["accuracy"] >= 0.0

    from buildml.activelearning.checkpoint import save_active_learning_bundle

    out = save_active_learning_bundle(
        tmp_path / "direct", new_plan, fit_result=fit2, eval_result=ev
    )
    assert (out / "activelearning_plan.joblib").is_file()


def test_needs_labeled_rows() -> None:
    frame = _frame()
    frame["y"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="at least 2 labeled"):
        session.fit_active_learner()


def test_null_label_refused() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, _ = _mask(session)
    session.fit_active_learner(label_budget=5)
    q = session.suggest_query(batch_size=1)
    with pytest.raises(ValidationError, match="null label"):
        session.label_rows(indices=q.indices, labels=[np.nan])


def test_explain_before() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    before = session.explain("fit_active_learner", moment="before")
    assert before.operation == "fit_active_learner"
    assert before.prerequisite_status.get("split") is True
