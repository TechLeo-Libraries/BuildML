"""Unit coverage for the active-learning thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.data.dataset import Dataset
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.ingest.detect import schema_from_dataframe
from buildml.activelearning.checkpoint import BUNDLE_FORMAT, load_active_learning_bundle


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(n // 2, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask_train(session: Session, fraction: float = 0.8, seed: int = 1) -> tuple[Session, pd.Series]:
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    truth = full["label"].copy()
    train_idx = list(session.split_plan.train_indices)
    blank = rng.choice(train_idx, size=max(1, int(fraction * len(train_idx))), replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session, truth


def _ready_session(*, mask: bool = True) -> tuple[Session, pd.Series | None]:
    session = (
        Session.ingest(_frame())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    if mask:
        return _mask_train(session)
    return session, None


def test_core_import_and_catalog() -> None:
    import buildml.activelearning as al

    assert hasattr(al, "fit_active_learner")
    assert hasattr(Session, "fit_active_learner")
    for name in (
        "fit_active_learner",
        "suggest_query",
        "label_rows",
        "evaluate_active_learning",
        "save_active_learning_bundle",
        "load_active_learning_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert (
        "activelearning-train-pool"
        in OPERATION_CATALOG["fit_active_learner"].concept_links
    )
    assert (
        "activelearning-bundle-boundary"
        in OPERATION_CATALOG["save_active_learning_bundle"].concept_links
    )


def test_fit_requires_split() -> None:
    session = Session.ingest(_frame()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_active_learner()


def test_margin_loop_evaluate_bundle(tmp_path: Path) -> None:
    session, truth = _ready_session()
    assert truth is not None
    fit = session.fit_active_learner(
        strategy="margin",
        base_estimator="logistic_regression",
        batch_size=6,
        label_budget=18,
    )
    assert fit.n_unlabeled_pool > 0
    assert fit.n_labeled_train >= 2
    assert session.activelearning_plan is not None

    q = session.suggest_query(batch_size=6)
    assert len(q.indices) == 6
    # Simulated oracle for tests only.
    labels = [int(truth.loc[i]) for i in q.indices]
    labeled = session.label_rows(indices=q.indices, labels=labels)
    assert labeled.n_newly_labeled == 6
    assert labeled.refit is True
    assert session.activelearning_plan.n_queries_used == 6

    ev = session.evaluate_active_learning(partition="test")
    assert ev.n_labeled_eval == ev.n_rows
    assert "accuracy" in ev.metrics
    assert ev.metrics["accuracy"] >= 0.5

    bundle = session.save_active_learning_bundle(tmp_path / "al")
    assert (bundle / "meta.json").is_file()
    plan = load_active_learning_bundle(bundle)
    assert plan.strategy == "margin"
    import json

    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == BUNDLE_FORMAT


def test_strategies_and_ai_allowlist() -> None:
    session, truth = _ready_session()
    assert truth is not None
    session.fit_active_learner(strategy="entropy", label_budget=20)
    q = session.suggest_query(batch_size=4, strategy="least_confidence")
    assert len(q.indices) == 4

    session2, _ = _ready_session()
    session2.fit_active_learner(strategy="committee", committee_size=4, label_budget=12)
    q2 = session2.suggest_query(batch_size=3)
    assert len(q2.indices) == 3
    assert q2.strategy == "committee"

    session3, _ = _ready_session()
    session3.fit_active_learner(strategy="expected_model_change_lite", label_budget=12)
    q3 = session3.suggest_query(batch_size=3)
    assert len(q3.indices) == 3

    registry = build_default_registry()
    for name in (
        "fit_active_learner",
        "suggest_query",
        "evaluate_active_learning",
        "save_active_learning_bundle",
        "load_active_learning_bundle",
    ):
        assert registry.get(name) is not None
    # label_rows stays Session-primary (human labels), not AI-allowlisted.
    assert registry.get("label_rows") is None


def test_refuse_test_indices_and_budget() -> None:
    session, truth = _ready_session()
    assert truth is not None
    session.fit_active_learner(strategy="margin", label_budget=2)
    test_idx = list(session.split_plan.test_indices)[0]
    with pytest.raises(ValidationError, match="validation/test"):
        session.label_rows(indices=[test_idx], labels=[0])

    q = session.suggest_query(batch_size=5)
    # Budget remaining is 2, so suggest_query should clamp.
    assert len(q.indices) <= 2
    labels = [int(truth.loc[i]) for i in q.indices]
    session.label_rows(indices=q.indices, labels=labels)
    q2 = session.suggest_query(batch_size=5)
    assert q2.indices == ()
    assert q2.budget_remaining == 0


def test_walkthrough_status() -> None:
    session, _ = _ready_session()
    session.fit_active_learner(strategy="margin")
    report = session.walkthrough()
    status = report.activelearning_status
    assert status["enabled"] is True
    assert status["strategy"] == "margin"
