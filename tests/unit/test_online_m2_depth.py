"""Deeper online-learning coverage: low-level API, classes, regression, fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.online.checkpoint import save_online_bundle
from buildml.online.evaluate import evaluate_online
from buildml.online.fit import fit_online
from buildml.online.update import partial_fit_online


def _clf_frame(n: int = 180, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.8, size=(n // 2, 2))
    x1 = rng.normal(2.5, 0.8, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _reg_frame(n: int = 160, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, 2))
    frame = pd.DataFrame(x, columns=["a", "b"])
    frame["y"] = 1.5 * frame["a"] - 0.7 * frame["b"] + rng.normal(0, 0.2, size=n)
    return frame


def test_low_level_fit_update_evaluate(tmp_path: Path) -> None:
    session = (
        Session.ingest(_clf_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_online(
        session.dataset,
        session.split_plan,
        estimator="passive_aggressive_classifier",
        chunk_size=35,
        n_init=35,
        classes=[0, 1],
        prefer_reduce_components=False,
    )
    assert fit.n_init_rows == 35
    plan2, upd = partial_fit_online(
        session.dataset, plan, session.split_plan, n_rows=35
    )
    assert upd.n_updates == 1
    ev = evaluate_online(session.dataset, plan2, session.split_plan, partition="test")
    assert ev.metrics["accuracy"] >= 0.0
    out = save_online_bundle(tmp_path / "direct", plan2, fit_result=fit, eval_result=ev)
    assert (out / "online_plan.joblib").is_file()


def test_explicit_classes_and_new_class_refused() -> None:
    session = (
        Session.ingest(_clf_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_online(classes=[0, 1], chunk_size=30, n_init=30)
    plan = session.online_plan
    assert plan is not None
    bad = pd.DataFrame({c: [0.0, 0.1] for c in plan.columns})
    bad[plan.target_column] = [0, 99]
    with pytest.raises(ValidationError, match="new class label"):
        session.partial_fit_online(frame=bad)


def test_regression_sgd() -> None:
    session = (
        Session.ingest(_reg_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_online(estimator="sgd_regressor", chunk_size=40, n_init=40)
    assert fit.task == "regression"
    session.partial_fit_online(n_rows=40)
    ev = session.evaluate_online(partition="test")
    assert "mae" in ev.metrics
    assert "rmse" in ev.metrics


def test_external_frame_update() -> None:
    session = (
        Session.ingest(_clf_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_online(chunk_size=30, n_init=30, classes=[0, 1])
    cursor_before = session.online_plan.cursor  # type: ignore[union-attr]
    chunk = session.to_pandas().loc[list(session.split_plan.train_indices)[:10]]
    # Use a copy so we don't depend on cursor advancement.
    upd = session.partial_fit_online(frame=chunk[list(session.online_plan.columns) + ["y"]])  # type: ignore[union-attr]
    assert upd.n_chunk_rows == 10
    assert session.online_plan.cursor == cursor_before  # type: ignore[union-attr]


def test_explain_before() -> None:
    session = (
        Session.ingest(_clf_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    before = session.explain("fit_online", moment="before")
    assert before.prerequisite_status.get("split") is True
