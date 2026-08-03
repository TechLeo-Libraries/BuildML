"""Unit coverage for the multi-task / multi-output thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.multitask.checkpoint import BUNDLE_FORMAT, load_multitask_bundle


def _frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(n // 2, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["t1"] = [0] * (n // 2) + [1] * (n - n // 2)
    frame["t2"] = ([0, 1] * (n // 2))[:n]
    return frame


def _ready_session() -> Session:
    return (
        Session.ingest(_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "t1": "target",
                "t2": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )


def test_core_import_and_catalog() -> None:
    import buildml.multitask as multitask

    assert hasattr(multitask, "fit_multitask")
    assert hasattr(Session, "fit_multitask")
    for name in (
        "fit_multitask",
        "predict_multitask",
        "evaluate_multitask",
        "save_multitask_bundle",
        "load_multitask_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "multitask-multi-output" in OPERATION_CATALOG["fit_multitask"].concept_links
    assert (
        "multitask-bundle-boundary"
        in OPERATION_CATALOG["save_multitask_bundle"].concept_links
    )


def test_fit_predict_evaluate_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_multitask(method="multi_output", task="classification")
    assert fit.n_tasks == 2
    assert session.multitask_plan is not None
    assert set(session.multitask_plan.target_columns) == {"t1", "t2"}

    preds = session.predict_multitask(partition="test")
    assert preds.n_rows == len(preds.predictions["t1"])
    assert set(preds.predictions) == {"t1", "t2"}

    ev = session.evaluate_multitask(partition="validation")
    assert "mean_accuracy" in ev.metrics
    assert "t1" in ev.per_task_metrics and "t2" in ev.per_task_metrics
    assert session.multitask_eval_result is not None

    before = session.explain("evaluate_multitask", moment="before")
    assert before.prerequisite_status.get("multitask-plan") is True

    bundle = session.save_multitask_bundle(tmp_path / "multitask_bundle")
    assert (bundle / "meta.json").is_file()
    plan = load_multitask_bundle(bundle, trusted=True)
    assert plan.n_train_rows == fit.n_train_rows

    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_multitask_bundle(bundle, trusted=True)
    assert restored.multitask_plan is not None
    assert restored.multitask_plan.method == "multi_output"


def test_refuse_without_split() -> None:
    session = Session.ingest(_frame()).set_roles(
        {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
    )
    with pytest.raises(LeakageError, match="split"):
        session.fit_multitask()


def test_refuse_single_target() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x": "feature", "y": "feature", "t1": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="at least 2 target"):
        session.fit_multitask()


def test_classical_fit_still_requires_one_target() -> None:
    from sklearn.linear_model import LogisticRegression

    session = _ready_session()
    with pytest.raises(ValidationError, match="exactly one target"):
        session.fit(LogisticRegression(max_iter=200))


def test_ai_allowlist() -> None:
    registry = build_default_registry()
    names = {t.name for t in registry.tools}
    for name in (
        "fit_multitask",
        "evaluate_multitask",
        "save_multitask_bundle",
        "load_multitask_bundle",
    ):
        assert name in names


def test_bundle_format_constant() -> None:
    assert BUNDLE_FORMAT == "buildml.multitask_bundle.v1"
