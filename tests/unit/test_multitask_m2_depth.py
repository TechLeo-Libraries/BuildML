"""M2 depth coverage for multi-task low-level APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.multitask.evaluate import evaluate_multitask
from buildml.multitask.fit import fit_multitask


def _cls_frame(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    frame = pd.DataFrame(x, columns=["x", "y"])
    frame["t1"] = (x[:, 0] > 0).astype(int)
    frame["t2"] = (x[:, 1] > 0).astype(int)
    return frame


def _reg_frame(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    frame = pd.DataFrame(x, columns=["x", "y"])
    frame["t1"] = x[:, 0] * 1.5 + rng.normal(0, 0.1, size=n)
    frame["t2"] = x[:, 1] * -0.8 + rng.normal(0, 0.1, size=n)
    return frame


def test_low_level_classifier_chain() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_multitask(
        session.dataset,
        session.split_plan,
        method="classifier_chain",
        task="classification",
        order=["t2", "t1"],
        reduce_plan=session._reduce_plan,
    )
    assert fit.method == "classifier_chain"
    assert plan.target_columns == ("t1", "t2")
    ev = evaluate_multitask(
        session.dataset, plan, session.split_plan, partition="validation"
    )
    assert set(ev.per_task_metrics) == {"t1", "t2"}
    assert "mean_f1_macro" in ev.metrics


def test_low_level_regression_multi_output() -> None:
    session = (
        Session.ingest(_reg_frame())
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_multitask(
        session.dataset,
        session.split_plan,
        method="multi_output",
        task="regression",
        base_estimator="ridge",
        reduce_plan=session._reduce_plan,
    )
    assert fit.task == "regression"
    assert plan.classes_per_task_ == {}
    ev = evaluate_multitask(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert "mean_mae" in ev.metrics
    assert "r2" in ev.per_task_metrics["t1"]


def test_explicit_targets_override() -> None:
    frame = _cls_frame()
    frame["t3"] = 1 - frame["t1"]
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "t1": "target",
                "t2": "target",
                "t3": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_multitask(targets=["t1", "t3"], task="classification")
    assert list(fit.target_columns) == ["t1", "t3"]


def test_refuse_mixed_task_kinds() -> None:
    frame = _cls_frame()
    rng = np.random.default_rng(0)
    frame["t_reg"] = frame["x"] * 2.0 + rng.normal(0, 0.05, size=len(frame))
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "t1": "target",
                "t_reg": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="Mixed classification/regression"):
        session.fit_multitask(task="auto")


def test_explain_prereq_before_fit() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    before = session.explain("fit_multitask", moment="before")
    assert before.prerequisite_status.get("split") is True
    after_fit = session.fit_multitask()
    assert after_fit.n_tasks == 2
    before_eval = session.explain("evaluate_multitask", moment="before")
    assert before_eval.prerequisite_status.get("multitask-plan") is True
