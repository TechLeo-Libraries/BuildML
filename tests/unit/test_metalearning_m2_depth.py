"""M2 depth coverage for meta-learning low-level APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.metalearning.adapt import adapt_to_task
from buildml.metalearning.evaluate import evaluate_metalearning
from buildml.metalearning.fit import fit_metalearning


def _frame(n_tasks: int = 6, n_per_task: int = 36, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for task in range(n_tasks):
        shift = rng.normal(0, 0.8, size=2)
        for i in range(n_per_task):
            label = i % 2
            center = shift + (1.0 if label else -1.0)
            x = rng.normal(center, 0.4, size=2)
            rows.append(
                {
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "label": int(label),
                    "task_id": task,
                }
            )
    return pd.DataFrame(rows)


def _session() -> Session:
    return (
        Session.ingest(_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "task_id": "group",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard")
    )


def test_low_level_warm_start() -> None:
    session = _session()
    plan, fit = fit_metalearning(
        session.dataset,
        session.split_plan,
        method="warm_start",
        base_estimator="logistic_regression",
        k_shot=3,
        n_query=5,
        n_episodes=8,
        reduce_plan=session._reduce_plan,
    )
    assert fit.method == "warm_start"
    assert plan.init_estimator_ is not None
    adapt = adapt_to_task(
        session.dataset,
        plan,
        session.split_plan,
        task_id=plan.train_task_ids[0],
        partition="train",
        max_support_per_class=3,
    )
    assert adapt.adapted_estimator_ is not None
    ev = evaluate_metalearning(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
        k_shot=3,
        prefer_novel_tasks=False,
    )
    assert ev.method == "warm_start"


def test_explicit_task_column() -> None:
    frame = _frame()
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "task_id": "feature",  # not group — pass task_column=
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_metalearning(
        session.dataset,
        session.split_plan,
        method="prototypical",
        task_column="task_id",
        k_shot=2,
        n_episodes=6,
        reduce_plan=session._reduce_plan,
    )
    assert plan.task_column == "task_id"
    assert fit.n_meta_train_tasks >= 2
    # task_id must not be in feature columns
    assert "task_id" not in plan.columns


def test_refuse_unknown_method() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="Unknown meta-learning method"):
        fit_metalearning(
            session.dataset,
            session.split_plan,
            method="maml",  # type: ignore[arg-type]
        )


def test_explain_before_prereq() -> None:
    session = _session()
    before = session.explain("adapt_to_task", moment="before")
    assert before.prerequisite_status.get("metalearning-plan") is False
    session.fit_metalearning(method="prototypical", k_shot=2, n_episodes=5)
    after = session.explain("adapt_to_task", moment="before")
    assert after.prerequisite_status.get("metalearning-plan") is True
