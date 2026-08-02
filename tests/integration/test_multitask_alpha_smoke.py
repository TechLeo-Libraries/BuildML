"""Integration smoke for multi-task Session loop + walkthrough + bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_multitask_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 260
    x0 = rng.normal([-1.0, -1.0], 0.5, size=(n // 2, 2))
    x1 = rng.normal([1.1, 1.0], 0.5, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["t1"] = [0] * (n // 2) + [1] * (n - n // 2)
    frame["t2"] = ([0, 1] * (n // 2))[:n]

    session = (
        Session.ingest(frame)
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_multitask(
        method="multi_output",
        task="classification",
        base_estimator="logistic_regression",
    )
    assert fit.n_tasks == 2

    ev = session.evaluate_multitask(partition="validation")
    assert "mean_accuracy" in ev.metrics
    assert set(ev.per_task_metrics) == {"t1", "t2"}

    walk = session.walkthrough()
    status = walk.multitask_status
    assert status.get("has_multitask_plan") is True
    assert status.get("n_tasks") == 2

    bundle = session.save_multitask_bundle(tmp_path / "mt_bundle")
    other = (
        Session.ingest(frame)
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
    )
    other._split_plan = session.split_plan
    other._dataset = session.dataset
    other.load_multitask_bundle(bundle)
    ev2 = other.evaluate_multitask(partition="test")
    assert "mean_accuracy" in ev2.metrics
    assert ev2.n_rows > 0
