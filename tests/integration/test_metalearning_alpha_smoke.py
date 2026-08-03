"""Integration smoke for meta-learning Session loop + walkthrough + bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_metalearning_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    rows: list[dict[str, object]] = []
    for task in range(10):
        shift = rng.normal(0, 1.0, size=2)
        for i in range(40):
            label = i % 2
            center = shift + (1.15 if label else -1.15)
            x = rng.normal(center, 0.4, size=2)
            rows.append(
                {
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "label": int(label),
                    "task_id": f"t{task}",
                }
            )
    frame = pd.DataFrame(rows)

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "task_id": "group",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_metalearning(
        method="prototypical",
        k_shot=3,
        n_query=6,
        n_episodes=15,
        task_holdout_fraction=0.3,
    )
    assert fit.n_meta_train_tasks >= 2

    # Episodic eval on train held-out tasks (true task-disjoint within train).
    ev_train = session.evaluate_metalearning(
        partition="train",
        k_shot=3,
        prefer_novel_tasks=True,
    )
    assert ev_train.n_tasks_evaluated >= 1
    assert "mean_accuracy" in ev_train.metrics

    adapt = session.adapt_to_task(
        task_id=session.metalearning_plan.train_task_ids[0],
        partition="train",
        max_support_per_class=3,
    )
    assert adapt.n_support >= 2

    walk = session.walkthrough()
    status = walk.metalearning_status
    assert status.get("has_metalearning_plan") is True
    assert status.get("method") == "prototypical"

    bundle = session.save_metalearning_bundle(tmp_path / "meta_bundle")
    other = Session.ingest(frame).set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "task_id": "group",
        }
    )
    other._split_plan = session.split_plan
    other._dataset = session.dataset
    other.load_metalearning_bundle(bundle, trusted=True)
    ev2 = other.evaluate_metalearning(partition="test", prefer_novel_tasks=False)
    assert ev2.method == "prototypical"
