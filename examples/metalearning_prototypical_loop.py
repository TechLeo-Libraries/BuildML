"""Meta-learning example: fit_metalearning → adapt → evaluate → bundle.

Honesty: tabular few-shot / episodic Session protocol (prototypical
nearest-centroid) — not foundation-model meta-learning or MAML-at-scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(21)
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
        n_episodes=20,
        task_holdout_fraction=0.3,
    )
    print(
        f"method={fit.method} n_meta_train_tasks={fit.n_meta_train_tasks} "
        f"meta_train_accuracy={fit.meta_train_accuracy}"
    )

    adapt = session.adapt_to_task(
        task_id=session.metalearning_plan.train_task_ids[0],
        partition="train",
        max_support_per_class=3,
    )
    print(
        f"adapt task_id={adapt.task_id} n_support={adapt.n_support} "
        f"n_classes={adapt.n_classes_adapted}"
    )

    ev = session.evaluate_metalearning(partition="train", k_shot=3)
    print(f"episodic metrics={ev.metrics}")
    print(f"per-task={ev.per_task_metrics}")

    out = Path("artifacts") / "metalearning_prototypical_bundle"
    session.save_metalearning_bundle(out)
    print(f"saved bundle -> {out}")


if __name__ == "__main__":
    main()
