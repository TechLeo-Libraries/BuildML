"""Multi-task example: fit_multitask → evaluate → bundle.

Honesty: sklearn MultiOutput on shared features with multiple same-type
targets — not a deep multi-head MTL research platform. Classical Session.fit
remains single-target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(9)
    n = 280
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(n // 2, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(n - n // 2, 2))
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
    print(
        f"method={fit.method} task={fit.task} n_tasks={fit.n_tasks} "
        f"targets={list(fit.target_columns)}"
    )

    ev = session.evaluate_multitask(partition="validation")
    print(f"aggregate metrics={ev.metrics}")
    print(f"per-task metrics={ev.per_task_metrics}")

    out = Path("artifacts") / "multitask_multioutput_bundle"
    session.save_multitask_bundle(out)
    print(f"saved bundle → {out}")


if __name__ == "__main__":
    main()
