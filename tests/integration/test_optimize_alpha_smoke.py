"""Integration smoke: decision policy fit → evaluate → bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session


def test_decision_end_to_end_smoke(tmp_path) -> None:
    x, y = make_classification(
        n_samples=360,
        n_features=8,
        n_informative=5,
        weights=[0.7, 0.3],
        random_state=11,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    frame["cost"] = np.linspace(1.0, 2.0, len(frame))

    session = (
        Session.ingest(frame)
        .set_roles(
            {**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=11)
        .fit(LogisticRegression(max_iter=500), task="classification")
    )

    session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=3.0,
    )
    eval_thr = session.evaluate_decisions(partition="test")
    assert eval_thr.realized_cost is not None

    session.fit_decision_policy(
        method="knapsack",
        partition="validation",
        budget=30.0,
        cost_column="cost",
        knapsack_solver="dp",
    )
    applied = session.apply_decisions(partition="test")
    assert applied.n_selected >= 1
    assert float(applied.selected_cost or 0.0) <= 30.0 + 1e-5

    bundle = tmp_path / "decision_bundle"
    session.save_decision_bundle(bundle)
    assert (bundle / "meta.json").is_file()
    assert (bundle / "decision_plan.joblib").is_file()

    walk = session.walkthrough()
    assert walk.decision_status.get("present") is True
