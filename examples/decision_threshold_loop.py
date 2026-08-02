"""Decision helpers: cost-sensitive threshold + knapsack allocation loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session


def main() -> None:
    x, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        weights=[0.75, 0.25],
        random_state=7,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    frame["cost"] = np.where(y == 1, 3.0, 1.0)
    frame["id"] = [f"row-{i}" for i in range(len(frame))]

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                **{c: "feature" for c in frame.columns if c.startswith("f")},
                "y": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=7)
        .fit(LogisticRegression(max_iter=800), task="classification")
    )

    # 1) Cost-sensitive operating point on validation (persisted DecisionPlan)
    thr = session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=4.0,
    )
    print("threshold fit:", thr.to_dict())

    # Classical diagnostic still works (same engine; does not replace the plan)
    diagnostic = session.tune_threshold(
        partition="validation", fp_cost=1.0, fn_cost=4.0
    )
    print(
        "tune_threshold recommended:",
        diagnostic.payload.get("recommended_threshold", {}).get("threshold"),
    )

    eval_thr = session.evaluate_decisions(partition="test")
    print("threshold eval:", eval_thr.to_dict())

    # 2) Budget-constrained selection using model scores + row costs
    knap = session.fit_decision_policy(
        method="knapsack",
        partition="validation",
        budget=50.0,
        cost_column="cost",
        id_column="id",
        score_source="model_proba",
        knapsack_solver="dp",
    )
    print("knapsack fit:", knap.to_dict())
    applied = session.apply_decisions(partition="test")
    print(
        f"selected={applied.n_selected} value={applied.selected_value} "
        f"cost={applied.selected_cost}"
    )
    print("first ids:", applied.selected_ids[:8])

    # 3) Refuse silent test tuning
    try:
        session.fit_decision_policy(
            method="threshold",
            partition="test",
            fp_cost=1.0,
            fn_cost=4.0,
        )
    except Exception as exc:  # LeakageError
        print("blocked test tuning:", type(exc).__name__, str(exc)[:120])

    out = Path("artifacts/decision_demo_bundle")
    session.save_decision_bundle(out)
    print("bundle:", out.resolve())


if __name__ == "__main__":
    main()
