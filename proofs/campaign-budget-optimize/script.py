"""Tier A proof: campaign-budget-optimize — marketing scoring + decision policy."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session
from proofs._lib import metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("campaign-budget-optimize", seed=29)
    x, y = make_classification(
        n_samples=620, n_features=10, n_informative=6,
        weights=[0.72, 0.28], random_state=ctx.seed,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["responded"] = y
    frame["cost"] = np.where(y == 1, 2.5, 1.0)
    frame["id"] = [f"lead-{i}" for i in range(len(frame))]
    session = (
        Session.ingest(frame)
        .set_roles({
            **{c: "feature" for c in frame.columns if c.startswith("f")},
            "responded": "target",
        })
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .fit(LogisticRegression(max_iter=800), task="classification")
    )
    thr = session.decision.fit(
        method="threshold", partition="validation", fp_cost=1.0, fn_cost=3.5,
    )
    eval_thr = session.decision.evaluate(partition="test")
    knap = session.decision.fit(
        method="knapsack", partition="validation", budget=55.0,
        cost_column="cost", id_column="id", score_source="model_proba", knapsack_solver="dp",
    )
    applied = session.decision.apply(partition="test")
    try:
        bundle = session.decision.save_bundle(ctx.artifacts_dir / "decision_bundle")
        bundle_path = str(bundle)
    except Exception as exc:  # noqa: BLE001
        bundle_path = f"unavailable: {exc}"
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_campaign_leads",
            "license": "synthetic/public-domain",
            "n_rows": int(len(frame)),
        },
        "threshold_policy": metrics_round(thr.to_dict() if hasattr(thr, "to_dict") else {}),
        "threshold_test": metrics_round(eval_thr.to_dict() if hasattr(eval_thr, "to_dict") else {}),
        "knapsack_policy": metrics_round(knap.to_dict() if hasattr(knap, "to_dict") else {}),
        "knapsack_applied": {
            "n_selected": int(applied.n_selected),
            "selected_value": float(applied.selected_value),
            "selected_cost": float(applied.selected_cost),
        },
        "bundle_path": bundle_path,
        "leakage_controls": [
            "Policies fit/selected on validation only",
            "Test session.decision.evaluate / session.decision.apply after lock",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: val cost-threshold twin; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Not general OR; disclosed knapsack/threshold helpers only",
            "Marketing campaign narrative distinct from collections",
        ],
    })
    print(
        "campaign-budget-optimize OK",
        eval_thr.to_dict() if hasattr(eval_thr, "to_dict") else eval_thr,
    )


if __name__ == "__main__":
    main()
