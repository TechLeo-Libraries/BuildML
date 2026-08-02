"""Tier C: sklearn DecisionTree twin for policy-rules-neuro-symbolic."""

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
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("policy-rules-neuro-symbolic", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 400
    age = rng.uniform(18, 80, size=n)
    income = rng.lognormal(10.5, 0.4, size=n)
    risk = rng.beta(2, 5, size=n)
    y = ((age < 25) & (risk > 0.45) | (income < 20000)).astype(int)
    frame = pd.DataFrame({"age": age, "income": income, "risk": risk, "deny": y})

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {"age": "feature", "income": "feature", "risk": "feature", "deny": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr, te = list(plan.train_indices), list(plan.test_indices)
    cols = ["age", "income", "risk"]

    clf = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=5, random_state=ctx.seed
    )
    clf.fit(frame.loc[tr, cols], frame.loc[tr, "deny"])
    pred = clf.predict(frame.loc[te, cols])
    industry_metrics = metrics_round(
        {"accuracy": float(accuracy_score(frame.loc[te, "deny"], pred))}
    )
    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.symbolic/decision_tree", "test_metrics": bml_metrics},
        industry={
            "backend": "sklearn.DecisionTreeClassifier(max_depth=4)",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same stratified split (seed=0)",
                "Tree fit on train only; test accuracy after lock",
            ],
        },
        split_counts={
            "train": len(tr),
            "validation": len(plan.validation_indices),
            "test": len(te),
        },
        delta_keys=("accuracy",),
    )
    print("policy-rules-neuro-symbolic Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
