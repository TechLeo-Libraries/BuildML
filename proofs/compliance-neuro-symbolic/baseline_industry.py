"""Tier C: sklearn DecisionTree twin for compliance-neuro-symbolic."""

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
    ctx = new_proof_context("compliance-neuro-symbolic", seed=27)
    rng = np.random.default_rng(ctx.seed)
    n = 420
    account_age_days = rng.uniform(1, 4000, size=n)
    wire_amount = rng.lognormal(9.2, 0.85, size=n)
    pep_score = rng.beta(2.5, 4.0, size=n)
    jurisdiction_risk = rng.beta(2.5, 3.5, size=n)
    escalate = (
        ((wire_amount > 8000) & (jurisdiction_risk > 0.35))
        | ((account_age_days < 365) & (pep_score > 0.30))
    ).astype(int)
    frame = pd.DataFrame({
        "account_age_days": account_age_days,
        "wire_amount": wire_amount,
        "pep_score": pep_score,
        "jurisdiction_risk": jurisdiction_risk,
        "escalate": escalate,
    })
    cols = ["account_age_days", "wire_amount", "pep_score", "jurisdiction_risk"]

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in cols}, "escalate": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr, te = list(plan.train_indices), list(plan.test_indices)

    clf = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=5, random_state=ctx.seed,
    )
    clf.fit(frame.loc[tr, cols], frame.loc[tr, "escalate"])
    pred = clf.predict(frame.loc[te, cols])
    industry_metrics = metrics_round(
        {"accuracy": float(accuracy_score(frame.loc[te, "escalate"], pred))}
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
                "Same stratified split (seed=27)",
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
    print("compliance-neuro-symbolic Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
