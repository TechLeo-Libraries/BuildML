"""Tier C: sklearn KNeighborsClassifier twin for warranty-cbr-memory."""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("warranty-cbr-memory", seed=28)
    rng = np.random.default_rng(ctx.seed)
    n = 320
    x = rng.normal(size=(n, 5))
    y = (x[:, 0] + 0.65 * x[:, 2] - 0.3 * x[:, 4] + rng.normal(scale=0.25, size=n) > 0).astype(int)
    cols = [
        "failure_severity", "usage_hours_z", "prior_claims_z", "parts_cost_z", "age_months_z",
    ]
    frame = pd.DataFrame(x, columns=cols)
    frame["approve_claim"] = y
    frame["claim_id"] = [f"w{i}" for i in range(n)]

    session = (
        Session.ingest(frame.copy())
        .set_roles({
            **{c: "feature" for c in cols},
            "approve_claim": "target",
            "claim_id": "id",
        })
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr, te = list(plan.train_indices), list(plan.test_indices)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(frame.loc[tr, cols])
    x_te = scaler.transform(frame.loc[te, cols])
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    clf.fit(x_tr, frame.loc[tr, "approve_claim"])
    pred = clf.predict(x_te)
    industry_metrics = metrics_round(
        {"accuracy": float(accuracy_score(frame.loc[te, "approve_claim"], pred))}
    )
    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.cbr/distance_weighted", "test_metrics": bml_metrics},
        industry={
            "backend": "sklearn.KNeighborsClassifier(k=5, weights=distance)",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same stratified split (seed=28)",
                "Scaler + kNN fit on train case memory only",
            ],
        },
        split_counts={
            "train": len(tr),
            "validation": len(plan.validation_indices),
            "test": len(te),
        },
        delta_keys=("accuracy",),
    )
    print("warranty-cbr-memory Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
