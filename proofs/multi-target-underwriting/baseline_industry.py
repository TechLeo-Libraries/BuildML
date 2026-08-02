"""Tier C: sklearn MultiOutputClassifier twin for multi-target-underwriting."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("multi-target-underwriting", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    x = rng.normal(size=(n, 6))
    t1 = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    t2 = (x[:, 2] - 0.4 * x[:, 3] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["approve"] = t1
    frame["high_limit"] = t2

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{f"f{i}": "feature" for i in range(6)},
                "approve": "target",
                "high_limit": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr, te = list(plan.train_indices), list(plan.test_indices)
    cols = [f"f{i}" for i in range(6)]
    targets = ["approve", "high_limit"]

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(frame.loc[tr, cols])
    x_te = scaler.transform(frame.loc[te, cols])
    y_tr = frame.loc[tr, targets].to_numpy()
    y_te = frame.loc[te, targets].to_numpy()

    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=ctx.seed))
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    accs = [accuracy_score(y_te[:, i], pred[:, i]) for i in range(2)]
    f1s = [
        f1_score(y_te[:, i], pred[:, i], average="macro", zero_division=0)
        for i in range(2)
    ]
    f1w = [
        f1_score(y_te[:, i], pred[:, i], average="weighted", zero_division=0)
        for i in range(2)
    ]
    industry_metrics = metrics_round(
        {
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "mean_f1_weighted": float(np.mean(f1w)),
        }
    )
    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.multitask/multioutput", "test_metrics": bml_metrics},
        industry={
            "backend": "sklearn.MultiOutputClassifier(LogisticRegression)",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same split indices as BuildML (seed=0)",
                "Scaler + MultiOutput fit on train only",
            ],
        },
        split_counts={
            "train": len(tr),
            "validation": len(plan.validation_indices),
            "test": len(te),
        },
        delta_keys=("mean_accuracy", "mean_f1_macro", "mean_f1_weighted"),
    )
    print("multi-target-underwriting Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
