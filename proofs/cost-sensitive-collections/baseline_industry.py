"""Tier C: sklearn cost-sensitive threshold twin for cost-sensitive-collections."""

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
from sklearn.metrics import f1_score, precision_score, recall_score

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _expected_cost(y_true, y_pred, fp_cost=1.0, fn_cost=4.0) -> float:
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    return fp * fp_cost + fn * fn_cost


def main() -> None:
    ctx = new_proof_context("cost-sensitive-collections", seed=7)
    x, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        weights=[0.75, 0.25],
        random_state=ctx.seed,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["delinquent"] = y
    frame["cost"] = np.where(y == 1, 3.0, 1.0)
    frame["id"] = [f"acct-{i}" for i in range(len(frame))]

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in frame.columns if c.startswith("f")},
                "delinquent": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr = list(plan.train_indices)
    va = list(plan.validation_indices)
    te = list(plan.test_indices)
    cols = [c for c in frame.columns if c.startswith("f")]

    clf = LogisticRegression(max_iter=800, random_state=ctx.seed)
    clf.fit(frame.loc[tr, cols], frame.loc[tr, "delinquent"])
    val_proba = clf.predict_proba(frame.loc[va, cols])[:, 1]
    y_va = frame.loc[va, "delinquent"].to_numpy()

    # Threshold selected on VALIDATION only (fp=1, fn=4) — never test.
    best_thr, best_cost = 0.5, float("inf")
    for thr in np.linspace(0.05, 0.95, 37):
        pred = (val_proba >= thr).astype(int)
        cost = _expected_cost(y_va, pred, fp_cost=1.0, fn_cost=4.0)
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)

    test_proba = clf.predict_proba(frame.loc[te, cols])[:, 1]
    y_te = frame.loc[te, "delinquent"].to_numpy()
    pred_te = (test_proba >= best_thr).astype(int)
    industry_metrics = metrics_round(
        {
            "threshold": best_thr,
            "expected_cost_total": float(
                _expected_cost(y_te, pred_te, fp_cost=1.0, fn_cost=4.0)
            ),
            "f1": float(f1_score(y_te, pred_te, zero_division=0)),
            "precision": float(precision_score(y_te, pred_te, zero_division=0)),
            "recall": float(recall_score(y_te, pred_te, zero_division=0)),
        }
    )

    bml = load_buildml_results(ctx.project_dir)
    thr_test = dict(bml.get("threshold_test", {}).get("metrics", {}))
    bml_metrics = metrics_round(
        {
            "threshold": float(bml.get("threshold_policy", {}).get("threshold") or thr_test.get("threshold") or 0.0),
            "expected_cost_total": float(thr_test.get("expected_cost_total", float("nan"))),
            "f1": float(thr_test.get("f1", float("nan"))),
            "precision": float(thr_test.get("precision", float("nan"))),
            "recall": float(thr_test.get("recall", float("nan"))),
        }
    )

    write_comparison(
        ctx,
        buildml={"backend": "buildml.decisions/threshold", "test_metrics": bml_metrics},
        industry={
            "backend": "sklearn.LogisticRegression + val cost-threshold sweep",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same stratified split (seed=7)",
                "Threshold selected on validation only (fp=1, fn=4)",
                "Test used once after threshold lock",
            ],
        },
        split_counts={"train": len(tr), "validation": len(va), "test": len(te)},
        delta_keys=("expected_cost_total", "f1", "precision", "recall"),
    )
    print("cost-sensitive-collections Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
