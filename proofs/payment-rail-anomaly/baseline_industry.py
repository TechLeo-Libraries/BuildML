"""Tier C: sklearn IsolationForest twin for payment-rail-anomaly."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_payment_rail_anomaly_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)

FEATURES = [
    "amount_z",
    "hour_sin",
    "hour_cos",
    "merchant_risk",
    "device_age_days",
    "velocity_1h",
]
LABEL = "is_attack"


def main() -> None:
    ctx = new_proof_context("payment-rail-anomaly", seed=108)
    frame, _ = load_payment_rail_anomaly_synthetic(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, LABEL: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, FEATURES])
    x_val = scaler.transform(frame.loc[val_idx, FEATURES])
    x_test = scaler.transform(frame.loc[test_idx, FEATURES])
    y_val = frame.loc[val_idx, LABEL].to_numpy()
    y_test = frame.loc[test_idx, LABEL].to_numpy()

    clf = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=ctx.seed,
        n_jobs=1,
    )
    clf.fit(x_train)
    val_scores = -clf.decision_function(x_val)
    test_scores = -clf.decision_function(x_test)

    qs = np.unique(np.quantile(val_scores, np.linspace(0.7, 0.99, 40)))
    best_thr, best_f1 = float(np.quantile(val_scores, 0.94)), -1.0
    for thr in qs:
        pred = (val_scores >= thr).astype(int)
        f1 = float(f1_score(y_val, pred, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    pred_test = (test_scores >= best_thr).astype(int)
    industry_metrics = metrics_round(
        {
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "average_precision": float(average_precision_score(y_test, test_scores)),
            "f1": float(f1_score(y_test, pred_test, zero_division=0)),
            "precision": float(precision_score(y_test, pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_labeled_metrics",),
        keys=("roc_auc", "average_precision", "f1", "precision", "recall"),
    )
    write_comparison(
        ctx,
        buildml={
            "backend": f"buildml.{bml_raw.get('backend', 'anomaly')}",
            "method": bml_raw.get("method", "unknown"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.ensemble.IsolationForest",
            "method": "isolation_forest",
            "test_metrics": industry_metrics,
            "threshold_source": "validation_f1_grid",
            "threshold": best_thr,
            "leakage_controls": [
                "StandardScaler fit on train only",
                "IsolationForest fit on train only",
                "Threshold tuned on validation labels only",
                "Test scored once after threshold lock",
                "Same SplitPlan indices as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("roc_auc", "average_precision", "f1", "precision", "recall"),
    )
    print("payment-rail-anomaly Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
