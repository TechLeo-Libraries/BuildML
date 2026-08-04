"""Tier C: holdout-blend sklearn twin for blending-payment-risk."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
TARGET = "is_attack"


def main() -> None:
    ctx = new_proof_context("blending-payment-risk", seed=105)
    frame, _ = load_payment_rail_anomaly_synthetic(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    x_train_full = frame.loc[train_idx, FEATURES].to_numpy(dtype=float)
    y_train_full = frame.loc[train_idx, TARGET].to_numpy()
    x_test = frame.loc[test_idx, FEATURES].to_numpy(dtype=float)
    y_test = frame.loc[test_idx, TARGET].to_numpy()

    scaler = StandardScaler()
    x_train_full = scaler.fit_transform(x_train_full)
    x_test = scaler.transform(x_test)

    # Holdout blend carved from train only (mirrors BuildML session.ensemble.fit_blending).
    x_base, x_blend, y_base, y_blend = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=ctx.seed,
        stratify=y_train_full,
    )
    lr = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    rf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=ctx.seed)
    lr.fit(x_base, y_base)
    rf.fit(x_base, y_base)
    blend_feats = np.column_stack(
        [lr.predict_proba(x_blend)[:, 1], rf.predict_proba(x_blend)[:, 1]]
    )
    meta = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    meta.fit(blend_feats, y_blend)

    # Refit bases on full train (parity with BuildML refit_bases_on_full_train=True).
    lr.fit(x_train_full, y_train_full)
    rf.fit(x_train_full, y_train_full)
    test_feats = np.column_stack(
        [lr.predict_proba(x_test)[:, 1], rf.predict_proba(x_test)[:, 1]]
    )
    proba = meta.predict_proba(test_feats)[:, 1]
    pred = (proba >= 0.5).astype(int)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("accuracy", "f1", "roc_auc", "f1_weighted"),
    )
    if "f1" not in bml_metrics and "f1_weighted" in bml_metrics:
        bml_metrics["f1"] = bml_metrics["f1_weighted"]

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.ensemble.fit_blending",
            "holdout_fraction": 0.2,
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn holdout blend (LR+RF → logistic meta)",
            "holdout_fraction": 0.2,
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Blend holdout carved from train only",
                "Bases refit on full train after meta fit",
                "Test evaluated once after lock",
                "Same SplitPlan as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
    )
    print("blending-payment-risk Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
