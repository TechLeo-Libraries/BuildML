"""Tier C: sklearn logistic baseline twin for credit-tda-shape (no TDA)."""

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
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("credit-tda-shape", seed=0)
    rng = np.random.default_rng(ctx.seed)
    a = rng.normal(size=(160, 4))
    b = rng.normal(size=(160, 4)) * 1.6 + np.array([2.5, 0.0, 0.0, 0.0])
    frame = pd.DataFrame(np.vstack([a, b]), columns=[f"f{i}" for i in range(4)])
    frame["approved"] = [0] * 160 + [1] * 160

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "approved": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed, stratify=True)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    val_idx = list(plan.validation_indices)
    feats = [f"f{i}" for i in range(4)]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, feats])
    y_train = frame.loc[train_idx, "approved"].to_numpy()
    x_test = scaler.transform(frame.loc[test_idx, feats])
    y_test = frame.loc[test_idx, "approved"].to_numpy()
    clf = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw, prefer=("test_metrics",), keys=("accuracy", "macro_f1", "f1")
    )
    if "macro_f1" not in bml_metrics and "f1" in bml_metrics:
        bml_metrics["macro_f1"] = bml_metrics["f1"]

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.tda.fit",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.LogisticRegression on raw scaled features",
            "test_metrics": industry_metrics,
            "note": (
                "Industry twin is a classical tabular baseline (not TDA). "
                "Competitiveness means TDA head is in the same ballpark, not that "
                "persistence images dominate linear features on this synthetic draw."
            ),
            "leakage_controls": [
                "Scaler + logistic fit on train only",
                "Same stratified SplitPlan as BuildML TDA path",
                "Test evaluated once after lock",
            ],
            "validation_rows_reserved": len(val_idx),
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "macro_f1"),
    )
    print("credit-tda-shape Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
