"""Tier C: sklearn SGDClassifier partial_fit twin for clickstream-online."""

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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


FEATS = ["pages_z", "dwell_z", "cart_adds_z"]


def main() -> None:
    ctx = new_proof_context("clickstream-online", seed=24)
    rng = np.random.default_rng(ctx.seed)
    bounce = rng.normal([-1.0, -0.8, 0.2], 0.55, size=(230, 3))
    convert = rng.normal([1.1, 0.9, -0.2], 0.55, size=(230, 3))
    frame = pd.DataFrame(np.vstack([bounce, convert]), columns=FEATS)
    frame["converted"] = [0] * 230 + [1] * 230

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATS}, "converted": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, FEATS])
    y_train = frame.loc[train_idx, "converted"].to_numpy()
    x_test = scaler.transform(frame.loc[test_idx, FEATS])
    y_test = frame.loc[test_idx, "converted"].to_numpy()

    clf = SGDClassifier(loss="log_loss", random_state=ctx.seed)
    chunk = 50
    n_init = 50
    clf.partial_fit(x_train[:n_init], y_train[:n_init], classes=np.array([0, 1]))
    cursor = n_init
    updates = 0
    while cursor < len(train_idx):
        end = min(cursor + chunk, len(train_idx))
        clf.partial_fit(x_train[cursor:end], y_train[cursor:end])
        cursor = end
        updates += 1

    proba = clf.predict_proba(x_test)[:, 1]
    pred = clf.predict(x_test)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "n_updates": updates,
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("accuracy", "f1", "roc_auc", "f1_weighted", "macro_f1"),
    )
    if "f1" not in bml_metrics:
        for alt in ("f1_weighted", "macro_f1"):
            if alt in bml_metrics:
                bml_metrics["f1"] = bml_metrics[alt]
                break

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.online.fit",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.SGDClassifier.partial_fit",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "partial_fit consumes train cursor only",
                "Validation/test never enter online updates",
                "Scaler fit on train only",
                "Same SplitPlan as BuildML Session",
            ],
            "validation_rows_reserved": len(val_idx),
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
    )
    print("clickstream-online Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
