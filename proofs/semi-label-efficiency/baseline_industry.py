"""Tier C: sklearn LabelPropagation twin for semi-label-efficiency."""

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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("semi-label-efficiency", seed=0)
    rng = np.random.default_rng(ctx.seed)
    x0 = rng.normal([-1.0, -1.0], 0.6, size=(180, 2))
    x1 = rng.normal([1.2, 1.0], 0.6, size=(180, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 180 + [1] * 180

    session = (
        Session.ingest(frame.copy())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    val_idx = list(plan.validation_indices)

    # Same masking policy as Tier A script (train only).
    y_all = frame["label"].to_numpy().astype(float)
    n_blank = max(1, int(0.75 * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    y_masked = y_all.copy()
    y_masked[blank] = -1  # sklearn semi-supervised convention

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, ["x", "y"]])
    x_test = scaler.transform(frame.loc[test_idx, ["x", "y"]])
    lp = LabelPropagation(kernel="knn", n_neighbors=7)
    lp.fit(x_train, y_masked[train_idx])
    pred = lp.predict(x_test)
    y_test = frame.loc[test_idx, "label"].to_numpy()
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "n_labeled_train": int(np.sum(y_masked[train_idx] >= 0)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw, prefer=("test_metrics",), keys=("accuracy", "f1", "f1_weighted", "macro_f1")
    )
    if "f1" not in bml_metrics:
        for alt in ("f1_weighted", "macro_f1"):
            if alt in bml_metrics:
                bml_metrics["f1"] = bml_metrics[alt]
                break

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session.fit_semisupervised",
            "method": bml_raw.get("fit", {}).get("method", "label_propagation"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.semi_supervised.LabelPropagation",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Masking applied to train indices only",
                "Holdouts keep full labels for evaluation only",
                "Scaler + LabelPropagation fit on train",
                "Same SplitPlan as BuildML Session",
            ],
            "validation_rows_reserved": len(val_idx),
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1"),
    )
    print("semi-label-efficiency Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
