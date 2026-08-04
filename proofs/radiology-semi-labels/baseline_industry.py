"""Tier C: sklearn LabelPropagation twin for radiology-semi-labels."""

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


FEATS = ["hu_mean", "texture_entropy", "edge_density", "asymmetry"]


def main() -> None:
    ctx = new_proof_context("radiology-semi-labels", seed=22)
    rng = np.random.default_rng(ctx.seed)
    neg = rng.normal([-0.9, -0.8, -0.5, 0.2], [0.55, 0.5, 0.45, 0.4], size=(190, 4))
    pos = rng.normal([1.1, 0.9, 0.7, -0.3], [0.55, 0.5, 0.45, 0.4], size=(190, 4))
    frame = pd.DataFrame(np.vstack([neg, pos]), columns=FEATS)
    frame["lesion_present"] = [0] * 190 + [1] * 190

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATS}, "lesion_present": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    val_idx = list(plan.validation_indices)

    y_all = frame["lesion_present"].to_numpy().astype(float)
    n_blank = max(1, int(0.78 * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    y_masked = y_all.copy()
    y_masked[blank] = -1

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, FEATS])
    x_test = scaler.transform(frame.loc[test_idx, FEATS])
    lp = LabelPropagation(kernel="knn", n_neighbors=7)
    lp.fit(x_train, y_masked[train_idx])
    pred = lp.predict(x_test)
    y_test = frame.loc[test_idx, "lesion_present"].to_numpy()
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
            "backend": "buildml.session.semisupervised.fit",
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
    print("radiology-semi-labels Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
