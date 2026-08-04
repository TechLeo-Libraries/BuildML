"""Tier C: PCA embedding + logistic probe twin for tabular-ssl-probe."""

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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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


def main() -> None:
    ctx = new_proof_context("tabular-ssl-probe", seed=32)
    rng = np.random.default_rng(ctx.seed)
    n = 420
    x = rng.normal(size=(n, 10))
    y = (
        0.8 * x[:, 0] - 0.4 * x[:, 3] + 0.35 * x[:, 7]
        + rng.normal(scale=0.28, size=n) > 0
    ).astype(int)
    frame = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(10)])
    frame["fault"] = y
    feats = [f"sensor_{i}" for i in range(10)]

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in feats}, "fault": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, feats])
    pca = PCA(n_components=5, random_state=ctx.seed)
    z_train = pca.fit_transform(x_train)
    z_test = pca.transform(scaler.transform(frame.loc[test_idx, feats]))
    y_train = frame.loc[train_idx, "fault"].to_numpy()
    y_test = frame.loc[test_idx, "fault"].to_numpy()

    probe = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    probe.fit(z_train, y_train)
    proba = probe.predict_proba(z_test)[:, 1]
    pred = probe.predict(z_test)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
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
            "backend": "buildml.session.ssl.fit_pretext",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn PCA pretext + LogisticRegression probe",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "PCA (label-free) fit on train only",
                "Probe fit on train embeddings + labels",
                "Validation reserved; test after lock",
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
    print("tabular-ssl-probe Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
