"""Tier C: per-category NearestCentroid k-shot twin for coldstart-meta-adapt."""

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
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("coldstart-meta-adapt", seed=26)
    rng = np.random.default_rng(ctx.seed)
    rows = []
    for cat in range(18):
        center = rng.normal(size=5) * (0.6 + cat * 0.025)
        for _ in range(28):
            x = center + rng.normal(scale=0.35, size=5)
            y = int((x[0] + 0.35 * x[2] - 0.2 * x[4]) > 0)
            rows.append({
                **{f"emb{j}": float(x[j]) for j in range(5)},
                "repurchase": y,
                "category_id": f"cat{cat}",
            })
    frame = pd.DataFrame(rows)
    session = (
        Session.ingest(frame.copy())
        .set_roles({
            **{f"emb{j}": "feature" for j in range(5)},
            "repurchase": "target",
            "category_id": "group",
        })
        .group_split(
            test_size=0.25, validation_size=0.15,
            random_state=ctx.seed, group_column="category_id",
        )
    )
    plan = session.split_plan
    assert plan is not None
    test_idx = list(plan.test_indices)
    cols = [f"emb{j}" for j in range(5)]
    k_shot = 5

    accs, f1s = [], []
    test_cats = sorted(frame.loc[test_idx, "category_id"].unique())
    for cat in test_cats:
        block = frame.loc[frame["category_id"] == cat].copy()
        block = block.sample(frac=1.0, random_state=ctx.seed).reset_index(drop=True)
        support_idx = []
        for cls in (0, 1):
            cls_rows = block.index[block["repurchase"] == cls].tolist()
            support_idx.extend(cls_rows[:k_shot])
        if len(support_idx) < 2:
            continue
        query_idx = [i for i in block.index if i not in support_idx]
        if not query_idx:
            continue
        scaler = StandardScaler()
        x_s = scaler.fit_transform(block.loc[support_idx, cols])
        y_s = block.loc[support_idx, "repurchase"].to_numpy()
        x_q = scaler.transform(block.loc[query_idx, cols])
        y_q = block.loc[query_idx, "repurchase"].to_numpy()
        if len(np.unique(y_s)) < 2:
            continue
        clf = NearestCentroid()
        clf.fit(x_s, y_s)
        pred = clf.predict(x_q)
        accs.append(accuracy_score(y_q, pred))
        f1s.append(f1_score(y_q, pred, average="macro", zero_division=0))

    industry_metrics = metrics_round(
        {
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "mean_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
            "n_tasks_scored": float(len(accs)),
        }
    )
    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.metalearning/prototypical", "test_metrics": bml_metrics},
        industry={
            "backend": "per-category NearestCentroid k-shot",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same group_split by category_id (seed=26)",
                "Support drawn only within each held-out category",
            ],
        },
        split_counts={
            "train": len(plan.train_indices),
            "validation": len(plan.validation_indices),
            "test": len(plan.test_indices),
        },
        delta_keys=("mean_accuracy", "mean_f1_macro"),
    )
    print("coldstart-meta-adapt Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
