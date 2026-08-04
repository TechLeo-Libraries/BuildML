"""Tier C: pointwise Ridge LTR twin for search-relevance-ltr."""

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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _judgments(n_queries=60, n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        q_center = float(q % 6)
        for item in range(n_items):
            f1 = float(rng.normal(q_center, 0.8))
            f2 = float(rng.normal(item / 2.0, 0.5))
            bm25 = float(rng.random())
            score = 3.0 - abs(f1 - q_center) + 0.4 * (item % 3 == 0) + 0.3 * bm25
            rel = float(max(0, min(4, int(round(score)))))
            rows.append(
                {
                    "query_id": f"q{q}",
                    "item_id": f"d{item}",
                    "f1": f1,
                    "f2": f2,
                    "bm25": bm25,
                    "relevance": rel,
                }
            )
    return pd.DataFrame(rows)


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def main() -> None:
    ctx = new_proof_context("search-relevance-ltr", seed=0)
    frame = _judgments(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "query_id": "group",
                "item_id": "id",
                "relevance": "target",
                "f1": "feature",
                "f2": "feature",
                "bm25": "feature",
            }
        )
        .group_split(test_size=0.25, validation_size=0.15, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    feats = ["f1", "f2", "bm25"]
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, feats])
    y_train = frame.loc[train_idx, "relevance"].to_numpy()
    x_test = scaler.transform(frame.loc[test_idx, feats])
    model = Ridge(alpha=1.0, random_state=ctx.seed)
    model.fit(x_train, y_train)
    scores = model.predict(x_test)
    test = frame.loc[test_idx].copy()
    test["score"] = scores
    ndcgs = []
    for _, grp in test.groupby("query_id"):
        ndcgs.append(
            _ndcg_at_k(grp["relevance"].to_numpy(), grp["score"].to_numpy(), k=5)
        )
    industry_metrics = metrics_round({"ndcg_at_k": float(np.mean(ndcgs)), "n_queries": int(len(ndcgs))})

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    for src, dst in (("ndcg", "ndcg_at_k"), ("ndcg@5", "ndcg_at_k"), ("ndcg_at_5", "ndcg_at_k")):
        if src in bml_metrics and dst not in bml_metrics:
            bml_metrics[dst] = bml_metrics[src]
    bml_metrics = metrics_round(bml_metrics)

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.ranking.fit",
            "method": bml_raw.get("method", "pointwise"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.Ridge pointwise LTR",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "group_split on query_id (no query leakage across partitions)",
                "Scaler + Ridge fit on train queries only",
                "Test nDCG after lock",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(plan.validation_indices),
            "test": len(test_idx),
        },
        delta_keys=("ndcg_at_k",),
    )
    print("search-relevance-ltr Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
