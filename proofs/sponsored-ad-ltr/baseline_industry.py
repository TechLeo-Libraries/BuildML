"""Tier C: pointwise Ridge LTR twin for sponsored-ad-ltr."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_ad_ltr_judgments_synthetic,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def main() -> None:
    ctx = new_proof_context("sponsored-ad-ltr", seed=114)
    frame, _ = load_ad_ltr_judgments_synthetic(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "query_id": "group",
                "ad_id": "id",
                "relevance": "target",
                "rel_feat": "feature",
                "bid": "feature",
                "ctr_prior": "feature",
            }
        )
        .group_split(test_size=0.25, validation_size=0.15, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    feats = ["rel_feat", "bid", "ctr_prior"]
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
    industry_metrics = metrics_round(
        {"ndcg_at_k": float(np.mean(ndcgs)), "n_queries": int(len(ndcgs))}
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    for src, dst in (
        ("ndcg", "ndcg_at_k"),
        ("ndcg@5", "ndcg_at_k"),
        ("ndcg_at_5", "ndcg_at_k"),
    ):
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
    print("sponsored-ad-ltr Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
