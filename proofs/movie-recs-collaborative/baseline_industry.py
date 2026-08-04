"""Tier C: popularity + item-cosine twin for movie-recs-collaborative."""

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
from sklearn.metrics.pairwise import cosine_similarity

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _ratings(n_users=80, n_items=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        liked = rng.choice(n_items, size=max(8, n_items // 6), replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": f"i{item}",
                    "rating": float(rng.integers(3, 6)),
                    "genre_code": float(item % 7),
                    "pop": float((item * 3) % 11),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ctx = new_proof_context("movie-recs-collaborative", seed=0)
    frame = _ratings(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "user_id": "id",
                "item_id": "id",
                "rating": "target",
                "genre_code": "feature",
                "pop": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train = frame.loc[list(plan.train_indices)].copy()
    test = frame.loc[list(plan.test_indices)].copy()

    users = sorted(train["user_id"].unique())
    items = sorted(frame["item_id"].unique())
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {it: i for i, it in enumerate(items)}
    mat = np.zeros((len(users), len(items)), dtype=float)
    for row in train.itertuples(index=False):
        if row.user_id in u_index and row.item_id in i_index:
            mat[u_index[row.user_id], i_index[row.item_id]] = row.rating

    # Item-item cosine on train interactions.
    item_sim = cosine_similarity(mat.T)
    pop = train.groupby("item_id").size().sort_values(ascending=False)
    pop_items = list(pop.index)

    k = 5
    hits = 0
    total = 0
    ndcg_sum = 0.0
    for user, grp in test.groupby("user_id"):
        truth = set(grp["item_id"])
        if user not in u_index:
            # Cold-start: popularity baseline.
            ranked = [it for it in pop_items if it not in set(train.loc[train.user_id == user, "item_id"])][:k]
        else:
            uvec = mat[u_index[user]]
            seen = set(train.loc[train.user_id == user, "item_id"])
            scores = item_sim @ uvec
            order = np.argsort(-scores)
            ranked = []
            for idx in order:
                it = items[idx]
                if it in seen:
                    continue
                ranked.append(it)
                if len(ranked) >= k:
                    break
            if not ranked:
                ranked = [it for it in pop_items if it not in seen][:k]
        total += 1
        hit_list = [1 if it in truth else 0 for it in ranked]
        hits += int(any(hit_list))
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hit_list))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(truth))))
        ndcg_sum += float(dcg / idcg) if idcg > 0 else 0.0

    industry_metrics = metrics_round(
        {
            "hit_rate_at_k": float(hits / max(total, 1)),
            "ndcg_at_k": float(ndcg_sum / max(total, 1)),
            "n_eval_users": int(total),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    # Normalize common recommender keys.
    for src, dst in (
        ("hit_rate", "hit_rate_at_k"),
        ("hr_at_k", "hit_rate_at_k"),
        ("ndcg", "ndcg_at_k"),
        ("ndcg@5", "ndcg_at_k"),
    ):
        if src in bml_metrics and dst not in bml_metrics:
            bml_metrics[dst] = bml_metrics[src]
    bml_metrics = metrics_round(bml_metrics)

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.recommender.fit",
            "method": bml_raw.get("method", "item_knn"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn item-cosine + popularity cold-start",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Item similarity fit on train interactions only",
                "Test users evaluated after model lock",
                "Same SplitPlan as BuildML Session",
            ],
        },
        split_counts={
            "train": len(plan.train_indices),
            "validation": len(plan.validation_indices),
            "test": len(plan.test_indices),
        },
        delta_keys=("hit_rate_at_k", "ndcg_at_k"),
    )
    print("movie-recs-collaborative Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
