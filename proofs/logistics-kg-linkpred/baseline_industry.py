"""Tier C: PMI co-occurrence twin for logistics-kg-linkpred."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from collections import defaultdict

import numpy as np
import pandas as pd

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _logistics_triples(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    warehouses = [f"wh{i}" for i in range(18)]
    hubs = [f"hub{i}" for i in range(8)]
    routes = [f"rt{i}" for i in range(14)]
    carriers = [f"car{i}" for i in range(6)]
    triples = []
    for i, wh in enumerate(warehouses):
        triples.append((wh, "ships_via", routes[i % len(routes)]))
        triples.append((routes[i % len(routes)], "serves", hubs[i % len(hubs)]))
        triples.append((carriers[i % len(carriers)], "operates", routes[i % len(routes)]))
        triples.append((wh, "feeds", hubs[i % len(hubs)]))
    for _ in range(40):
        a, b = rng.choice(warehouses, size=2, replace=False)
        triples.append((str(a), "transfers_to", str(b)))
    return (
        pd.DataFrame(triples, columns=["head", "relation", "tail"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _rank_metrics(ranks: list[float]) -> dict:
    ranks_a = np.asarray(ranks, dtype=float)
    hits = lambda k: float(np.mean(ranks_a <= k))
    return {
        "hits_at_1": hits(1),
        "hits_at_3": hits(3),
        "hits_at_10": hits(10),
        "mean_rank": float(np.mean(ranks_a)),
        "mrr": float(np.mean(1.0 / ranks_a)),
    }


def main() -> None:
    ctx = new_proof_context("logistics-kg-linkpred", seed=115)
    frame = _logistics_triples(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train = frame.loc[list(plan.train_indices)]
    test = frame.loc[list(plan.test_indices)]

    hr_count: dict[tuple[str, str], int] = defaultdict(int)
    rt_count: dict[tuple[str, str], int] = defaultdict(int)
    r_count: dict[str, int] = defaultdict(int)
    true_by_hr: dict[tuple[str, str], set[str]] = defaultdict(set)
    entities = set(frame["head"]) | set(frame["tail"])
    for h, r, t in train.itertuples(index=False):
        hr_count[(h, r)] += 1
        rt_count[(r, t)] += 1
        r_count[r] += 1
        true_by_hr[(h, r)].add(t)

    ranks = []
    for h, r, t_true in test.itertuples(index=False):
        scores = []
        known = true_by_hr.get((h, r), set())
        for t in entities:
            if t != t_true and t in known:
                continue
            s = (hr_count[(h, r)] * rt_count[(r, t)]) / max(r_count[r], 1)
            scores.append((s, t))
        scores.sort(key=lambda x: (-x[0], x[1]))
        rank = next(
            (i + 1 for i, (_, t) in enumerate(scores) if t == t_true),
            len(entities),
        )
        ranks.append(float(rank))

    industry_metrics = metrics_round(_rank_metrics(ranks))
    bml = load_buildml_results(ctx.project_dir)
    bml_metrics = metrics_round(dict(bml.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={"backend": "buildml.kg/transe", "test_metrics": bml_metrics},
        industry={
            "backend": "train co-occurrence PMI / filtered ranking",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same triple split as BuildML (seed=115)",
                "Co-occurrence counts from train triples only",
                "Filtered ranking removes other train-true tails",
            ],
        },
        split_counts={
            "train": len(plan.train_indices),
            "validation": len(plan.validation_indices),
            "test": len(plan.test_indices),
        },
        delta_keys=("hits_at_1", "hits_at_3", "hits_at_10", "mean_rank", "mrr"),
    )
    print("logistics-kg-linkpred Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
