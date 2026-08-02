"""Tier A proof: search-relevance-ltr."""

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

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import extra_available, metrics_round, new_proof_context, write_results


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
            rows.append({
                "query_id": f"q{q}", "item_id": f"d{item}",
                "f1": f1, "f2": f2, "bm25": bm25, "relevance": rel,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ctx = new_proof_context("search-relevance-ltr", seed=0)
    frame = _judgments(seed=ctx.seed)
    lgbm = extra_available("lightgbm")
    session = (
        Session.ingest(frame)
        .set_roles({
            "query_id": "group", "item_id": "id", "relevance": "target",
            "f1": "feature", "f2": "feature", "bm25": "feature",
        })
        .group_split(test_size=0.25, validation_size=0.15, random_state=ctx.seed)
    )
    method = "lambdarank" if lgbm else "pointwise"
    try:
        if lgbm:
            fit = session.fit_ranker(
                method="lambdarank", query_column="query_id", item_column="item_id",
                random_state=ctx.seed,
            )
        else:
            raise MissingExtraError("ranking-industry", "lambdarank")
    except (MissingExtraError, TypeError, ValueError):
        fit = session.fit_ranker(
            method="pointwise", query_column="query_id", item_column="item_id",
            pointwise_estimator="ridge", random_state=ctx.seed,
        )
        method = "pointwise"
    ranked = session.rank(partition="test", k=5)
    ev = session.evaluate_ranker(partition="test", k=5)
    bundle = session.save_ranker_bundle(ctx.artifacts_dir / "ranker_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_ltr_judgments", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "lightgbm_available": lgbm,
        "method": method,
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "rank_sample": metrics_round(ranked.to_dict() if hasattr(ranked, "to_dict") else {}),
        "test_metrics": metrics_round(dict(ev.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": ["group_split on query_id", "Train-only ranker fit", "Test nDCG after lock"],
        "industry_comparison": {"status": "stub"},
        "limitations": ["Synthetic graded judgments"],
    })
    print("search-relevance-ltr OK", dict(ev.metrics))


if __name__ == "__main__":
    main()
