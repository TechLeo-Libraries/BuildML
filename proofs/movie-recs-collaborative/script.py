"""Tier A proof: movie-recs-collaborative."""

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


def _ratings(n_users=80, n_items=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        liked = rng.choice(n_items, size=max(8, n_items // 6), replace=False)
        for item in liked:
            rows.append({
                "user_id": f"u{u}",
                "item_id": f"i{item}",
                "rating": float(rng.integers(3, 6)),
                "genre_code": float(item % 7),
                "pop": float((item * 3) % 11),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ctx = new_proof_context("movie-recs-collaborative", seed=0)
    frame = _ratings(seed=ctx.seed)
    impl_ok = extra_available("implicit")
    session = (
        Session.ingest(frame)
        .set_roles({
            "user_id": "id", "item_id": "id", "rating": "target",
            "genre_code": "feature", "pop": "feature",
        })
        .split(test_size=0.2, validation_size=0.15, random_state=ctx.seed)
    )
    method = "als" if impl_ok else "item_knn"
    try:
        if impl_ok:
            fit = session.fit_recommender(
                method="als", feedback="implicit",
                user_column="user_id", item_column="item_id",
                random_state=ctx.seed,
            )
        else:
            fit = session.fit_recommender(
                method="item_knn",
                user_column="user_id", item_column="item_id",
                n_neighbors=25, random_state=ctx.seed,
            )
            method = "item_knn"
    except (MissingExtraError, TypeError, ValueError):
        fit = session.fit_recommender(
            method="item_knn",
            user_column="user_id", item_column="item_id",
            n_neighbors=25, random_state=ctx.seed,
        )
        method = "item_knn"
    recs = session.recommend(partition="test", k=5)
    ev = session.evaluate_recommender(partition="test", k=5)
    bundle = session.save_recommender_bundle(ctx.artifacts_dir / "rec_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_movie_ratings", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "implicit_available": impl_ok,
        "method": method,
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "recommend_sample": metrics_round(recs.to_dict() if hasattr(recs, "to_dict") else {}),
        "test_metrics": metrics_round(dict(ev.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": ["Split before fit", "Train-only recommender fit", "Test metrics after lock"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: popularity + item-cosine twin on the same split; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Synthetic interactions",
            "ALS uses implicit when installed; otherwise item_knn fallback",
        ],
    })
    print("movie-recs-collaborative OK", dict(ev.metrics))


if __name__ == "__main__":
    main()
