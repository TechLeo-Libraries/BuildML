"""Integration smoke: Session LTR path + bundle + walkthrough."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for q in range(36):
        for item in range(8):
            f1 = float(rng.normal(q % 5, 1.0))
            f2 = float(item)
            bm25 = float(rng.random())
            rel = float(max(0, int(3 - abs(f1 - (q % 5)) + (item % 3 == 0))))
            rows.append(
                {
                    "query_id": f"q{q}",
                    "item_id": f"i{item}",
                    "f1": f1,
                    "f2": f2,
                    "bm25": bm25,
                    "relevance": rel,
                }
            )
    return pd.DataFrame(rows)


def test_ranking_alpha_smoke(tmp_path: Path) -> None:
    frame = _frame()
    session = (
        Session.ingest(frame)
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
        .group_split(test_size=0.25, validation_size=0.15, random_state=0)
    )

    fit = session.fit_ranker(
        method="pointwise",
        query_column="query_id",
        item_column="item_id",
    )
    assert fit.n_train_rows > 0
    assert fit.n_train_queries > 0
    ranked = session.rank(partition="test", k=5)
    assert ranked.n_queries > 0
    ev = session.evaluate_ranker(partition="test", k=5)
    assert set(ev.metrics) >= {"ndcg_at_k", "map_at_k", "mrr_at_k"}

    bundle = tmp_path / "ranker_bundle"
    session.save_ranker_bundle(bundle)
    assert (bundle / "meta.json").is_file()
    assert (bundle / "ranker_plan.joblib").is_file()

    other = (
        Session.ingest(frame)
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
        .group_split(test_size=0.25, validation_size=0.15, random_state=0)
    )
    other.load_ranker_bundle(bundle)
    assert other.ranker_plan is not None
    assert other.evaluate_ranker(k=5).n_holdout_rows > 0

    walk = session.walkthrough()
    assert walk.ranking_status.get("has_ranker_plan") is True
