"""Integration smoke: Session recommender path + bundle + walkthrough."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_recommender_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(45):
        liked = rng.choice(28, size=9, replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{item}",
                    "rating": float(rng.integers(2, 6)),
                    "f1": float(item % 4),
                    "f2": float(item // 4),
                }
            )
    frame = pd.DataFrame(rows)
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "user_id": "id",
                "item_id": "id",
                "rating": "target",
                "f1": "feature",
                "f2": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=0)
    )

    fit = session.fit_recommender(
        method="item_knn",
        user_column="user_id",
        item_column="item_id",
        n_neighbors=20,
    )
    assert fit.n_train_interactions > 0
    recs = session.recommend(partition="test", k=5)
    assert recs.n_users > 0
    ev = session.evaluate_recommender(partition="test", k=5)
    assert set(ev.metrics) >= {
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "map_at_k",
    }

    bundle = tmp_path / "recommender_bundle"
    session.save_recommender_bundle(bundle)
    assert (bundle / "meta.json").is_file()
    assert (bundle / "recommender_plan.joblib").is_file()

    other = (
        Session.ingest(frame)
        .set_roles(
            {
                "user_id": "id",
                "item_id": "id",
                "rating": "target",
                "f1": "feature",
                "f2": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=0)
    )
    other.load_recommender_bundle(bundle, trusted=True)
    assert other.recommender_plan is not None
    assert other.evaluate_recommender(k=5).n_holdout_interactions > 0

    walk = session.walkthrough()
    assert walk.recommender_status.get("has_recommender_plan") is True
