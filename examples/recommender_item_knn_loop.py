"""Session recommender loop: fit → recommend → evaluate → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def _synthetic_ratings(n_users: int = 40, n_items: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for user in range(n_users):
        liked = rng.choice(n_items, size=8, replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{item}",
                    "rating": float(rng.integers(3, 6)),
                    "f1": float(item % 5),
                    "f2": float(item // 5),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    artifacts = Path(__file__).resolve().parent / '.artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    frame = _synthetic_ratings()
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

    fit = session.recommender.fit(
        method=None,
        feedback="explicit",
        user_column="user_id",
        item_column="item_id",
        n_neighbors=20,
    )
    print("fit", fit.to_dict())

    recs = session.recommender.recommend(partition="test", k=5)
    print("recommend", recs.to_dict())

    ev = session.recommender.evaluate(partition="test", k=5)
    print("eval", ev.metrics)

    out = Path(__file__).resolve().parent / ".artifacts" / "recommender_demo_bundle"
    session.recommender.save_bundle(out)
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
    other.recommender.load_bundle(out, trusted=True)
    print("reloaded eval", other.recommender.evaluate(partition="test", k=5).metrics)


if __name__ == "__main__":
    main()
