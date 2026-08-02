"""Session LTR loop: fit_ranker → rank → evaluate_ranker → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def _synthetic_judgments(
    n_queries: int = 48, n_items: int = 10, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for q in range(n_queries):
        q_center = float(q % 6)
        for item in range(n_items):
            f1 = float(rng.normal(q_center, 0.8))
            f2 = float(rng.normal(item / 2.0, 0.5))
            bm25 = float(rng.random())
            # Graded relevance correlated with feature alignment
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


def main() -> None:
    frame = _synthetic_judgments()
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
        pointwise_estimator="ridge",
        random_state=0,
    )
    print("fit", fit.to_dict())

    ranked = session.rank(partition="test", k=5)
    print("rank", ranked.to_dict())

    ev = session.evaluate_ranker(partition="test", k=5)
    print("eval", ev.metrics)

    out = Path("artifacts/ranker_demo_bundle")
    session.save_ranker_bundle(out)
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
    other.load_ranker_bundle(out)
    print("reloaded eval", other.evaluate_ranker(partition="test", k=5).metrics)


if __name__ == "__main__":
    main()
