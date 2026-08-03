"""Integration smoke: Session KG path + bundle + walkthrough."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    people = [f"p{i}" for i in range(28)]
    orgs = [f"o{i}" for i in range(6)]
    cities = [f"c{i}" for i in range(4)]
    rows: list[dict[str, str]] = []
    for i, p in enumerate(people):
        rows.append({"head": p, "relation": "works_at", "tail": orgs[i % len(orgs)]})
        rows.append({"head": p, "relation": "lives_in", "tail": cities[i % len(cities)]})
        rows.append(
            {
                "head": orgs[i % len(orgs)],
                "relation": "located_in",
                "tail": cities[i % len(cities)],
            }
        )
        rows.append({"head": p, "relation": "knows", "tail": people[(i + 1) % len(people)]})
    for _ in range(50):
        a, b = rng.choice(people, size=2, replace=False)
        rows.append({"head": str(a), "relation": "knows", "tail": str(b)})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def test_kg_alpha_smoke(tmp_path: Path) -> None:
    frame = _frame()
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=0)
    )

    fit = session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=24,
        epochs=40,
        batch_size=64,
        learning_rate=0.05,
        neg_ratio=2,
        random_state=0,
    )
    assert fit.n_train_triples > 0
    assert fit.n_entities > 0
    assert fit.n_relations > 0

    preds = session.predict_links(
        mode="tail", heads=["p0"], relations=["works_at"], k=5
    )
    assert preds.n_queries == 1
    assert len(preds.predictions[0]) > 0

    nbrs = session.query_kg(mode="neighbors", entity="p0")
    assert nbrs.n_results >= 1

    path = session.query_kg(mode="path", source="p0", target="c0", max_hops=3)
    assert path.mode == "path"

    ev = session.evaluate_kg(partition="test", k=5)
    assert set(ev.metrics) >= {"mrr", "hits_at_1", "hits_at_3", "hits_at_5"}

    bundle = tmp_path / "kg_bundle"
    session.save_kg_bundle(bundle)
    assert (bundle / "meta.json").is_file()
    assert (bundle / "kg_plan.joblib").is_file()

    other = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=0)
    )
    other.load_kg_bundle(bundle, trusted=True)
    assert other.kg_plan is not None
    ev2 = other.evaluate_kg(partition="test", k=5)
    assert ev2.n_triples_scored == ev.n_triples_scored
    assert ev2.metrics["mrr"] == pytest.approx(ev.metrics["mrr"])

    walk = session.walkthrough()
    assert walk.kg_status.get("has_kg_plan") is True
