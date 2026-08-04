"""Depth tests: leakage, DistMult, filtered metrics, symbolic query, relation pred."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.kg.features import hits_at_k, mrr_from_ranks
from buildml.kg.query import query_kg


def _kg_frame(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    people = [f"p{i}" for i in range(20)]
    orgs = [f"o{i}" for i in range(5)]
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
    for _ in range(30):
        a, b = rng.choice(people, size=2, replace=False)
        rows.append({"head": str(a), "relation": "knows", "tail": str(b)})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _session() -> Session:
    return (
        Session.ingest(_kg_frame())
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=2)
    )


def test_metric_helpers() -> None:
    assert mrr_from_ranks([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)
    assert hits_at_k([1, 2, 5], 3) == pytest.approx(2 / 3)


def test_never_trains_on_test_triples() -> None:
    session = _session()
    session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=16,
        epochs=20,
        batch_size=32,
        random_state=0,
    )
    plan = session.kg_plan
    assert plan is not None
    train_set = set(
        zip(
            [plan.entity_ids[i] for i in plan.train_heads_],
            [plan.relation_ids[i] for i in plan.train_relations_],
            [plan.entity_ids[i] for i in plan.train_tails_],
            strict=True,
        )
    )
    assert plan.n_train_triples == len(train_set)
    train_frame = session.dataset.frame.iloc[list(session._split_plan.train_indices)]
    train_unique = train_frame[["head", "relation", "tail"]].dropna().drop_duplicates()
    assert plan.n_train_triples == len(train_unique)


def test_distmult_fit_and_eval() -> None:
    session = _session()
    fit = session.fit_kg(
        method="distmult",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=16,
        epochs=25,
        batch_size=32,
        learning_rate=0.05,
        neg_ratio=2,
        random_state=0,
    )
    assert fit.method == "distmult"
    assert fit.neg_ratio == 2
    ev = session.evaluate_kg(partition="test", k=5)
    assert set(ev.metrics) >= {"mrr", "hits_at_1", "hits_at_3", "hits_at_5"}
    assert ev.n_triples_scored >= 0


def test_relation_prediction_mode() -> None:
    session = _session()
    session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=16,
        epochs=20,
        random_state=0,
    )
    plan = session.kg_plan
    assert plan is not None
    h = plan.entity_ids[0]
    t = plan.entity_ids[min(1, len(plan.entity_ids) - 1)]
    pred = session.predict_links(mode="relation", heads=[h], tails=[t], k=3)
    assert pred.mode == "relation"
    assert pred.n_queries == 1
    assert len(pred.predictions[0]) <= 3


def test_symbolic_neighbors_typed_path() -> None:
    session = _session()
    session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=8,
        epochs=10,
        random_state=0,
    )
    plan = session.kg_plan
    assert plan is not None

    # Pick a train entity that has out-edges
    src_id = next(iter(plan.out_edges_))
    src = plan.entity_ids[src_id]
    nbrs = query_kg(plan, mode="neighbors", entity=src, direction="out")
    assert nbrs.n_results >= 1

    # Typed works_at if present
    if "works_at" in plan.relation_ids:
        # Find someone with works_at
        rid = plan.relation_index_["works_at"]
        person = None
        for eid, edges in plan.out_edges_.items():
            if any(r == rid for r, _ in edges):
                person = plan.entity_ids[eid]
                break
        if person is not None:
            typed = query_kg(
                plan, mode="typed", entity=person, relation="works_at"
            )
            assert typed.n_results >= 1

    # Path between two connected train entities
    # Take first out-edge end as target via one hop
    r0, dst_id = plan.out_edges_[src_id][0]
    dst = plan.entity_ids[dst_id]
    path = query_kg(plan, mode="path", source=src, target=dst, max_hops=2)
    assert path.n_results >= 1
    assert path.results[0][0] == src
    assert path.results[-1][2] == dst


def test_query_does_not_see_holdout_only_edges() -> None:
    """Holdout-only triples must not appear in train adjacency queries."""
    rows = [
        ("a", "r", "x"),
        ("b", "r", "x"),
        ("a", "s", "y"),
        ("b", "s", "y"),
        ("c", "r", "x"),
        ("c", "s", "z"),
        ("d", "r", "y"),
        ("d", "s", "z"),
        ("a", "r", "z"),
        ("b", "r", "z"),
        ("HOLD_H", "secret", "HOLD_T"),
    ]
    frame = pd.DataFrame(rows, columns=["head", "relation", "tail"])
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.25, validation_size=0.15, random_state=0)
    )
    session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=8,
        epochs=8,
        random_state=0,
    )
    plan = session.kg_plan
    assert plan is not None
    # Secret relation appears only once; if it landed in holdout it is absent
    # from train vocab. If it landed in train, assert typed query still only
    # returns train edges (no invented neighbors).
    if "secret" not in plan.relation_index_:
        q = session.query_kg(mode="typed", entity="HOLD_H", relation="secret")
        assert q.n_results == 0
    else:
        # Relation in train: neighbors of HOLD_H must be exactly train tails
        q = session.query_kg(mode="neighbors", entity="HOLD_H", direction="out")
        train_tails = {
            plan.entity_ids[n]
            for r, n in plan.out_edges_.get(plan.entity_index_["HOLD_H"], [])
        }
        assert {row[0] for row in q.results} == train_tails


def test_native_bundle_roundtrip_reevaluate(tmp_path) -> None:
    """Save → load → re-evaluate must reproduce filtered ranking metrics."""
    session = _session()
    session.kg.fit(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=16,
        epochs=25,
        batch_size=32,
        learning_rate=0.05,
        neg_ratio=2,
        random_state=0,
    )
    ev = session.kg.evaluate(partition="test", k=5)
    assert set(ev.metrics) >= {"mrr", "hits_at_1", "hits_at_3", "hits_at_5", "mean_rank"}
    assert ev.n_triples_scored >= 1

    bundle = tmp_path / "kg_native_bundle"
    session.kg.save_bundle(bundle)

    other = (
        Session.ingest(_kg_frame())
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=2)
    )
    other.kg.load_bundle(bundle, trusted=True)
    assert other.kg.plan is not None
    ev2 = other.kg.evaluate(partition="test", k=5)
    assert ev2.n_triples_scored == ev.n_triples_scored
    assert ev2.metrics["mrr"] == pytest.approx(ev.metrics["mrr"])
    assert ev2.metrics["mean_rank"] == pytest.approx(ev.metrics["mean_rank"])
    assert ev2.metrics["hits_at_5"] == pytest.approx(ev.metrics["hits_at_5"])
