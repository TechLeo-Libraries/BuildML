"""Tier B product: Lattice Supply Graph.

Composes classical graph node features + knowledge-graph link prediction +
classical supervised delay / default risk on supply-network nodes.
"""

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
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import metrics_round, new_proof_context, write_results


FEATURES = ["lead_time_z", "fill_rate_z", "cost_z", "reliability_z"]
TARGET = "late_risk"


def _supply_network(n: int = 160, seed: int = 54):
    rng = np.random.default_rng(seed)
    community = np.array([0] * (n // 2) + [1] * (n - n // 2))
    lead = rng.normal(0, 1, size=n) + 0.9 * community
    fill = rng.normal(0, 1, size=n) - 0.7 * community
    cost = rng.normal(0, 1, size=n) + 0.4 * community
    rel = rng.normal(0, 1, size=n) - 0.55 * community
    logit = -1.2 + 0.9 * lead - 0.7 * fill + 0.45 * cost - 0.6 * rel + rng.normal(
        0, 0.35, size=n
    )
    late = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    nodes = pd.DataFrame(
        {
            "node_id": [f"sup-{i}" for i in range(n)],
            "lead_time_z": lead,
            "fill_rate_z": fill,
            "cost_z": cost,
            "reliability_z": rel,
            "late_risk": late,
        }
    )
    edges = []
    half = n // 2
    for i in range(n):
        lo, hi = (0, half) if i < half else (half, n)
        for j in rng.choice(range(lo, hi), size=4, replace=True):
            if i != int(j):
                edges.append((f"sup-{i}", f"sup-{int(j)}"))
    edge_frame = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    meta = {
        "name": "lattice_supply_network",
        "license": "synthetic/public-domain",
        "n_nodes": n,
        "n_edges": int(len(edge_frame)),
        "positive_rate": float(late.mean()),
    }
    return nodes, edge_frame, meta


def _supply_kg(seed: int = 54) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    warehouses = [f"wh{i}" for i in range(16)]
    hubs = [f"hub{i}" for i in range(8)]
    routes = [f"rt{i}" for i in range(12)]
    carriers = [f"car{i}" for i in range(6)]
    triples = []
    for i, wh in enumerate(warehouses):
        triples.append((wh, "ships_via", routes[i % len(routes)]))
        triples.append((routes[i % len(routes)], "serves", hubs[i % len(hubs)]))
        triples.append((carriers[i % len(carriers)], "operates", routes[i % len(routes)]))
        triples.append((wh, "feeds", hubs[i % len(hubs)]))
    for _ in range(35):
        a, b = rng.choice(warehouses, size=2, replace=False)
        triples.append((str(a), "transfers_to", str(b)))
    return (
        pd.DataFrame(triples, columns=["head", "relation", "tail"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def main() -> None:
    ctx = new_proof_context("lattice-supply-graph", seed=54)
    nodes, edges, data_meta = _supply_network(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = Session.ingest(nodes.copy())
    session.set_roles(
        {
            "node_id": "id",
            **{c: "feature" for c in FEATURES},
            TARGET: "target",
        }
    )
    session.split(
        test_size=0.2,
        validation_size=0.2,
        stratify=True,
        random_state=ctx.seed,
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    session.scale(method="standard")

    # --- Stage 1: classical graph ---
    try:
        session.set_graph(
            edges,
            source_col="source",
            target_col="target",
            node_id_col="node_id",
        )
        g_fit = session.fit_graph(
            method="classical", mode="inductive", random_state=ctx.seed
        )
        g_val = session.evaluate_graph(partition="validation")
        g_test = session.evaluate_graph(partition="test")
        stages["graph"] = {
            "status": "ok",
            "fit": metrics_round(g_fit.to_dict() if hasattr(g_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(g_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(g_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["graph"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"graph: {exc}")
    write_results(ctx, stages["graph"], filename="graph.json")

    # --- Stage 2: KG link prediction ---
    kg_frame = _supply_kg(seed=ctx.seed)
    try:
        kg_session = (
            Session.ingest(kg_frame)
            .set_roles({"head": "id", "relation": "id", "tail": "id"})
            .split(test_size=0.2, validation_size=0.1, random_state=ctx.seed)
        )
        kg_fit = kg_session.fit_kg(
            method="transe",
            head_column="head",
            relation_column="relation",
            tail_column="tail",
            embedding_dim=32,
            epochs=40,
            batch_size=64,
            learning_rate=0.05,
            neg_ratio=2,
            random_state=ctx.seed,
        )
        kg_val = kg_session.evaluate_kg(partition="validation")
        kg_test = kg_session.evaluate_kg(partition="test")
        stages["kg"] = {
            "status": "ok",
            "n_triples": int(len(kg_frame)),
            "fit": metrics_round(kg_fit.to_dict() if hasattr(kg_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                dict(getattr(kg_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(dict(getattr(kg_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["kg"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"kg: {exc}")
    write_results(ctx, stages["kg"], filename="kg.json")

    # --- Stage 3: classical supervised late risk ---
    c_session = (
        Session.ingest(nodes.copy())
        .set_roles(
            {
                "node_id": "id",
                **{c: "feature" for c in FEATURES},
                TARGET: "target",
            }
        )
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
    c_session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    c_val = c_session.evaluate(partition="validation")
    c_test = c_session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(c_val.metrics)),
        "test_metrics": metrics_round(dict(c_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    summary = {
        "status": "completed",
        "product": "Lattice Supply Graph",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified node split before graph / supervised fit",
            "Classical graph features from train graph view",
            "KG triple split before TransE",
            "Test evaluate after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Graph features conditioned on test labels overstate community risk",
            "Training TransE on all triples makes link metrics meaningless",
            "Supervised late-risk trained with test rows overstates TMS readiness",
        ],
        "limitations": [
            "Synthetic supplier communities — not a licensed TMS extract",
            "Classical graph path is primary",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "lattice-supply-graph OK",
        {
            "supervised_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
