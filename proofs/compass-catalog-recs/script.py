"""Tier B product: Compass Catalog Recs.

Composes collaborative recommenders + classical graph node features over
item co-purchase edges + classical supervised repurchase scoring.
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
from proofs._lib import (
    extra_available,
    load_catalog_interactions_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _item_graph(interactions: pd.DataFrame, seed: int):
    """Build item co-purchase edges + node features from interactions."""
    rng = np.random.default_rng(seed)
    # Pair items that share users
    user_items = interactions.groupby("user_id")["item_id"].apply(list)
    edges = set()
    for items in user_items:
        if len(items) < 2:
            continue
        for _ in range(min(6, len(items))):
            a, b = rng.choice(items, size=2, replace=False)
            if a != b:
                edges.add(tuple(sorted((a, b))))
    edge_frame = pd.DataFrame(list(edges), columns=["source", "target"])
    item_stats = (
        interactions.groupby("item_id")
        .agg(
            mean_rating=("rating", "mean"),
            n_users=("user_id", "nunique"),
            category_code=("category_code", "first"),
            price_band=("price_band", "first"),
        )
        .reset_index()
    )
    # Synthetic repurchase label from popularity + category
    logit = (
        -1.0
        + 0.35 * (item_stats["mean_rating"] - 3.5)
        + 0.04 * item_stats["n_users"]
        + 0.08 * item_stats["category_code"]
        + rng.normal(0, 0.35, size=len(item_stats))
    )
    item_stats["repurchase"] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    item_stats = item_stats.rename(columns={"item_id": "node_id"})
    return item_stats, edge_frame


def main() -> None:
    ctx = new_proof_context("compass-catalog-recs", seed=113)
    interactions, data_meta = load_catalog_interactions_synthetic(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: recommender ---
    try:
        impl_ok = extra_available("implicit")
        rec_session = (
            Session.ingest(interactions.copy())
            .set_roles(
                {
                    "user_id": "id",
                    "item_id": "id",
                    "rating": "target",
                    "category_code": "feature",
                    "price_band": "feature",
                }
            )
            .split(test_size=0.2, validation_size=0.15, random_state=ctx.seed)
        )
        method = "als" if impl_ok else "item_knn"
        try:
            if impl_ok:
                fit = rec_session.recommender.fit(
                    method="als",
                    feedback="implicit",
                    user_column="user_id",
                    item_column="item_id",
                    random_state=ctx.seed,
                )
            else:
                fit = rec_session.recommender.fit(
                    method="item_knn",
                    user_column="user_id",
                    item_column="item_id",
                    n_neighbors=25,
                    random_state=ctx.seed,
                )
                method = "item_knn"
        except (MissingExtraError, TypeError, ValueError):
            fit = rec_session.recommender.fit(
                method="item_knn",
                user_column="user_id",
                item_column="item_id",
                n_neighbors=25,
                random_state=ctx.seed,
            )
            method = "item_knn"
        recs = rec_session.recommender.recommend(partition="test", k=5)
        ev = rec_session.recommender.evaluate(partition="test", k=5)
        stages["recommender"] = {
            "status": "ok",
            "method": method,
            "implicit_available": impl_ok,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "recommend_sample": metrics_round(
                recs.to_dict() if hasattr(recs, "to_dict") else {}
            ),
            "test_metrics": metrics_round(dict(ev.metrics)),
        }
        write_results(ctx, stages["recommender"], filename="recommender.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["recommender"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"recommender: {exc}")

    # --- Stage 2: graph over item co-purchase ---
    nodes, edges = _item_graph(interactions, seed=ctx.seed)
    try:
        g_session = Session.ingest(nodes.copy())
        g_session.set_roles(
            {
                "node_id": "id",
                "mean_rating": "feature",
                "n_users": "feature",
                "category_code": "feature",
                "price_band": "feature",
                "repurchase": "target",
            }
        )
        g_session.split(
            test_size=0.25,
            validation_size=0.15,
            stratify=True,
            random_state=ctx.seed,
        )
        g_session.graph.set_spec(
            edges,
            source_col="source",
            target_col="target",
            node_id_col="node_id",
        )
        g_fit = g_session.graph.fit(method="classical", mode="inductive", random_state=ctx.seed)
        g_ev = g_session.graph.evaluate(partition="test")
        plan_g = g_session.split_plan
        assert plan_g is not None
        stages["graph"] = {
            "status": "ok",
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
            "fit": metrics_round(g_fit.to_dict() if hasattr(g_fit, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(g_ev, "metrics", {}) or {})),
            "split_counts": {
                "train": len(plan_g.train_indices),
                "validation": len(plan_g.validation_indices),
                "test": len(plan_g.test_indices),
            },
        }
        write_results(ctx, stages["graph"], filename="graph.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["graph"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"graph: {exc}")
        g_session = None
        plan_g = None

    # --- Stage 3: classical supervised on item nodes ---
    try:
        if plan_g is None:
            classical = (
                Session.ingest(nodes.copy())
                .set_roles(
                    {
                        "node_id": "id",
                        "mean_rating": "feature",
                        "n_users": "feature",
                        "category_code": "feature",
                        "price_band": "feature",
                        "repurchase": "target",
                    }
                )
                .split(
                    test_size=0.25,
                    validation_size=0.15,
                    stratify=True,
                    random_state=ctx.seed,
                )
                .scale(method="standard")
            )
        else:
            classical = (
                Session.ingest(nodes.copy())
                .set_roles(
                    {
                        "node_id": "id",
                        "mean_rating": "feature",
                        "n_users": "feature",
                        "category_code": "feature",
                        "price_band": "feature",
                        "repurchase": "target",
                    }
                )
                .inject_split(
                    train_indices=list(plan_g.train_indices),
                    validation_indices=list(plan_g.validation_indices),
                    test_indices=list(plan_g.test_indices),
                )
                .scale(method="standard")
            )
        classical.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = classical.evaluate(partition="test")
        stages["classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "test_metrics": metrics_round(dict(c_test.metrics)),
        }
        write_results(ctx, stages["classical"], filename="classical.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["classical"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"classical: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Compass Catalog Recs",
        "data": data_meta,
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Recommender split before fit; train-only ALS / item_knn",
            "Graph node split before classical graph features",
            "Classical repurchase scorer uses the same node inject_split",
            "Test session.recommender.recommend / session.graph.evaluate / evaluate after locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting recommenders on test interactions invents recall@k",
            "Graph features conditioned on test labels overstate ring repurchase",
            "Fitting classical scores on the full catalog invents holdout ROC",
        ],
        "limitations": [
            "Synthetic catalog interactions — not a real retail extract",
            "Co-purchase graph is derived from the same interactions table",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "compass-catalog-recs OK",
        {
            "recommender": (stages.get("recommender") or {}).get("status"),
            "graph": (stages.get("graph") or {}).get("status"),
            "classical": (stages.get("classical") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
