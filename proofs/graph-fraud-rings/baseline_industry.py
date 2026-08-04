"""Tier C: networkx degree features + sklearn twin for graph-fraud-rings."""

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extra_available,
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("graph-fraud-rings", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 120
    nodes = pd.DataFrame(
        {
            "node_id": [f"n{i}" for i in range(n)],
            "feat1": rng.normal(size=n),
            "feat2": rng.normal(size=n),
            "is_fraud": [
                1 if i >= n // 2 and rng.random() < 0.35 else 0 for i in range(n)
            ],
        }
    )
    edges = []
    for i in range(n):
        community = i // (n // 2)
        lo, hi = community * (n // 2), (community + 1) * (n // 2)
        for j in rng.choice(range(lo, hi), size=3, replace=True):
            if i != j:
                edges.append((f"n{i}", f"n{int(j)}"))
    edge_frame = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()

    session = Session.ingest(nodes.copy())
    session.set_roles(
        {"node_id": "id", "feat1": "feature", "feat2": "feature", "is_fraud": "target"}
    )
    session.split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    val_idx = list(plan.validation_indices)

    # Graph features from train-visible edges only (no test-node label leakage via edges is OK;
    # still avoid using test labels). Degree computed on full topology is common; we restrict
    # to edges whose BOTH endpoints are in train∪val for a stricter train-graph view.
    visible = set(nodes.loc[train_idx + val_idx, "node_id"])
    deg = {nid: 0 for nid in nodes["node_id"]}
    if extra_available("networkx"):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(nodes["node_id"])
        for src, tgt in edge_frame.itertuples(index=False):
            if src in visible and tgt in visible:
                g.add_edge(src, tgt)
        deg = dict(g.degree())
        clustering = nx.clustering(g)
    else:
        clustering = {nid: 0.0 for nid in nodes["node_id"]}
        for src, tgt in edge_frame.itertuples(index=False):
            if src in visible and tgt in visible:
                deg[src] += 1
                deg[tgt] += 1

    feat = nodes.copy()
    feat["degree"] = feat["node_id"].map(deg).astype(float)
    feat["clustering"] = feat["node_id"].map(clustering).astype(float)
    cols = ["feat1", "feat2", "degree", "clustering"]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(feat.loc[train_idx, cols])
    y_train = feat.loc[train_idx, "is_fraud"].to_numpy()
    x_test = scaler.transform(feat.loc[test_idx, cols])
    y_test = feat.loc[test_idx, "is_fraud"].to_numpy()
    clf = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)[:, 1]
    pred = clf.predict(x_test)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw, prefer=("test_metrics",), keys=("accuracy", "f1", "roc_auc", "f1_weighted")
    )
    if "f1" not in bml_metrics and "f1_weighted" in bml_metrics:
        bml_metrics["f1"] = bml_metrics["f1_weighted"]

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.graph.fit(classical)",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "networkx degree/clustering + sklearn LogisticRegression",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Node split before fit",
                "Graph structural features from train∪val visible graph only",
                "Classifier fit on train nodes only",
                "Test evaluated after lock",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
    )
    print("graph-fraud-rings Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
