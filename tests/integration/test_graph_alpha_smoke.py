"""Integration smoke: Session graph path + bundle + walkthrough."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.graph.extras import networkx_available


def _community_graph(n_per: int = 40, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_per * 2
    labels = np.array([0] * n_per + [1] * n_per)
    x = labels.astype(float).reshape(-1, 1) + rng.normal(scale=0.35, size=(n, 2))
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(n),
            "f1": x[:, 0],
            "f2": x[:, 1],
            "y": labels,
        }
    )
    edges: list[tuple[int, int]] = []
    for start in (0, n_per):
        members = list(range(start, start + n_per))
        for i in members:
            for j in members:
                if i < j and rng.random() < 0.18:
                    edges.append((i, j))
    for i in range(n_per):
        for j in range(n_per, n):
            if rng.random() < 0.02:
                edges.append((i, j))
    return nodes, pd.DataFrame(edges, columns=["source", "target"])


@pytest.mark.skipif(not networkx_available(), reason="buildml[graph] / networkx missing")
def test_graph_alpha_smoke(tmp_path: Path) -> None:
    nodes, edges = _community_graph()
    session = (
        Session.ingest(nodes)
        .set_roles(
            {
                "node_id": "id",
                "f1": "feature",
                "f2": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    session.set_graph(edges, node_id_col="node_id", directed=False)
    session.scale(columns=["f1", "f2"], method="standard")

    fit = session.fit_graph(
        method="classical",
        mode="inductive",
        classical_estimator="logistic_regression",
        random_state=0,
    )
    assert fit.n_train_nodes > 0
    pred = session.predict_graph(partition="test")
    assert pred.n_nodes > 0
    ev = session.evaluate_graph(partition="validation")
    assert "accuracy" in ev.metrics

    bundle = tmp_path / "graph_bundle"
    session.save_graph_bundle(bundle)
    assert (bundle / "meta.json").is_file()

    walk = session.walkthrough()
    assert walk.graph_status.get("has_graph_plan") is True

    other_nodes, _ = _community_graph()
    other = (
        Session.ingest(other_nodes)
        .set_roles(
            {
                "node_id": "id",
                "f1": "feature",
                "f2": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(columns=["f1", "f2"], method="standard")
    )
    other.load_graph_bundle(bundle, trusted=True)
    assert other.graph_plan is not None
    assert other.evaluate_graph(partition="test").n_nodes > 0
