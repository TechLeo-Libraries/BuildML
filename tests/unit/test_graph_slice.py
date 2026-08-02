"""Session-facing slice tests for Graph ML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.graph.extras import networkx_available


def _community_graph(n_per: int = 36, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_per * 2
    labels = np.array([0] * n_per + [1] * n_per)
    x = labels.astype(float).reshape(-1, 1) + rng.normal(scale=0.4, size=(n, 2))
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
                if i < j and rng.random() < 0.2:
                    edges.append((i, j))
    for i in range(n_per):
        for j in range(n_per, n):
            if rng.random() < 0.03:
                edges.append((i, j))
    return nodes, pd.DataFrame(edges, columns=["source", "target"])


def _demo_session() -> Session:
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
    return session


def test_core_import_and_catalog() -> None:
    import buildml.graph as graph_mod

    assert hasattr(graph_mod, "fit_graph")
    assert hasattr(Session, "fit_graph")
    for op in (
        "set_graph",
        "fit_graph",
        "predict_graph",
        "evaluate_graph",
        "save_graph_bundle",
        "load_graph_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "graph-data-model" in OPERATION_CATALOG["set_graph"].concept_links
    assert "graph-classical-features" in OPERATION_CATALOG["fit_graph"].concept_links
    assert "graph-bundle-boundary" in OPERATION_CATALOG["save_graph_bundle"].concept_links

    registry = build_default_registry()
    for name in ("set_graph", "fit_graph", "evaluate_graph", "predict_graph"):
        assert name in registry


def test_fit_requires_split() -> None:
    nodes, edges = _community_graph(n_per=20)
    session = Session.ingest(nodes).set_roles(
        {
            "node_id": "id",
            "f1": "feature",
            "f2": "feature",
            "y": "target",
        }
    )
    session.set_graph(edges, node_id_col="node_id")
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_graph(method="classical")


@pytest.mark.skipif(not networkx_available(), reason="buildml[graph] / networkx missing")
def test_session_fit_predict_eval_bundle(tmp_path: Path) -> None:
    session = _demo_session()
    fit = session.fit_graph(
        method="classical",
        mode="inductive",
        classical_estimator="logistic_regression",
        random_state=0,
    )
    assert session.graph_plan is not None
    assert fit.n_train_nodes > 0
    pred = session.predict_graph(partition="test")
    assert pred.n_nodes > 0
    ev = session.evaluate_graph(partition="validation")
    assert "accuracy" in ev.metrics

    out = tmp_path / "graph_bundle"
    session.save_graph_bundle(out)
    assert (out / "meta.json").is_file()

    other_nodes, other_edges = _community_graph()
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
    other.load_graph_bundle(out)
    assert other.graph_plan is not None
    assert "accuracy" in other.evaluate_graph(partition="test").metrics
