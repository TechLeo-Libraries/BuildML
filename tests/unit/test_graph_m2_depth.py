"""M2 depth coverage for graph low-level + Session APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import importlib.util

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.explain.sync import REQUIRED_AI_TOOL_SESSION_METHODS
from buildml.graph.extras import networkx_available, pyg_available

_TORCH_SPEC = importlib.util.find_spec("torch") is not None
_PYG_SPEC = pyg_available()


def _community_graph(n_per: int = 40, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-community graph with assortative edges + node features."""
    rng = np.random.default_rng(seed)
    n = n_per * 2
    labels = np.array([0] * n_per + [1] * n_per)
    # Features: community signal + noise
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
    # Dense within-community, sparse between
    for c, start in enumerate((0, n_per)):
        members = list(range(start, start + n_per))
        for i in members:
            for j in members:
                if i < j and rng.random() < 0.18:
                    edges.append((i, j))
    for i in range(n_per):
        for j in range(n_per, n):
            if rng.random() < 0.02:
                edges.append((i, j))
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    return nodes, edges_df


def _session() -> Session:
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
    # Snapshot node ids before scale; scale features only (id stays stable).
    session.set_graph(edges, node_id_col="node_id", directed=False)
    session.scale(columns=["f1", "f2"], method="standard")
    return session


@pytest.mark.skipif(not networkx_available(), reason="buildml[graph] / networkx missing")
def test_classical_inductive_fit_eval_bundle(tmp_path) -> None:
    session = _session()
    fit = session.fit_graph(
        method="classical",
        mode="inductive",
        classical_estimator="logistic_regression",
        random_state=0,
    )
    assert fit.method == "classical"
    assert fit.mode == "inductive"
    assert fit.n_train_nodes > 0
    assert fit.train_accuracy is not None and fit.train_accuracy > 0.55
    ev = session.evaluate_graph(partition="validation")
    assert "accuracy" in ev.metrics
    assert ev.metrics["accuracy"] >= 0.5
    pred = session.predict_graph(partition="test")
    assert pred.n_nodes > 0
    out = tmp_path / "graph_bundle"
    session.save_graph_bundle(out)
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
    assert other.graph_spec is not None
    ev2 = other.evaluate_graph(partition="test")
    assert "f1_macro" in ev2.metrics


@pytest.mark.skipif(not networkx_available(), reason="buildml[graph] / networkx missing")
def test_transductive_classical_path() -> None:
    session = _session()
    fit = session.fit_graph(method="classical", mode="transductive", random_state=1)
    assert fit.mode == "transductive"
    assert any("Transductive" in d or "transductive" in d for d in fit.disclosures)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
@pytest.mark.skipif(
    __import__("sys").platform.startswith("win"),
    reason="torch DLL import can AV on some Windows Python 3.13 setups",
)
def test_gcn_inductive_path() -> None:
    session = _session()
    fit = session.fit_graph(
        method="gcn",
        mode="inductive",
        epochs=40,
        hidden_dim=16,
        random_state=0,
    )
    assert fit.method == "gcn"
    assert fit.train_accuracy is not None and fit.train_accuracy > 0.5
    ev = session.evaluate_graph(partition="test")
    assert ev.metrics["accuracy"] >= 0.45


@pytest.mark.skipif(not _PYG_SPEC, reason="buildml[graph-pyg] / torch-geometric missing")
@pytest.mark.skipif(
    __import__("sys").platform.startswith("win"),
    reason="torch/PyG DLL import can AV on some Windows Python 3.13 setups",
)
def test_pyg_graphsage_inductive_path() -> None:
    session = _session()
    fit = session.fit_graph(
        method="pyg",
        pyg_model="graphsage",
        mode="inductive",
        epochs=40,
        hidden_dim=16,
        random_state=0,
    )
    assert fit.method == "pyg"
    assert fit.train_accuracy is not None and fit.train_accuracy > 0.5
    ev = session.evaluate_graph(partition="test")
    assert ev.metrics["accuracy"] >= 0.45


@pytest.mark.skipif(not _PYG_SPEC, reason="buildml[graph-pyg] / torch-geometric missing")
@pytest.mark.skipif(
    __import__("sys").platform.startswith("win"),
    reason="torch/PyG DLL import can AV on some Windows Python 3.13 setups",
)
def test_pyg_gat_inductive_path() -> None:
    session = _session()
    fit = session.fit_graph(
        method="pyg",
        pyg_model="gat",
        mode="inductive",
        epochs=40,
        hidden_dim=16,
        heads=2,
        random_state=0,
    )
    assert fit.method == "pyg"
    assert fit.train_accuracy is not None and fit.train_accuracy > 0.5


def test_graph_capability_matrix() -> None:
    from buildml.graph.catalog import graph_capability_matrix

    matrix = graph_capability_matrix()
    assert "classical" in matrix["backends"]
    assert "gcn" in matrix["backends"]
    assert "pyg" in matrix["backends"]
    assert matrix["backends"]["pyg"]["methods"] == ["gcn", "graphsage", "gat"]


def test_refuse_fit_without_set_graph() -> None:
    nodes, _ = _community_graph()
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
        .split(test_size=0.2, validation_size=0.2, random_state=0)
    )
    with pytest.raises(ValidationError, match="No GraphSpec"):
        session.fit_graph(method="classical")


def test_refuse_duplicate_node_ids() -> None:
    nodes, edges = _community_graph()
    nodes.loc[1, "node_id"] = nodes.loc[0, "node_id"]
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
        .split(test_size=0.2, random_state=0)
    )
    with pytest.raises(ValidationError, match="unique"):
        session.set_graph(edges, node_id_col="node_id")


def test_ai_allowlist_includes_graph() -> None:
    for name in ("set_graph", "fit_graph", "evaluate_graph", "predict_graph"):
        assert name in REQUIRED_AI_TOOL_SESSION_METHODS


def test_import_buildml_without_forcing_networkx() -> None:
    """Core import path must not require networkx at import time."""
    import buildml
    import buildml.graph as graph_mod

    assert buildml.__version__
    # Accessing classical fit without networkx should raise MissingExtraError
    # only when networkx is absent; when present, just ensure lazy attr works.
    assert hasattr(graph_mod, "fit_graph")
    if not networkx_available():
        from buildml.graph.extras import require_networkx

        with pytest.raises(MissingExtraError):
            require_networkx()
