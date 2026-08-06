"""Classical NetworkX-style node featurization for Graph ML."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.graph.extras import require_networkx
from buildml.graph.types import GraphMode


def compute_graph_metrics(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    directed: bool,
    mode: GraphMode,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute per-node classical graph metrics under leakage rules.

    Callers pass an already mode-filtered edge set (train-induced at fit;
    train↔holdout at inductive score; full graph when transductive).

    Parameters
    ----------
    n_nodes:
        Number of nodes (Session rows).
    src, dst:
        Edge endpoint row indices after mode filtering.
    directed:
        When False, build an undirected NetworkX graph.
    mode:
        ``inductive`` or ``transductive`` graph-learning mode.
    train_mask:
        Reserved for future mask-aware disclosures.

    Returns
    -------
    features:
        Array of shape ``(n_nodes, n_metrics)``.
    feature_names:
        Metric column names.
    disclosures:
        Honesty / leakage notes.
    """
    nx = require_networkx(feature="Classical graph-feature node classification")

    # Callers pass an already mode-filtered edge set (train-induced at fit;
    # train↔holdout at inductive score; full graph when transductive).
    disclosures: list[str] = [
        f"Classical graph metrics via NetworkX (mode={mode}).",
    ]
    if mode == "inductive":
        disclosures.append(
            "Inductive classical path: metrics computed on the provided "
            "edge set (train-induced at fit; train↔holdout at score)."
        )
    else:
        disclosures.append(
            "Transductive classical path: metrics use the full topology; "
            "holdout structure participates in centrality / clustering."
        )

    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    if len(src):
        g.add_edges_from(zip(src.tolist(), dst.tolist(), strict=True))

    # Cheap, stable metrics suitable for small Session graphs.
    degree = dict(g.degree())
    if directed:
        clustering = {n: 0.0 for n in g.nodes()}
        disclosures.append(
            "Directed graphs: clustering coefficient set to 0 (NetworkX "
            "clustering is undirected-oriented in this surface)."
        )
    else:
        clustering = nx.clustering(g)

    # PageRank / eigenvector can fail on empty graphs; fall back to zeros.
    try:
        pagerank = nx.pagerank(g, alpha=0.85, max_iter=100)
    except Exception:
        pagerank = {n: 0.0 for n in g.nodes()}
        disclosures.append("PageRank failed to converge; filled zeros.")

    try:
        if directed:
            avg_nei = {n: 0.0 for n in g.nodes()}
        else:
            avg_nei = nx.average_neighbor_degree(g)
    except Exception:
        avg_nei = {n: 0.0 for n in g.nodes()}

    # Betweenness is O(n^3)-ish; only for tiny graphs.
    betweenness: dict[Any, float]
    if n_nodes <= 200 and g.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(g, normalized=True)
        disclosures.append("Included betweenness_centrality (n_nodes <= 200).")
    else:
        betweenness = {n: 0.0 for n in g.nodes()}
        disclosures.append(
            "Skipped betweenness_centrality (n_nodes > 200 or no edges)."
        )

    names = [
        "graph_degree",
        "graph_clustering",
        "graph_pagerank",
        "graph_avg_neighbor_degree",
        "graph_betweenness",
    ]
    feats = np.zeros((n_nodes, len(names)), dtype=np.float64)
    for i in range(n_nodes):
        feats[i, 0] = float(degree.get(i, 0))
        feats[i, 1] = float(clustering.get(i, 0.0))
        feats[i, 2] = float(pagerank.get(i, 0.0))
        feats[i, 3] = float(avg_nei.get(i, 0.0))
        feats[i, 4] = float(betweenness.get(i, 0.0))

    # Nodes isolated under inductive filtering get zero metrics: disclose.
    isolated = int((feats[:, 0] == 0).sum())
    if isolated and mode == "inductive":
        disclosures.append(
            f"{isolated} node(s) have degree 0 under the inductive edge filter "
            "(common for holdout nodes with no train edge)."
        )
    _ = train_mask  # reserved for future mask-aware disclosures
    return feats, names, disclosures


def build_classical_design(
    tabular: np.ndarray,
    tabular_names: list[str],
    graph_feats: np.ndarray | None,
    graph_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate tabular and graph-metric features for classical Graph ML.

    At least one non-empty block is required so sklearn receives a design
    matrix with one or more columns.

    Parameters
    ----------
    tabular:
        Numeric tabular node features, or empty array.
    tabular_names:
        Column names aligned to ``tabular`` columns.
    graph_feats:
        Optional NetworkX-derived metric matrix.
    graph_names:
        Column names aligned to ``graph_feats`` columns.

    Returns
    -------
    X:
        Horizontally stacked design matrix.
    names:
        Combined feature names in column order.

    Raises
    ------
    ValidationError
        When both tabular and graph blocks are empty.
    """
    parts: list[np.ndarray] = []
    names: list[str] = []
    if tabular is not None and tabular.size and tabular.shape[1] > 0:
        parts.append(tabular)
        names.extend(tabular_names)
    if graph_feats is not None and graph_feats.size and graph_feats.shape[1] > 0:
        parts.append(graph_feats)
        names.extend(graph_names or [])
    if not parts:
        raise ValidationError(
            "Classical Graph ML needs tabular node features and/or graph "
            "metrics. Provide numeric features or set include_graph_metrics=True."
        )
    return np.hstack(parts), names


def design_frame(
    X: np.ndarray,
    names: list[str],
) -> pd.DataFrame:
    """Wrap a design matrix as a DataFrame for sklearn pipelines.

    Preserves feature names so classical estimators and history summaries stay
    aligned with the concatenated tabular + graph-metric columns.

    Parameters
    ----------
    X:
        Numeric design matrix of shape ``(n_nodes, n_features)``.
    names:
        Column names aligned to ``X`` columns.

    Returns
    -------
    pandas.DataFrame
        Feature frame suitable for sklearn estimators.
    """
    return pd.DataFrame(X, columns=names)
