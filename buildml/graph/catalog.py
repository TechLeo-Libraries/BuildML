"""Graph ML backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_available, torch_spec_available
from buildml.graph.extras import (
    networkx_available,
    networkx_spec_present,
    pyg_runtime_available,
    pyg_spec_present,
)

GraphBackendName = Literal["classical", "gcn", "pyg"]


def graph_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for graph backends and optional extras.

    Reports install availability, supported methods/modes, message-passing
    semantics, and non-goals so walkthrough panels disclose Graph ML limits
    without overstating PyG or classical coverage.

    Returns
    -------
    dict[str, Any]
        Backend availability, install hints, default backend selection, and
        honesty notes separating Graph ML from KG / Neo4j products.
    """
    return {
        "backends": {
            "classical": {
                "available": networkx_available(),
                "extra": "graph",
                "methods": ["logistic_regression", "random_forest"],
                "modes": ["inductive", "transductive"],
                "message_passing": "NetworkX topology metrics + tabular concat",
                "notes": (
                    "Requires buildml[graph] (NetworkX). Degree, clustering, "
                    "PageRank, avg neighbor degree; betweenness when n≤200."
                ),
            },
            "gcn": {
                "available": torch_available(),
                "extra": "torch",
                "methods": ["gcn"],
                "modes": ["inductive", "transductive"],
                "message_passing": "Pure-Torch Kipf–Welling GCN (dense adjacency)",
                "notes": (
                    "No PyTorch Geometric. Symmetric normalized adjacency with "
                    "self-loops; train-mask cross-entropy only. Dense guard ≤5000 nodes."
                ),
            },
            "pyg": {
                "available": pyg_runtime_available(),
                "extra": "graph-pyg",
                "methods": ["gcn", "graphsage", "gat"],
                "modes": ["inductive", "transductive"],
                "message_passing": "PyTorch Geometric conv layers (sparse edge_index)",
                "notes": (
                    "Requires buildml[graph-pyg] (torch-geometric + torch). "
                    "GCNConv, SAGEConv, GATConv with train-label-only loss. "
                    "Same inductive/transductive edge filters as classical/gcn."
                ),
            },
        },
        "tasks": ["node_classification"],
        "evaluation_metrics": ["accuracy", "f1_macro", "roc_auc"],
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_pyg_model_when_installed": "gcn",
        "install_hints": {
            "graph": (
                "pip install 'buildml[graph]'  "
                "# NetworkX classical node features + sklearn"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# pure-Torch GCN (no PyG)"
            ),
            "graph-pyg": (
                "pip install 'buildml[graph-pyg]'  "
                "# PyTorch Geometric GCN / GraphSAGE / GAT"
            ),
        },
        "non_goals": [
            "Neo4j / graph-database product",
            "Knowledge-graph link prediction (see buildml.kg)",
            "Graph-level classification zoo",
            "Full PyG research algorithm catalog beyond GCN/SAGE/GAT",
            "Link prediction product depth on this surface",
        ],
        "graph_extra_present": networkx_spec_present(),
        "graph_runtime_present": networkx_available(),
        "torch_spec_present": torch_spec_available(),
        "pyg_extra_present": pyg_spec_present(),
        "pyg_runtime_present": pyg_runtime_available(),
        "pyg_import_honesty": (
            "pyg backend 'available' requires torch-geometric install AND a working "
            "torch import (pyg_runtime_available). pyg_extra_present / "
            "graph_extra_present / *_spec_present are find_spec only; "
            "graph_runtime_present / pyg_runtime_present are import probes. "
            "pyg_available() is an alias of pyg_spec_present for back-compat."
        ),
        "train_only_honesty": (
            "All backends fit with train-node labels only. Inductive fit uses "
            "train–train edges; transductive uses full topology with disclosed "
            "train-label-only supervision."
        ),
    }


def _default_backend_when_installed() -> str:
    if pyg_runtime_available():
        return "pyg"
    if torch_available():
        return "gcn"
    if networkx_available():
        return "classical"
    return "classical"
