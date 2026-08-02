"""Graph ML domain (node classification: classical + pure-Torch GCN + PyG).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1–9. Semi-supervised → … → Causal ML — done.
  10. Graph ML / GNNs — **this module** (PASS vs Phase-1 bar).
  Next: **Evolutionary algorithms** (as search/HPO backend — not swarm zoo).
  Later deepenings: NLP/CV if still partial. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite. Knowledge-graph *learning* is a
**separate** Session path (``buildml.kg``) — still not a Neo4j product.

Honesty (this package):
  - Session rows = nodes; edge list attached via ``set_graph``; splits are
    **node** partitions.
  - Three complete paths:
      1. Classical: NetworkX metrics + sklearn classifier (``buildml[graph]``).
      2. Pure-Torch GCN (``buildml[torch]``) — dense adjacency, no PyG.
      3. PyTorch Geometric (``buildml[graph-pyg]``) — GCN / GraphSAGE / GAT.
  - Default ``mode='inductive'``: fit on train-induced subgraph; score may use
    train↔holdout edges; holdout↔holdout dropped. ``transductive`` uses full
    topology with train-label-only supervision (disclosed).
  - Not KG triples/link-prediction (see ``buildml.kg``), not graph-level
    classify zoo, not Neo4j.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn.
  - ``buildml[graph]`` → NetworkX (classical path).
  - ``buildml[torch]`` → Torch (pure-Torch GCN path).
  - ``buildml[graph-pyg]`` → torch-geometric + torch (industry GNN path).

Lazy imports — ``import buildml`` never requires networkx, torch, or pyg.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ClassicalEstimator",
    "GraphConfig",
    "GraphEvalResult",
    "GraphFitResult",
    "GraphMethod",
    "GraphMode",
    "GraphPlan",
    "GraphPredictResult",
    "GraphSpec",
    "GraphTask",
    "PyGModel",
    "evaluate_graph",
    "fit_graph",
    "graph_capability_matrix",
    "graph_status",
    "graph_status_for_session",
    "load_graph_bundle",
    "networkx_available",
    "predict_graph",
    "pyg_available",
    "require_networkx",
    "require_pyg",
    "save_graph_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ClassicalEstimator",
        "GraphConfig",
        "GraphMethod",
        "GraphMode",
        "GraphSpec",
        "GraphTask",
        "PyGModel",
    }:
        from buildml.graph import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "GraphPlan",
        "GraphFitResult",
        "GraphPredictResult",
        "GraphEvalResult",
    }:
        from buildml.graph import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_graph":
        from buildml.graph.fit import fit_graph

        return fit_graph
    if name == "predict_graph":
        from buildml.graph.predict import predict_graph

        return predict_graph
    if name == "evaluate_graph":
        from buildml.graph.evaluate import evaluate_graph

        return evaluate_graph
    if name == "graph_capability_matrix":
        from buildml.graph.catalog import graph_capability_matrix

        return graph_capability_matrix
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_graph_bundle",
        "load_graph_bundle",
    }:
        from buildml.graph import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"graph_status", "graph_status_for_session"}:
        from buildml.graph import explain_hooks as hooks

        return getattr(hooks, name)
    if name in {"require_networkx", "networkx_available", "require_pyg", "pyg_available"}:
        from buildml.graph import extras as extras_mod

        return getattr(extras_mod, name)
    raise AttributeError(f"module 'buildml.graph' has no attribute {name!r}")
