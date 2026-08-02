"""Unsupervised learning domain (clustering + train-fit / holdout-assign path).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**):
  1. Unsupervised learning — clustering Session path integrating with
     ``Session.reduce_dimensions`` (PCA). **This module.**
  2. Ensemble learning — native stacking/voting/blending (see
     ``buildml.ensemble``).
  3. AutoML — pipeline/model search beyond HPO (done; see ``buildml.automl``).
  4. Time-series forecasting — done (see ``buildml.forecasting``).
  5. Anomaly / fraud detection — done (see ``buildml.anomaly``).

Phase 2 progress:
  1. Semi-supervised — done (``buildml.semisupervised``).
  2. Self-supervised hooks — done (``buildml.selfsupervised``).
  3. Active learning — done (``buildml.activelearning``).
  4. Online / continual — done (``buildml.online``); next = multi-task.
  Later: graph (causal done in ``buildml.causal``; probabilistic in ``buildml.probabilistic``)
  (EDA stays associational), graph, evolutionary, symbolic, CBR, IL+RL, TDA,
  recommenders / LTR / KG / optimisation / synthetic / NLP-CV deepenings.
  Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Clustering uses
core sklearn — no optional extra required for ``import buildml``.

Lazy imports — core never grows heavy unsupervised stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "AssignStrategy",
    "ClusterAssignResult",
    "ClusterConfig",
    "ClusterEvalResult",
    "ClusterFitResult",
    "ClusterMethod",
    "ClusterPlan",
    "assign_clusters",
    "evaluate_clustering",
    "fit_clusterer",
    "load_unsupervised_bundle",
    "save_unsupervised_bundle",
    "unsupervised_status",
    "unsupervised_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {"ClusterMethod", "AssignStrategy", "ClusterConfig"}:
        from buildml.unsupervised import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ClusterPlan",
        "ClusterFitResult",
        "ClusterAssignResult",
        "ClusterEvalResult",
    }:
        from buildml.unsupervised import results as results_mod

        return getattr(results_mod, name)
    if name in {"fit_clusterer", "assign_clusters"}:
        from buildml.unsupervised import cluster as cluster_mod

        return getattr(cluster_mod, name)
    if name == "evaluate_clustering":
        from buildml.unsupervised.evaluate import evaluate_clustering

        return evaluate_clustering
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_unsupervised_bundle",
        "load_unsupervised_bundle",
    }:
        from buildml.unsupervised import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"unsupervised_status", "unsupervised_status_for_session"}:
        from buildml.unsupervised import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.unsupervised' has no attribute {name!r}")
