"""Federated learning domain (local FedAvg-style Session simulation).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1. Semi-supervised learning — done (``buildml.semisupervised``).
  2. Self-supervised learning hooks — done (``buildml.selfsupervised``).
  3. Active learning — done (``buildml.activelearning``).
  4. Online / continual (partial_fit) — done (``buildml.online``).
  5. Multi-task learning — done (``buildml.multitask``).
  6. Meta-learning — done (``buildml.metalearning``).
  7. Federated learning — **this module**.
  8. Bayesian / probabilistic — done (``buildml.probabilistic``); next = Causal.
  Later: graph, evolutionary,
  symbolic, CBR, IL+RL, TDA, recommenders / LTR / KG / optimisation / synthetic /
  NLP-CV deepenings. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Honesty (this package):
  - Local FedAvg-style (and FedProx) orchestration on Session data partitioned
    by a client/group column — **not** a production FL network stack.
  - Not a Flower / OpenFL replacement.
  - No cryptographic secure aggregation; the in-process orchestrator sees
    client coefficient updates (privacy limits disclosed).
  - Train-only local updates; validation/test are evaluation-only.
  - Deep path: sklearn linear / SGD coefficient averaging.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Federated
simulation uses sklearn façades — no optional extra required for
``import buildml``.

Lazy imports — core never grows heavy FL stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "FederatedConfig",
    "FederatedEstimator",
    "FederatedEvalResult",
    "FederatedFitResult",
    "FederatedMethod",
    "FederatedPlan",
    "FederatedPredictResult",
    "FederatedTask",
    "evaluate_federated",
    "federated_status",
    "federated_status_for_session",
    "fit_federated",
    "load_federated_bundle",
    "predict_federated",
    "save_federated_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "FederatedMethod",
        "FederatedEstimator",
        "FederatedTask",
        "FederatedConfig",
    }:
        from buildml.federated import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "FederatedPlan",
        "FederatedFitResult",
        "FederatedEvalResult",
        "FederatedPredictResult",
    }:
        from buildml.federated import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_federated":
        from buildml.federated.fit import fit_federated

        return fit_federated
    if name == "evaluate_federated":
        from buildml.federated.evaluate import evaluate_federated

        return evaluate_federated
    if name == "predict_federated":
        from buildml.federated.predict import predict_federated

        return predict_federated
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_federated_bundle",
        "load_federated_bundle",
    }:
        from buildml.federated import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"federated_status", "federated_status_for_session"}:
        from buildml.federated import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.federated' has no attribute {name!r}")
