"""Federated learning domain (local FedAvg-style Session simulation).

Behavior and limitations:
  - Local FedAvg-style (and FedProx) orchestration on Session data partitioned
    by a client/group column: **not** a production FL network stack unless you
    deploy one separately.
  - ``backend='flower'`` uses Flower (flwr) NumPyClient + aggregation helpers
    but still runs in-process on Session partitions by default.
  - No cryptographic secure aggregation; the in-process orchestrator sees
    client coefficient updates (privacy limits disclosed).
  - Train-only local updates; validation/test are evaluation-only.
  - Deep path: sklearn linear / SGD coefficient averaging.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Native
simulation uses sklearn façades: no optional extra required for
``import buildml``. ``buildml[federated-industry]`` adds Flower (flwr).

Lazy imports: core never grows heavy FL stacks.
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
    "export_round_history",
    "federated_capability_matrix",
    "federated_status",
    "federated_status_for_session",
    "fit_federated",
    "list_federated_methods",
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
    if name == "federated_capability_matrix":
        from buildml.federated.catalog import federated_capability_matrix

        return federated_capability_matrix
    if name == "list_federated_methods":
        from buildml.federated.catalog import list_federated_methods

        return list_federated_methods
    if name == "export_round_history":
        from buildml.federated.results import export_round_history

        return export_round_history
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
