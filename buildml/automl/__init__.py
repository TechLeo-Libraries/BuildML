"""AutoML: model-family + fold-local recipe search beyond single-estimator HPO.

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**):
  1. Unsupervised learning — clustering Session path (done; see
     ``buildml.unsupervised``).
  2. Ensemble learning — native stacking/voting/blending (done; see
     ``buildml.ensemble``).
  3. AutoML — pipeline/model search beyond HPO. **This module.**
  4. Time-series forecasting — done (see ``buildml.forecasting``).
  5. Anomaly / fraud detection — done (see ``buildml.anomaly``).

Later phases and explicit non-goals: see ``buildml.unsupervised`` module doc.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Randomized/grid
AutoML needs no optional extra. Optuna-backed AutoML uses ``buildml[automl]``.
Industry adapters (FLAML / AutoGluon) and GBDT families use
``buildml[automl-industry]``.

Lazy imports — core never grows heavy AutoML stacks beyond sklearn (+ optional
Optuna / industry adapters when explicitly requested).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "AutoMLBackend",
    "AutoMLBudget",
    "AutoMLConfig",
    "AutoMLMethod",
    "AutoMLPlan",
    "AutoMLResult",
    "AutoMLSelection",
    "AutoMLTrial",
    "CandidateKind",
    "EnsembleMode",
    "automl_capability_matrix",
    "export_comparison_metrics",
    "list_automl_methods",
    "load_automl_bundle",
    "run_automl",
    "save_automl_bundle",
    "automl_status",
    "automl_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {
        "AutoMLBudget",
        "AutoMLConfig",
        "AutoMLMethod",
        "AutoMLBackend",
        "AutoMLSelection",
        "CandidateKind",
        "EnsembleMode",
    }:
        from buildml.automl import types as types_mod

        return getattr(types_mod, name)
    if name in {"AutoMLPlan", "AutoMLResult", "AutoMLTrial"}:
        from buildml.automl import results as results_mod

        return getattr(results_mod, name)
    if name == "run_automl":
        from buildml.automl.search import run_automl

        return run_automl
    if name == "export_comparison_metrics":
        from buildml.automl.search import export_comparison_metrics

        return export_comparison_metrics
    if name in {"automl_capability_matrix", "list_automl_methods"}:
        from buildml.automl import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_automl_bundle",
        "load_automl_bundle",
    }:
        from buildml.automl import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"automl_status", "automl_status_for_session"}:
        from buildml.automl import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.automl' has no attribute {name!r}")
