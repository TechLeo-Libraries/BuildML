"""Native ensemble learning (voting / stacking / holdout blending).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**):
  1. Unsupervised learning — clustering Session path (done; see
     ``buildml.unsupervised``).
  2. Ensemble learning — native stacking/voting/blending. **This module.**
  3. AutoML — pipeline/model search beyond HPO (done; see ``buildml.automl``).
  4. Time-series forecasting — done (see ``buildml.forecasting``).
  5. Anomaly / fraud detection — done (see ``buildml.anomaly``).

Later phases and explicit non-goals: see ``buildml.unsupervised`` module doc.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Ensembles use
core sklearn Voting*/Stacking* plus an honest holdout-blend estimator —
no optional extra required for ``import buildml``.

Lazy imports — core never grows heavy ensemble stacks beyond sklearn.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "BlendMethod",
    "EnsembleConfig",
    "EnsembleFitResult",
    "EnsemblePlan",
    "EnsembleStrategy",
    "VotingMethod",
    "fit_blending_ensemble",
    "fit_stacking_ensemble",
    "fit_voting_ensemble",
    "load_ensemble_bundle",
    "save_ensemble_bundle",
    "ensemble_status",
    "ensemble_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {"EnsembleStrategy", "VotingMethod", "BlendMethod", "EnsembleConfig"}:
        from buildml.ensemble import types as types_mod

        return getattr(types_mod, name)
    if name in {"EnsemblePlan", "EnsembleFitResult"}:
        from buildml.ensemble import results as results_mod

        return getattr(results_mod, name)
    if name in {
        "fit_voting_ensemble",
        "fit_stacking_ensemble",
        "fit_blending_ensemble",
    }:
        from buildml.ensemble import fit as fit_mod

        return getattr(fit_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_ensemble_bundle",
        "load_ensemble_bundle",
    }:
        from buildml.ensemble import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"ensemble_status", "ensemble_status_for_session"}:
        from buildml.ensemble import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.ensemble' has no attribute {name!r}")
