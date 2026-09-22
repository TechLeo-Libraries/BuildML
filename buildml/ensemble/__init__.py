"""Native ensemble learning (voting / stacking / holdout blending).

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Ensembles use
core sklearn Voting*/Stacking* plus an honest holdout-blend estimator :
no optional extra required for ``import buildml``.

Lazy imports: core never grows heavy ensemble stacks beyond sklearn.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "BaseLearnerContribution",
    "BlendMethod",
    "EnsembleConfig",
    "EnsembleDiversitySummary",
    "EnsembleEvalReport",
    "EnsembleFitResult",
    "EnsemblePlan",
    "EnsembleStrategy",
    "VotingMethod",
    "build_ensemble_eval_report",
    "ensemble_capability_matrix",
    "ensemble_status_payload",
    "fit_blending_ensemble",
    "fit_stacking_ensemble",
    "fit_voting_ensemble",
    "load_ensemble_bundle",
    "save_ensemble_bundle",
    "ensemble_status",
    "ensemble_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {"ensemble_capability_matrix", "ensemble_status_payload"}:
        from buildml.ensemble import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {"EnsembleStrategy", "VotingMethod", "BlendMethod", "EnsembleConfig"}:
        from buildml.ensemble import types as types_mod

        return getattr(types_mod, name)
    if name in {"EnsemblePlan", "EnsembleFitResult"}:
        from buildml.ensemble import results as results_mod

        return getattr(results_mod, name)
    if name in {
        "BaseLearnerContribution",
        "EnsembleDiversitySummary",
        "EnsembleEvalReport",
        "build_ensemble_eval_report",
    }:
        from buildml.ensemble import evaluate as evaluate_mod

        return getattr(evaluate_mod, name)
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
