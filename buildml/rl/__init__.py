"""Imitation learning + reinforcement learning (Session-shaped, honest scope).

Phase coverage (internal tracker: depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  … through IL+RL: **this module (PASS)**.
  Next: Topological Data Analysis (TDA) → then application systems
  (recommenders first).

Honesty (this package):
  - Behavioral cloning from demonstration tables (state → action) on **train only**.
  - Contextual bandits from logged (context, action, reward) tables on **train only**;
    holdout metrics are **offline** (DM / IPS) and disclosed as such.
  - Optional Gymnasium env loops behind ``buildml[rl]``: REINFORCE-lite linear
    softmax (policy gradient) and tabular TD control: Q-learning / SARSA /
    Expected SARSA / Double Q-learning: with explicit state discretization.
  - **Not** a MuJoCo / robotics / AV / multi-agent world-sim platform.
  - Core stays numpy/pandas/sklearn: ``gymnasium`` is optional and never required
    for ``import buildml`` or BC / bandit paths.

Lazy imports: keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT_IMITATION",
    "BUNDLE_FORMAT_RL",
    "CHECKPOINT_BOUNDARY",
    "ImitationConfig",
    "ImitationEstimator",
    "ImitationEvalResult",
    "ImitationFitResult",
    "ImitationPlan",
    "ImitationPredictResult",
    "ImitationTask",
    "RlActResult",
    "RlConfig",
    "RlEvalResult",
    "RlFitResult",
    "RlMode",
    "RlPlan",
    "BanditAlgorithm",
    "TabularAlgorithm",
    "TabularValuePolicy",
    "ObservationDiscretizer",
    "TABULAR_ALGORITHMS",
    "act_rl",
    "evaluate_imitation",
    "evaluate_rl",
    "fit_imitation",
    "fit_rl",
    "rl_capability_matrix",
    "list_imitation_methods",
    "list_rl_algorithms",
    "rl_industry_available",
    "stable_baselines3_available",
    "imitation_status",
    "imitation_status_for_session",
    "load_imitation_bundle",
    "load_rl_bundle",
    "predict_imitation_action",
    "require_gymnasium",
    "gymnasium_available",
    "rl_status",
    "rl_status_for_session",
    "save_imitation_bundle",
    "save_rl_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ImitationTask",
        "ImitationEstimator",
        "ImitationConfig",
        "RlMode",
        "BanditAlgorithm",
        "TabularAlgorithm",
        "RlConfig",
    }:
        from buildml.rl import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "TabularValuePolicy",
        "ObservationDiscretizer",
        "TABULAR_ALGORITHMS",
    }:
        from buildml.rl import tabular as tabular_mod

        return getattr(tabular_mod, name)
    if name in {
        "ImitationPlan",
        "ImitationFitResult",
        "ImitationEvalResult",
        "ImitationPredictResult",
        "RlPlan",
        "RlFitResult",
        "RlEvalResult",
        "RlActResult",
    }:
        from buildml.rl import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_imitation":
        from buildml.rl.imitation import fit_imitation

        return fit_imitation
    if name == "predict_imitation_action":
        from buildml.rl.imitation import predict_imitation_action

        return predict_imitation_action
    if name == "evaluate_imitation":
        from buildml.rl.imitation import evaluate_imitation

        return evaluate_imitation
    if name == "fit_rl":
        from buildml.rl.fit import fit_rl

        return fit_rl
    if name == "act_rl":
        from buildml.rl.act import act_rl

        return act_rl
    if name == "evaluate_rl":
        from buildml.rl.evaluate import evaluate_rl

        return evaluate_rl
    if name in {
        "BUNDLE_FORMAT_IMITATION",
        "BUNDLE_FORMAT_RL",
        "CHECKPOINT_BOUNDARY",
        "save_imitation_bundle",
        "load_imitation_bundle",
        "save_rl_bundle",
        "load_rl_bundle",
    }:
        from buildml.rl import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"require_gymnasium", "gymnasium_available", "rl_industry_available", "stable_baselines3_available"}:
        from buildml.rl import extras as extras_mod

        return getattr(extras_mod, name)
    if name in {
        "rl_capability_matrix",
        "list_imitation_methods",
        "list_rl_algorithms",
    }:
        from buildml.rl import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "imitation_status",
        "imitation_status_for_session",
        "rl_status",
        "rl_status_for_session",
    }:
        from buildml.rl import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.rl' has no attribute {name!r}")
