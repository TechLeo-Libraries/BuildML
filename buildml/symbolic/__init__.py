"""Symbolic + neuro-symbolic tabular ML (rules + sklearn hybrid).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1–9. Semi-supervised → … → causal / graph / evolutionary HPO — prior items.
  **This module (complete):** Symbolic AI + Neuro-symbolic AI.
  Next delivered: Case-based reasoning (`buildml.cbr`), then imitation+RL,
  then TDA, then app systems.

Explicit non-goals (no product surfaces): full expert-system products, fuzzy
logic as a standalone product, Prolog/Z3 required in core, AGI symbolic
reasoners, neuromorphic/SNN, swarm zoo, digital twins, AV stack, multi-agent
world sims, TTS, robotics/control product.

Honesty (this package):
  - Explicit if-then rule knowledge bases over tabular columns.
  - Induction: sklearn DecisionTree path export and sequential-covering
    decision lists (train-only). Declared expert rules also supported.
  - Neuro-symbolic: sklearn base estimator + rule overlay / rules-as-features /
    constraint-repair as a single Session API.
  - **Not** an AGI symbolic reasoner, Prolog engine, or Z3 SMT solver.
  - Core stays light: numpy/pandas/sklearn only — no Prolog/Z3 dependency.

Lazy imports — core never grows heavy logic-programming stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "BaseEstimatorName",
    "NeuroSymbolicConfig",
    "NeuroSymbolicFitResult",
    "NeuroSymbolicMode",
    "NeuroSymbolicPlan",
    "Predicate",
    "Rule",
    "RuleKnowledgeBase",
    "RuleTrace",
    "SymbolicConfig",
    "SymbolicEvalResult",
    "SymbolicFitResult",
    "SymbolicPlan",
    "SymbolicPredictResult",
    "SymbolicSource",
    "SymbolicTask",
    "evaluate_neuro_symbolic",
    "evaluate_symbolic",
    "fit_neuro_symbolic",
    "fit_symbolic",
    "load_symbolic_bundle",
    "predict_neuro_symbolic",
    "predict_symbolic",
    "save_symbolic_bundle",
    "symbolic_status",
    "symbolic_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {
        "SymbolicTask",
        "SymbolicSource",
        "NeuroSymbolicMode",
        "BaseEstimatorName",
        "SymbolicConfig",
        "NeuroSymbolicConfig",
    }:
        from buildml.symbolic import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "SymbolicPlan",
        "NeuroSymbolicPlan",
        "SymbolicFitResult",
        "NeuroSymbolicFitResult",
        "SymbolicEvalResult",
        "SymbolicPredictResult",
    }:
        from buildml.symbolic import results as results_mod

        return getattr(results_mod, name)
    if name in {"Predicate", "Rule", "RuleKnowledgeBase", "RuleTrace"}:
        from buildml.symbolic import rules as rules_mod

        return getattr(rules_mod, name)
    if name == "fit_symbolic":
        from buildml.symbolic.fit import fit_symbolic

        return fit_symbolic
    if name == "fit_neuro_symbolic":
        from buildml.symbolic.fit import fit_neuro_symbolic

        return fit_neuro_symbolic
    if name == "evaluate_symbolic":
        from buildml.symbolic.evaluate import evaluate_symbolic

        return evaluate_symbolic
    if name == "evaluate_neuro_symbolic":
        from buildml.symbolic.evaluate import evaluate_neuro_symbolic

        return evaluate_neuro_symbolic
    if name == "predict_symbolic":
        from buildml.symbolic.predict import predict_symbolic

        return predict_symbolic
    if name == "predict_neuro_symbolic":
        from buildml.symbolic.predict import predict_neuro_symbolic

        return predict_neuro_symbolic
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_symbolic_bundle",
        "load_symbolic_bundle",
    }:
        from buildml.symbolic import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"symbolic_status", "symbolic_status_for_session"}:
        from buildml.symbolic import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.symbolic' has no attribute {name!r}")
