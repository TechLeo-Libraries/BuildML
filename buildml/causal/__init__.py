"""Causal ML domain (assumption-declared backdoor ATE).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1–8. Semi-supervised → … → Bayesian / probabilistic — done.
  9. Causal ML — **this module** (PASS vs Phase-1 bar).
  10. Graph ML / GNNs — see ``buildml.graph``.
  Next: **Evolutionary algorithms** (search/HPO backend — not swarm zoo).
  Later: symbolic, CBR, IL+RL, TDA, recommenders / LTR / KG / optimisation /
  synthetic / NLP-CV deepenings. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Honesty (this package):
  - Requires an explicit ``CausalAssumptions`` object (treatment, outcome,
    confounders, estimand=ATE, backdoor identification, and acknowledgements
    of unconfoundedness + positivity). Estimation **refuses** without it.
  - EDA / association / feature-importance paths remain associational and
    never populate or satisfy these assumptions.
  - Native sklearn nuisance models for T-learner, IPW, and AIPW ATE with
    train-only fit and optional bootstrap uncertainty.
  - Optional simple placebo / random-confounder sensitivity disclosures —
    **not** a full DoWhy refutation suite.
  - **Not** causal discovery, **not** IV / front-door (instruments refused
    until an IV path exists), **not** a DoWhy/EconML required dependency.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. No
``buildml[causal]`` extra is required for ``import buildml``. Optional
DoWhy/EconML was considered and skipped so the Session path stays honest
and dependency-light for this depth.

Lazy imports — core never grows heavy causal stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "CausalAssumptions",
    "CausalConfig",
    "CausalEstimateResult",
    "CausalEstimand",
    "CausalEvalResult",
    "CausalFitResult",
    "CausalIdentification",
    "CausalMethod",
    "CausalPlan",
    "CausalRefuteKind",
    "CausalRefuteResult",
    "declare_causal_assumptions",
    "estimate_causal",
    "evaluate_causal",
    "fit_causal",
    "load_causal_bundle",
    "refute_causal",
    "save_causal_bundle",
    "causal_status",
    "causal_status_for_session",
]


def declare_causal_assumptions(**kwargs: Any) -> Any:
    """Validate and return a :class:`CausalAssumptions` instance."""
    from buildml.causal.types import CausalAssumptions

    assumptions = CausalAssumptions.from_mapping(kwargs)
    assumptions.validate()
    return assumptions


def __getattr__(name: str) -> Any:
    if name in {
        "CausalAssumptions",
        "CausalConfig",
        "CausalEstimand",
        "CausalIdentification",
        "CausalMethod",
        "CausalRefuteKind",
    }:
        from buildml.causal import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "CausalPlan",
        "CausalFitResult",
        "CausalEstimateResult",
        "CausalEvalResult",
        "CausalRefuteResult",
    }:
        from buildml.causal import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_causal":
        from buildml.causal.fit import fit_causal

        return fit_causal
    if name == "estimate_causal":
        from buildml.causal.estimate import estimate_causal

        return estimate_causal
    if name == "evaluate_causal":
        from buildml.causal.evaluate import evaluate_causal

        return evaluate_causal
    if name == "refute_causal":
        from buildml.causal.refute import refute_causal

        return refute_causal
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_causal_bundle",
        "load_causal_bundle",
    }:
        from buildml.causal import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"causal_status", "causal_status_for_session"}:
        from buildml.causal import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.causal' has no attribute {name!r}")
