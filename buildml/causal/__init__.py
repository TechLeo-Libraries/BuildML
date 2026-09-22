"""Causal ML domain (assumption-declared backdoor ATE).

Behavior and limitations:
  - Requires an explicit ``CausalAssumptions`` object (treatment, outcome,
    confounders, estimand=ATE, backdoor identification, and acknowledgements
    of unconfoundedness + positivity). Estimation **refuses** without it.
  - EDA / association / feature-importance paths remain associational and
    never populate or satisfy these assumptions.
  - Native sklearn nuisance models for T-learner, IPW, and AIPW ATE with
    train-only fit and optional bootstrap uncertainty.
  - Optional DoWhy (``buildml[causal-industry]``): causal graph from declared
    confounders, identification, industry refutation suite.
  - Optional EconML (``buildml[causal-industry]``): DML, CausalForestDML,
    PolicyTree on declared backdoor sets.
  - **Not** causal discovery, **not** IV / front-door (instruments refused
    until an IV path exists).

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Industry causal
backends install via ``buildml[causal-industry]`` (dowhy, econml). EDA /
association paths never satisfy CausalAssumptions.

Lazy imports: core never grows heavy causal stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "CausalAssumptions",
    "CausalBackend",
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
    "causal_capability_matrix",
]


def declare_causal_assumptions(**kwargs: Any) -> Any:
    """Validate and return a :class:`CausalAssumptions` instance.

    Convenience entry point for Session and AI tools: parses keyword arguments
    into :class:`~buildml.causal.types.CausalAssumptions`, validates the full
    backdoor declaration, and returns the ready-to-fit object.

    Parameters
    ----------
    **kwargs:
        Mapping accepted by :meth:`~buildml.causal.types.CausalAssumptions.from_mapping`
        (``treatment``, ``outcome``, ``confounders``, acknowledgement flags).

    Returns
    -------
    CausalAssumptions
        Validated assumption object safe to pass to :func:`fit_causal`.
    """
    from buildml.causal.types import CausalAssumptions

    assumptions = CausalAssumptions.from_mapping(kwargs)
    assumptions.validate()
    return assumptions


def __getattr__(name: str) -> Any:
    if name in {
        "CausalAssumptions",
        "CausalBackend",
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
    if name == "causal_capability_matrix":
        from buildml.causal.catalog import causal_capability_matrix

        return causal_capability_matrix
    raise AttributeError(f"module 'buildml.causal' has no attribute {name!r}")
