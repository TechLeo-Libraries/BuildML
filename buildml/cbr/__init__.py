"""Case-based reasoning (tabular case memory → retrieve → reuse/adapt).

Phase coverage (internal tracker: depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2:
  1–10. Semi-supervised → … → symbolic / neuro-symbolic: prior items.
  **This module:** Case-based reasoning (Session retrieval/memory path).
  Next: Imitation learning + Reinforcement learning (coherent delivery or
  RL with IL as mode: both to full bar), then TDA, then app systems.

Explicit non-goals (no product surfaces): RAG-as-CBR (document retrieval for
generation), vector DB products, full cognitive CBR research suites, Prolog/Z3,
fuzzy products, imitation+RL (next), TDA, app systems.

Honesty (this package):
  - Tabular case base from Session **train** rows (features + solution/label).
  - Retrieve k nearest / most similar cases (euclidean / manhattan / cosine /
    mixed Gower-style); reuse via majority / distance-weighted vote or
    local mean / local Ridge; optional lite adapt + retain hooks.
  - Explanation traces disclose which cases influenced each answer.
  - **Not** RAG. Sharing "nearest neighbors" does not make CBR a RAG submodule.
  - Core stays light: numpy/pandas/sklearn distances only.

Lazy imports: core never grows heavy IR / vector-DB stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "Case",
    "CaseBase",
    "CaseTrace",
    "CbrAdaptMode",
    "CbrBackend",
    "CbrConfig",
    "CbrEvalResult",
    "CbrFitResult",
    "CbrMetric",
    "CbrPlan",
    "CbrPredictResult",
    "CbrRetainResult",
    "CbrRetrieveResult",
    "CbrReuseMode",
    "CbrTask",
    "cbr_capability_matrix",
    "cbr_status",
    "cbr_status_for_session",
    "evaluate_cbr",
    "fit_cbr",
    "load_cbr_bundle",
    "predict_cbr",
    "retain_cbr",
    "retrieve_cases",
    "save_cbr_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CbrTask",
        "CbrMetric",
        "CbrReuseMode",
        "CbrAdaptMode",
        "CbrBackend",
        "CbrConfig",
    }:
        from buildml.cbr import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "CbrPlan",
        "CbrFitResult",
        "CbrEvalResult",
        "CbrPredictResult",
        "CbrRetrieveResult",
        "CbrRetainResult",
    }:
        from buildml.cbr import results as results_mod

        return getattr(results_mod, name)
    if name in {"Case", "CaseBase", "CaseTrace"}:
        from buildml.cbr import cases as cases_mod

        return getattr(cases_mod, name)
    if name == "fit_cbr":
        from buildml.cbr.fit import fit_cbr

        return fit_cbr
    if name == "evaluate_cbr":
        from buildml.cbr.evaluate import evaluate_cbr

        return evaluate_cbr
    if name == "predict_cbr":
        from buildml.cbr.predict import predict_cbr

        return predict_cbr
    if name == "retrieve_cases":
        from buildml.cbr.retrieve import retrieve_cases

        return retrieve_cases
    if name == "retain_cbr":
        from buildml.cbr.retain import retain_cbr

        return retain_cbr
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_cbr_bundle",
        "load_cbr_bundle",
    }:
        from buildml.cbr import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name == "cbr_capability_matrix":
        from buildml.cbr.catalog import cbr_capability_matrix

        return cbr_capability_matrix
    if name in {"cbr_status", "cbr_status_for_session"}:
        from buildml.cbr import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.cbr' has no attribute {name!r}")
