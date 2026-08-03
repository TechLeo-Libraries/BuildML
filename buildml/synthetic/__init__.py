"""Synthetic-data systems (Session-shaped train-fitted tabular generators).

Phase coverage (internal tracker: depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1–2 complete. Phase 3: Application systems:
  Recommendation systems (**PASS**).
  Search / LTR (**PASS**).
  Knowledge graphs (**PASS**).
  Optimisation / decision helpers (**PASS**).
  **Synthetic-data systems (this module)**: **PASS** (Phase-1 bar + R6.10 industry).

Industry depth (R6.10):
  - Native fallback: bootstrap / Gaussian copula / SMOTE wrap.
  - SDV CTGAN/TVAE/CopulaGAN (``buildml[synthetic-industry]``).
  - SDMetrics quality reports when extra installed; built-in KS/TV/corr always.
  - ``validate_synthetic`` built-in checks + optional GE lite when installed.

Honesty (this package):
  - Train-fitted generators only (never fit on validation/test).
  - Distinct from ``Session.resample`` (class-balance preprocess lineage).
  - ``evaluate_synthetic`` offers fidelity metrics or TSTR utility: disclosed.
  - Merge into Session train only with explicit provenance (role=ignore).
  - **Not** a differential-privacy product.
  - Core stays light; SDV stack is optional ``buildml[synthetic-industry]``.

Lazy imports: keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "SyntheticBackend",
    "SyntheticEvalResult",
    "SyntheticSampleResult",
    "SyntheticValidationResult",
    "SynthesizerConfig",
    "SynthesizerFitResult",
    "SynthesizerMethod",
    "SynthesizerPlan",
    "EvalBackend",
    "evaluate_synthetic",
    "fit_synthesizer",
    "list_synthetic_methods",
    "load_synthetic_bundle",
    "sample_synthetic",
    "save_synthetic_bundle",
    "synthetic_capability_matrix",
    "synthetic_status",
    "synthetic_status_for_session",
    "validate_synthetic",
]


def __getattr__(name: str) -> Any:
    if name in {
        "SynthesizerConfig",
        "SynthesizerMethod",
        "SyntheticBackend",
        "ColumnKind",
        "EvalMode",
        "EvalBackend",
        "MergeMode",
        "ColumnSchemaSpec",
    }:
        from buildml.synthetic import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "SynthesizerPlan",
        "SynthesizerFitResult",
        "SyntheticSampleResult",
        "SyntheticEvalResult",
    }:
        from buildml.synthetic import results as results_mod

        return getattr(results_mod, name)
    if name == "SyntheticValidationResult":
        from buildml.synthetic.validation import SyntheticValidationResult

        return SyntheticValidationResult
    if name == "fit_synthesizer":
        from buildml.synthetic.fit import fit_synthesizer

        return fit_synthesizer
    if name == "sample_synthetic":
        from buildml.synthetic.sample import sample_synthetic

        return sample_synthetic
    if name == "evaluate_synthetic":
        from buildml.synthetic.evaluate import evaluate_synthetic

        return evaluate_synthetic
    if name == "validate_synthetic":
        from buildml.synthetic.validation import validate_synthetic

        return validate_synthetic
    if name in {"synthetic_capability_matrix", "list_synthetic_methods"}:
        from buildml.synthetic import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_synthetic_bundle",
        "load_synthetic_bundle",
    }:
        from buildml.synthetic import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"synthetic_status", "synthetic_status_for_session"}:
        from buildml.synthetic import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.synthetic' has no attribute {name!r}")
