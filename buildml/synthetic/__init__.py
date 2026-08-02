"""Synthetic-data systems (Session-shaped train-fitted tabular generators).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1–2 complete. Phase 3 — Application systems:
  Recommendation systems (**PASS**).
  Search / LTR (**PASS**).
  Knowledge graphs (**PASS**).
  Optimisation / decision helpers (**PASS**).
  **Synthetic-data systems (this module)** — **PASS** (Phase-1 bar).

Post-Phase-3 residuals (intentional):
  NLP/CV remain Torch/preprocess **hooks** (text_features, text/image loaders,
  pretrained zoo, speech ASR finetune-lite) — not separate Phase-3 packages.
  No focused deepening required unless a product beyond those hooks is desired.

Honesty (this package):
  - Train-fitted generators only (never fit on validation/test).
  - Methods: bootstrap / smoothed bootstrap, Gaussian copula (mixed types),
    and optional SMOTE wrap via ``buildml[imbalanced]``.
  - Distinct from ``Session.resample`` (class-balance preprocess lineage).
  - ``evaluate_synthetic`` offers fidelity metrics or TSTR utility — disclosed.
  - Merge into Session train only with explicit provenance (role=ignore);
    default sample returns a Frame without poisoning roles.
  - **Not** a differential-privacy product; bootstrap can near-duplicate rows.
  - Core stays light (numpy/scipy/sklearn); no SDV/CTGAN stack required.
    Optional ``buildml[imbalanced]`` only for method='smote'.

Lazy imports — keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "SyntheticEvalResult",
    "SyntheticSampleResult",
    "SynthesizerConfig",
    "SynthesizerFitResult",
    "SynthesizerMethod",
    "SynthesizerPlan",
    "evaluate_synthetic",
    "fit_synthesizer",
    "load_synthetic_bundle",
    "sample_synthetic",
    "save_synthetic_bundle",
    "synthetic_status",
    "synthetic_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {
        "SynthesizerConfig",
        "SynthesizerMethod",
        "ColumnKind",
        "EvalMode",
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
    if name == "fit_synthesizer":
        from buildml.synthetic.fit import fit_synthesizer

        return fit_synthesizer
    if name == "sample_synthetic":
        from buildml.synthetic.sample import sample_synthetic

        return sample_synthetic
    if name == "evaluate_synthetic":
        from buildml.synthetic.evaluate import evaluate_synthetic

        return evaluate_synthetic
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
