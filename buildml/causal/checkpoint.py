"""Causal-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.causal.results import (
    CausalEvalResult,
    CausalFitResult,
    CausalPlan,
)
from buildml.core.errors import ValidationError

BUNDLE_FORMAT = "buildml.causal_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Causal bundles, probabilistic bundles, classical pipeline bundles, "
    "Torch trainer bundles, RAG bundles, and Session checkpoints are "
    "complementary, not interchangeable. A causal bundle "
    "(buildml.causal_bundle.v1) stores a CausalPlan (declared "
    "CausalAssumptions + fitted nuisance models + train ATE / bootstrap). "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the causal learner. "
    "Reload tabular workflow via checkpoint_load; reload the learner via "
    "load_causal_bundle. Honesty: backdoor ATE under caller-declared "
    "assumptions — not causal discovery; not a DoWhy/EconML platform. "
    "EDA remains associational and never substitutes for CausalAssumptions."
)


def save_causal_bundle(
    path: str | Path,
    plan: CausalPlan,
    *,
    fit_result: CausalFitResult | None = None,
    eval_result: CausalEvalResult | None = None,
) -> Path:
    """Write a causal bundle directory (``buildml.causal_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No CausalPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "causal_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_causal_bundle(path: str | Path) -> CausalPlan:
    """Load a causal bundle into a :class:`CausalPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "causal_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete causal bundle at {root}. "
            f"Expected meta.json and causal_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported causal bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, CausalPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "causal_plan.joblib must contain a CausalPlan or a payload "
            "with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, CausalPlan):
        raise ValidationError("Loaded plan object is not a CausalPlan")
    return plan
