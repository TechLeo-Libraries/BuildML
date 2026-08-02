"""Probabilistic-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.probabilistic.results import (
    ProbabilisticEvalResult,
    ProbabilisticFitResult,
    ProbabilisticPlan,
)

BUNDLE_FORMAT = "buildml.probabilistic_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Probabilistic bundles, federated bundles, online bundles, classical "
    "pipeline bundles, Torch trainer bundles, RAG bundles, and Session "
    "checkpoints are complementary, not interchangeable. A probabilistic "
    "bundle (buildml.probabilistic_bundle.v1) stores a ProbabilisticPlan "
    "(sklearn Bayesian/GP/NB estimator + optional split-conformal quantile + "
    "train carve indices). A Session checkpoint stores data, roles, splits, "
    "history, and optional classical preprocess plans; it does not embed the "
    "probabilistic learner. Reload tabular workflow via checkpoint_load; "
    "reload the learner via load_probabilistic_bundle. Honesty: sklearn "
    "uncertainty quantification — not a PyMC/Stan MCMC platform."
)


def save_probabilistic_bundle(
    path: str | Path,
    plan: ProbabilisticPlan,
    *,
    fit_result: ProbabilisticFitResult | None = None,
    eval_result: ProbabilisticEvalResult | None = None,
) -> Path:
    """Write a probabilistic bundle directory (``buildml.probabilistic_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No ProbabilisticPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "probabilistic_plan.joblib")
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


def load_probabilistic_bundle(path: str | Path) -> ProbabilisticPlan:
    """Load a probabilistic bundle into a :class:`ProbabilisticPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "probabilistic_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete probabilistic bundle at {root}. "
            f"Expected meta.json and probabilistic_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported probabilistic bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, ProbabilisticPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "probabilistic_plan.joblib must contain a ProbabilisticPlan or a "
            "payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, ProbabilisticPlan):
        raise ValidationError("Loaded plan object is not a ProbabilisticPlan")
    return plan
