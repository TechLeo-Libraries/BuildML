"""Meta-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.metalearning.results import (
    MetaAdaptResult,
    MetaLearningEvalResult,
    MetaLearningFitResult,
    MetaLearningPlan,
)

BUNDLE_FORMAT = "buildml.metalearning_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Meta-learning bundles, multi-task bundles, online bundles, "
    "active-learning bundles, semi-supervised bundles, self-supervised "
    "bundles, anomaly bundles, unsupervised bundles, classical pipeline "
    "bundles, Torch trainer bundles, RAG bundles, and Session checkpoints "
    "are complementary, not interchangeable. A meta-learning bundle "
    "(buildml.metalearning_bundle.v1) stores a MetaLearningPlan "
    "(episodic few-shot protocol + feature/task contract + optional "
    "warm-start init estimator). A Session checkpoint stores data, roles, "
    "splits, history, and optional classical preprocess plans; it does not "
    "embed the meta-learner. Reload tabular workflow via checkpoint_load; "
    "reload the learner via load_metalearning_bundle. This is practical "
    "tabular few-shot / episodic meta-learning — not foundation-model "
    "MAML-at-scale."
)


def save_metalearning_bundle(
    path: str | Path,
    plan: MetaLearningPlan,
    *,
    fit_result: MetaLearningFitResult | None = None,
    eval_result: MetaLearningEvalResult | None = None,
    adapt_result: MetaAdaptResult | None = None,
) -> Path:
    """Write a meta-learning bundle directory (``buildml.metalearning_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No MetaLearningPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "metalearning_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "adapt": None if adapt_result is None else adapt_result.to_dict(),
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return destination


def load_metalearning_bundle(path: str | Path) -> MetaLearningPlan:
    """Load a meta-learning bundle into a :class:`MetaLearningPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "metalearning_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete meta-learning bundle at {root}. "
            f"Expected meta.json and metalearning_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported meta-learning bundle format {fmt!r}; "
            f"expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, MetaLearningPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "metalearning_plan.joblib must contain a MetaLearningPlan or a "
            "payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, MetaLearningPlan):
        raise ValidationError("Loaded plan object is not a MetaLearningPlan")
    return plan
