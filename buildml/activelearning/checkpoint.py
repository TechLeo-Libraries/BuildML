"""Active-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.activelearning.results import (
    ActiveLearningEvalResult,
    ActiveLearningFitResult,
    ActiveLearningPlan,
)

BUNDLE_FORMAT = "buildml.activelearning_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Active-learning bundles, semi-supervised bundles, self-supervised bundles, "
    "anomaly bundles, unsupervised bundles, classical pipeline bundles, Torch "
    "trainer bundles, RAG bundles, and Session checkpoints are complementary, "
    "not interchangeable. An active-learning bundle "
    "(buildml.activelearning_bundle.v1) stores a train-fitted ActiveLearningPlan "
    "(estimator + labeled/pool indices + query history + budget). A Session "
    "checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the active learner. Reload tabular "
    "workflow via checkpoint_load; reload the learner via "
    "load_active_learning_bundle. Labels always come from the user — the bundle "
    "does not embed a fake oracle."
)


def save_active_learning_bundle(
    path: str | Path,
    plan: ActiveLearningPlan,
    *,
    fit_result: ActiveLearningFitResult | None = None,
    eval_result: ActiveLearningEvalResult | None = None,
) -> Path:
    """Write an active-learning bundle directory (``buildml.activelearning_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No ActiveLearningPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "activelearning_plan.joblib")
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


def load_active_learning_bundle(path: str | Path) -> ActiveLearningPlan:
    """Load an active-learning bundle into a :class:`ActiveLearningPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "activelearning_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete active-learning bundle at {root}. "
            f"Expected meta.json and activelearning_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported active-learning bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, ActiveLearningPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "activelearning_plan.joblib must contain an ActiveLearningPlan or a "
            "payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, ActiveLearningPlan):
        raise ValidationError("Loaded plan object is not an ActiveLearningPlan")
    return plan
