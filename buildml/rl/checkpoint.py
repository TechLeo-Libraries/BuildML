"""Imitation + RL bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.rl.results import (
    ImitationEvalResult,
    ImitationFitResult,
    ImitationPlan,
    RlEvalResult,
    RlFitResult,
    RlPlan,
)

BUNDLE_FORMAT_IMITATION = "buildml.imitation_bundle.v1"
BUNDLE_FORMAT_RL = "buildml.rl_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Imitation bundles, RL bundles, CBR bundles, classical pipeline bundles, "
    "Torch trainer bundles, RAG bundles, and Session checkpoints are complementary, "
    "not interchangeable. An imitation bundle (buildml.imitation_bundle.v1) stores a "
    "train-fitted ImitationPlan (behavioral cloning policy). An RL bundle "
    "(buildml.rl_bundle.v1) stores a train-fitted RlPlan (contextual bandit or "
    "Gymnasium REINFORCE-lite policy). A Session checkpoint stores data, roles, "
    "splits, history, and optional classical preprocess plans; it does not embed "
    "IL/RL policies. Reload tabular workflow via checkpoint_load; reload policies "
    "via load_imitation_bundle / load_rl_bundle. Honesty: not MuJoCo/robotics."
)


def save_imitation_bundle(
    path: str | Path,
    plan: ImitationPlan,
    *,
    fit_result: ImitationFitResult | None = None,
    eval_result: ImitationEvalResult | None = None,
) -> Path:
    """Write an imitation bundle directory (``buildml.imitation_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No ImitationPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"plan": plan}, destination / "imitation_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT_IMITATION,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_imitation_bundle(path: str | Path) -> ImitationPlan:
    """Load an imitation bundle into an :class:`ImitationPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "imitation_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete imitation bundle at {root}. "
            f"Expected meta.json and imitation_plan.joblib ({BUNDLE_FORMAT_IMITATION})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT_IMITATION:
        raise ValidationError(
            f"Unsupported imitation bundle format {fmt!r}; expected {BUNDLE_FORMAT_IMITATION}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, ImitationPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "imitation_plan.joblib must contain an ImitationPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, ImitationPlan):
        raise ValidationError("Loaded plan object is not an ImitationPlan")
    return plan


def save_rl_bundle(
    path: str | Path,
    plan: RlPlan,
    *,
    fit_result: RlFitResult | None = None,
    eval_result: RlEvalResult | None = None,
) -> Path:
    """Write an RL bundle directory (``buildml.rl_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No RlPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"plan": plan}, destination / "rl_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT_RL,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_rl_bundle(path: str | Path) -> RlPlan:
    """Load an RL bundle into an :class:`RlPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "rl_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete RL bundle at {root}. "
            f"Expected meta.json and rl_plan.joblib ({BUNDLE_FORMAT_RL})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT_RL:
        raise ValidationError(
            f"Unsupported RL bundle format {fmt!r}; expected {BUNDLE_FORMAT_RL}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, RlPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "rl_plan.joblib must contain an RlPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, RlPlan):
        raise ValidationError("Loaded plan object is not an RlPlan")
    return plan
