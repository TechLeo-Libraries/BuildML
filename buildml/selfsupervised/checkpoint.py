"""Self-supervised bundle persistence (distinct from Session checkpoints / Torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.selfsupervised.results import (
    SSLHeadFitResult,
    SSLHeadPlan,
    SelfSupervisedEvalResult,
    SelfSupervisedFitResult,
    SelfSupervisedPlan,
)

BUNDLE_FORMAT = "buildml.selfsupervised_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Self-supervised bundles, semi-supervised bundles, Torch trainer bundles, "
    "pretrained zoo backbones, classical pipeline bundles, RAG bundles, and "
    "Session checkpoints are complementary, not interchangeable. "
    "A self-supervised bundle (buildml.selfsupervised_bundle.v1) stores a "
    "train-fitted SelfSupervisedPlan (masked tabular encoder + feature contract) "
    "and optionally an SSLHeadPlan. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the SSL encoder. "
    "Reload tabular workflow via checkpoint_load; reload SSL via "
    "load_ssl_bundle. Vision/audio/speech transfer remains "
    "load_pretrained_backbone / attach_backbone_head."
)


def save_ssl_bundle(
    path: str | Path,
    plan: SelfSupervisedPlan,
    *,
    fit_result: SelfSupervisedFitResult | None = None,
    head_plan: SSLHeadPlan | None = None,
    head_fit_result: SSLHeadFitResult | None = None,
    eval_result: SelfSupervisedEvalResult | None = None,
) -> Path:
    """Write a self-supervised bundle directory (``buildml.selfsupervised_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No SelfSupervisedPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan, "head_plan": head_plan}
    joblib.dump(payload, destination / "ssl_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "head": None if head_plan is None else head_plan.to_dict(),
        "head_fit": None if head_fit_result is None else head_fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_ssl_bundle(path: str | Path) -> tuple[SelfSupervisedPlan, SSLHeadPlan | None]:
    """Load a self-supervised bundle into plan (+ optional head)."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "ssl_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete self-supervised bundle at {root}. "
            f"Expected meta.json and ssl_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported self-supervised bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, SelfSupervisedPlan):
        return loaded, None
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "ssl_plan.joblib must contain a SelfSupervisedPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, SelfSupervisedPlan):
        raise ValidationError("Loaded plan object is not a SelfSupervisedPlan")
    head = loaded.get("head_plan")
    if head is not None and not isinstance(head, SSLHeadPlan):
        raise ValidationError("Loaded head_plan object is not an SSLHeadPlan")
    return plan, head
