"""CBR bundle persistence (distinct from Session checkpoints and RAG bundles)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.cbr.results import CbrEvalResult, CbrFitResult, CbrPlan
from buildml.core.errors import ValidationError

BUNDLE_FORMAT = "buildml.cbr_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "CBR bundles, classical pipeline bundles, Torch trainer bundles, "
    "RAG bundles, symbolic bundles, and Session checkpoints are complementary, "
    "not interchangeable. A CBR bundle (buildml.cbr_bundle.v1) stores a CbrPlan "
    "(train-built case memory + metric/reuse config). A Session checkpoint stores "
    "data, roles, splits, history, and optional classical preprocess plans; it "
    "does not embed the case memory. Reload tabular workflow via checkpoint_load; "
    "reload the learner via load_cbr_bundle. Honesty: tabular case→solution CBR "
    "— not RAG document retrieval, not a vector DB product."
)


def save_cbr_bundle(
    path: str | Path,
    plan: CbrPlan,
    *,
    fit_result: CbrFitResult | None = None,
    eval_result: CbrEvalResult | None = None,
) -> Path:
    """Write a CBR bundle directory (``buildml.cbr_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No CbrPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "cbr_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "kind": "cbr",
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return destination


def load_cbr_bundle(path: str | Path) -> CbrPlan:
    """Load a CBR bundle into a plan object."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "cbr_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete CBR bundle at {root}. "
            f"Expected meta.json and cbr_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported CBR bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, CbrPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "cbr_plan.joblib must contain a plan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, CbrPlan):
        raise ValidationError("Loaded plan object is not a CbrPlan.")
    return plan
