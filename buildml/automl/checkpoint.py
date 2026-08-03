"""AutoML bundle persistence (distinct from Session checkpoints / pipelines)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.automl.results import AutoMLPlan, AutoMLResult, fit_result_from_plan
from buildml.core.errors import ValidationError
from buildml.model.supervised import FitResult

BUNDLE_FORMAT = "buildml.automl_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "AutoML bundles, classical pipeline bundles, ensemble bundles, unsupervised "
    "bundles, Torch trainer bundles, RAG bundles, and Session checkpoints are "
    "complementary, not interchangeable. "
    "An AutoML bundle (buildml.automl_bundle.v1) stores a train-selected "
    "AutoMLPlan (family/recipe disclosures + fitted estimator, often a "
    "fold-local preprocess Pipeline) and the classical FitResult feature "
    "contract. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the AutoML plan. "
    "Prefer save_pipeline when Session-global preprocess plans must travel with "
    "the estimator; use save_automl_bundle when AutoML search disclosures and "
    "AutoMLPlan matter."
)

def save_automl_bundle(
    path: str | Path,
    plan: AutoMLPlan,
    *,
    fit_result: FitResult | None = None,
    automl_result: AutoMLResult | None = None,
) -> Path:
    """Write an AutoML bundle directory (``buildml.automl_bundle.v1``).

    Persists the plan and optional classical fit / search summaries as joblib +
    JSON metadata. Distinct from Session checkpoints.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Fitted :class:`~buildml.automl.results.AutoMLPlan`.
    fit_result:
        Optional classical :class:`~buildml.model.supervised.FitResult` to
        embed alongside the plan.
    automl_result:
        Optional search summary to embed in ``meta.json``.

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No AutoMLPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "fit_result": fit_result,
    }
    joblib.dump(payload, destination / "automl_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "automl": None if automl_result is None else automl_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination

def load_automl_bundle(path: str | Path) -> tuple[AutoMLPlan, FitResult | None]:
    """Load an AutoML bundle into an :class:`AutoMLPlan` and optional FitResult.

    Validates bundle format version and plan object type before returning.
    Reconstructs a classical FitResult from the plan when none was stored.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``automl_plan.joblib``.

    Returns
    -------
    tuple[AutoMLPlan, FitResult | None]
        Deserialized plan with estimator attached, plus optional FitResult for
        classical evaluate/predict paths.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or plan type is wrong.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "automl_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete AutoML bundle at {root}. "
            f"Expected meta.json and automl_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported AutoML bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, AutoMLPlan):
        plan = loaded
        fit_result = None
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        fit_result = loaded.get("fit_result")
    else:
        raise ValidationError(
            "automl_plan.joblib must contain an AutoMLPlan or a payload with key 'plan'."
        )
    if not isinstance(plan, AutoMLPlan):
        raise ValidationError("Loaded plan object is not an AutoMLPlan")
    if fit_result is not None and not isinstance(fit_result, FitResult):
        fit_result = fit_result_from_plan(plan)
    elif fit_result is None:
        fit_result = fit_result_from_plan(plan)
    return plan, fit_result
