"""Ensemble bundle persistence (distinct from Session checkpoints / pipelines)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.ensemble.results import EnsembleFitResult, EnsemblePlan
from buildml.model.supervised import FitResult

BUNDLE_FORMAT = "buildml.ensemble_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Ensemble bundles, classical pipeline bundles, unsupervised bundles, Torch "
    "trainer bundles, RAG bundles, and Session checkpoints are complementary, "
    "not interchangeable. "
    "An ensemble bundle (buildml.ensemble_bundle.v1) stores a train-fitted "
    "EnsemblePlan (strategy disclosures + sklearn-compatible ensemble estimator) "
    "and the classical FitResult feature contract. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the ensemble. "
    "Prefer save_pipeline when preprocess plans must travel with the estimator; "
    "use save_ensemble_bundle when strategy disclosures and EnsemblePlan matter."
)


def save_ensemble_bundle(
    path: str | Path,
    plan: EnsemblePlan,
    *,
    fit_result: FitResult | None = None,
    ensemble_fit_result: EnsembleFitResult | None = None,
) -> Path:
    """Write an ensemble bundle directory (``buildml.ensemble_bundle.v1``).

    Layout
    ------
    ``meta.json``, ``ensemble_plan.joblib``.
    """
    if plan is None:
        raise ValidationError("No EnsemblePlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "fit_result": fit_result,
    }
    joblib.dump(payload, destination / "ensemble_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "ensemble_fit": None
        if ensemble_fit_result is None
        else ensemble_fit_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_ensemble_bundle(path: str | Path) -> tuple[EnsemblePlan, FitResult | None]:
    """Load an ensemble bundle into an :class:`EnsemblePlan` (+ optional FitResult)."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "ensemble_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete ensemble bundle at {root}. "
            f"Expected meta.json and ensemble_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported ensemble bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, EnsemblePlan):
        plan = loaded
        fit_result = None
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        fit_result = loaded.get("fit_result")
    else:
        raise ValidationError(
            "ensemble_plan.joblib must contain an EnsemblePlan or a payload with key 'plan'."
        )
    if not isinstance(plan, EnsemblePlan):
        raise ValidationError("Loaded plan object is not an EnsemblePlan")
    if fit_result is not None and not isinstance(fit_result, FitResult):
        # Reconstruct a FitResult from the plan estimator when payload is partial.
        fit_result = FitResult(
            estimator=plan.estimator_,
            task=plan.task,
            feature_columns=plan.feature_columns,
            target_column=plan.target_column,
            n_train_rows=plan.n_train_rows,
        )
    elif fit_result is None:
        fit_result = FitResult(
            estimator=plan.estimator_,
            task=plan.task,
            feature_columns=plan.feature_columns,
            target_column=plan.target_column,
            n_train_rows=plan.n_train_rows,
        )
    return plan, fit_result
