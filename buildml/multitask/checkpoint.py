"""Multi-task bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.multitask.results import MultiTaskEvalResult, MultiTaskFitResult, MultiTaskPlan

BUNDLE_FORMAT = "buildml.multitask_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Multi-task bundles, online bundles, active-learning bundles, "
    "semi-supervised bundles, self-supervised bundles, anomaly bundles, "
    "unsupervised bundles, classical pipeline bundles, Torch trainer bundles, "
    "RAG bundles, and Session checkpoints are complementary, not "
    "interchangeable. A multi-task bundle (buildml.multitask_bundle.v1) stores "
    "a MultiTaskPlan (multi-output / chain estimator + target contract + "
    "per-task label encoders). A Session checkpoint stores data, roles, "
    "splits, history, and optional classical preprocess plans; it does not "
    "embed the multi-task learner. Reload tabular workflow via "
    "checkpoint_load; reload the learner via load_multitask_bundle. This is "
    "sklearn MultiOutput / Chain on shared features: not a deep MTL platform."
)


def save_multitask_bundle(
    path: str | Path,
    plan: MultiTaskPlan,
    *,
    fit_result: MultiTaskFitResult | None = None,
    eval_result: MultiTaskEvalResult | None = None,
) -> Path:
    """Write a multi-task bundle directory (``buildml.multitask_bundle.v1``).

    Persists the plan and optional fit/eval summaries as joblib + JSON metadata.
    Distinct from Session checkpoints.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Fitted :class:`~buildml.multitask.results.MultiTaskPlan`.
    fit_result:
        Optional fit summary to embed in ``meta.json``.
    eval_result:
        Optional evaluation summary to embed in ``meta.json``.

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
        raise ValidationError("No MultiTaskPlan to save.")
    _validate_plan_contract(plan)
    if fit_result is not None:
        _validate_summary(plan, fit_result.to_dict())
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "multitask_plan.joblib")
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


def load_multitask_bundle(path: str | Path, *, trusted: bool = False) -> MultiTaskPlan:
    """Load a multi-task bundle into a :class:`MultiTaskPlan`.

    Validates bundle format and fitted head contracts before returning.
    Older fixed-task plans with contradictory head kinds require refitting;
    loading cannot repair a model trained with the wrong head type.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``multitask_plan.joblib``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    MultiTaskPlan
        Deserialized plan with estimator and label encoders attached.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or plan type is wrong.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "multitask_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete multi-task bundle at {root}. "
            f"Expected meta.json and multitask_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported multi-task bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, MultiTaskPlan):
        plan = loaded
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
    else:
        raise ValidationError(
            "multitask_plan.joblib must contain a MultiTaskPlan or a payload with key 'plan'."
        )
    _validate_plan_contract(plan)
    _validate_summary(plan, meta.get("plan"))
    _validate_summary(plan, meta.get("fit"))
    return plan


def _validate_plan_contract(plan: MultiTaskPlan) -> None:
    """Refuse unusable or internally inconsistent model contracts."""
    if not isinstance(plan, MultiTaskPlan):
        raise ValidationError("Loaded plan object is not a MultiTaskPlan")
    for name, values in (("columns", plan.columns), ("target_columns", plan.target_columns)):
        if (
            not isinstance(values, (tuple, list))
            or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValidationError(f"Multi-task {name} must contain unique non-empty names.")
    if plan.task not in {"regression", "classification", "mixed"}:
        raise ValidationError(f"Invalid multi-task task {plan.task!r}.")
    if (
        isinstance(plan.n_train_rows, bool)
        or not isinstance(plan.n_train_rows, Integral)
        or plan.n_train_rows < 1
    ):
        raise ValidationError("Multi-task n_train_rows must be a positive integer.")
    model = plan.estimator_
    if not callable(getattr(model, "predict", None)):
        raise ValidationError("Multi-task plan requires a fitted predictive estimator.")
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_is_fitted

    if isinstance(model, BaseEstimator):
        try:
            check_is_fitted(model)
        except (ValueError, TypeError) as exc:
            raise ValidationError("Multi-task estimator is not fitted.") from exc
    if hasattr(model, "module_") and model.module_ is None:
        raise ValidationError("Multi-task Torch estimator is not fitted.")
    width = getattr(model, "n_features_in_", getattr(model, "n_features_", None))
    if width is not None and width != len(plan.columns):
        raise ValidationError("Multi-task feature columns disagree with fitted estimator width.")
    heads = getattr(model, "estimators_", None)
    if heads is not None and len(heads) != len(plan.target_columns):
        raise ValidationError("Multi-task targets disagree with fitted estimator head count.")
    stored_targets = getattr(model, "target_columns", None)
    if stored_targets is not None and tuple(stored_targets) != tuple(plan.target_columns):
        raise ValidationError("Multi-task target order disagrees with fitted estimator.")
    stored_kinds = getattr(model, "task_kinds", None)
    for target in plan.target_columns:
        kind = plan.task_kinds_.get(target, plan.task)
        if kind not in {"classification", "regression"}:
            raise ValidationError(f"Missing or invalid task kind for {target!r}.")
        if plan.task != "mixed" and kind != plan.task:
            raise ValidationError(
                f"Multi-task fixed task {plan.task!r} disagrees with head {target!r}. "
                "Refit this plan with consistent task kinds."
            )
        if stored_kinds is not None and stored_kinds.get(target) != kind:
            raise ValidationError(
                f"Multi-task head {target!r} disagrees with fitted estimator task kinds. "
                "Refit this plan with consistent task kinds."
            )
        if kind == "classification":
            encoder = plan.label_encoders_.get(target)
            classes = getattr(encoder, "classes_", None)
            if classes is None or list(classes) != list(plan.classes_per_task_.get(target, ())):
                raise ValidationError(f"Multi-task label encoder/classes disagree for {target!r}.")


def _validate_summary(plan: MultiTaskPlan, summary: Any) -> None:
    """Check available summaries while retaining older summary-free bundles."""
    if summary is None:
        return
    if not isinstance(summary, dict):
        raise ValidationError("Multi-task bundle summary must be an object.")
    expected = plan.to_dict()
    for key in (
        "columns",
        "target_columns",
        "task",
        "method",
        "backend",
        "n_train_rows",
        "n_tasks",
        "classes_per_task",
        "task_kinds",
    ):
        if key in summary and summary[key] != expected[key]:
            raise ValidationError(f"Multi-task bundle summary {key!r} disagrees with fitted plan.")
