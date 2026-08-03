"""Multi-task bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
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

    Validates bundle format version and plan object type before returning.

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
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "multitask_plan.joblib must contain a MultiTaskPlan or a payload "
            "with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, MultiTaskPlan):
        raise ValidationError("Loaded plan object is not a MultiTaskPlan")
    return plan
