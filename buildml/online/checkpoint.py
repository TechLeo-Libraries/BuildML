"""Online-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.online.results import OnlineEvalResult, OnlineFitResult, OnlinePlan

BUNDLE_FORMAT = "buildml.online_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Online / continual bundles, active-learning bundles, semi-supervised "
    "bundles, self-supervised bundles, anomaly bundles, unsupervised bundles, "
    "classical pipeline bundles, Torch trainer bundles, RAG bundles, and "
    "Session checkpoints are complementary, not interchangeable. An online "
    "bundle (buildml.online_bundle.v1) stores an OnlinePlan (incremental "
    "estimator + cursor + update history + class vocabulary). A Session "
    "checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the online learner. Reload tabular "
    "workflow via checkpoint_load; reload the learner via load_online_bundle. "
    "This is batch/stream-chunk partial_fit state — not a streaming platform."
)


def save_online_bundle(
    path: str | Path,
    plan: OnlinePlan,
    *,
    fit_result: OnlineFitResult | None = None,
    eval_result: OnlineEvalResult | None = None,
) -> Path:
    """Write an online-learning bundle directory (``buildml.online_bundle.v1``).

    Persists the plan and optional fit/eval summaries as joblib + JSON metadata.
    Distinct from Session checkpoints.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Fitted :class:`~buildml.online.results.OnlinePlan`.
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
        raise ValidationError("No OnlinePlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "online_plan.joblib")
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


def load_online_bundle(path: str | Path) -> OnlinePlan:
    """Load an online-learning bundle into an :class:`OnlinePlan`.

    Validates bundle format version and plan object type before returning.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``online_plan.joblib``.

    Returns
    -------
    OnlinePlan
        Deserialized plan with estimator and label encoder attached.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or plan type is wrong.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "online_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete online-learning bundle at {root}. "
            f"Expected meta.json and online_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported online-learning bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, OnlinePlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "online_plan.joblib must contain an OnlinePlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, OnlinePlan):
        raise ValidationError("Loaded plan object is not an OnlinePlan")
    return plan
