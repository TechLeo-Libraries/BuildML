"""Semi-supervised bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.semisupervised.results import (
    SemiSupervisedEvalResult,
    SemiSupervisedFitResult,
    SemiSupervisedPlan,
)

BUNDLE_FORMAT = "buildml.semisupervised_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Semi-supervised bundles, self-supervised bundles, anomaly bundles, "
    "unsupervised bundles, classical pipeline bundles, Torch trainer bundles, "
    "RAG bundles, and Session checkpoints are complementary, not interchangeable. "
    "A semi-supervised bundle (buildml.semisupervised_bundle.v1) stores a "
    "train-fitted SemiSupervisedPlan (estimator + label missingness contract). "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the semi-supervised estimator. "
    "Reload tabular workflow via checkpoint_load; reload semi-supervised via "
    "load_semisupervised_bundle."
)


def save_semisupervised_bundle(
    path: str | Path,
    plan: SemiSupervisedPlan,
    *,
    fit_result: SemiSupervisedFitResult | None = None,
    eval_result: SemiSupervisedEvalResult | None = None,
) -> Path:
    """Write a semi-supervised bundle directory (``buildml.semisupervised_bundle.v1``).

Persists or restores plan state as joblib plus JSON metadata. Distinct from Session checkpoints: reload workflow via checkpoint_load separately.

Parameters
----------
path:
    Filesystem path to the bundle directory.
plan:
    Fitted plan object carrying model state and feature contract.
fit_result:
    Optional fit summary to embed in bundle metadata or history.
eval_result:
    Optional evaluation summary for bundle metadata or history.

Returns
-------
Path
    Resolved filesystem path to the written bundle directory.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if plan is None:
        raise ValidationError("No SemiSupervisedPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "semisupervised_plan.joblib")
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


def load_semisupervised_bundle(path: str | Path, *, trusted: bool = False) -> SemiSupervisedPlan:
    """Load a semi-supervised bundle into a :class:`SemiSupervisedPlan`.

    Persists or restores plan state as joblib plus JSON metadata. Distinct from Session checkpoints: reload workflow via checkpoint_load separately.

    Parameters
    ----------
    path:
        Filesystem path to the bundle directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    SemiSupervisedPlan
        Fitted plan object (SemiSupervisedPlan) with private estimators attached.

    Raises
    ------
    ValidationError
        When preconditions for this operation are not met.
        
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "semisupervised_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete semi-supervised bundle at {root}. "
            f"Expected meta.json and semisupervised_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported semi-supervised bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, SemiSupervisedPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "semisupervised_plan.joblib must contain a SemiSupervisedPlan or a "
            "payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, SemiSupervisedPlan):
        raise ValidationError("Loaded plan object is not a SemiSupervisedPlan")
    return plan
