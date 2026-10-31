"""Semi-supervised bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from numbers import Integral
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
    _validate_plan_contract(plan)
    if fit_result is not None:
        _validate_summary(plan, fit_result.to_dict())
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
        plan = loaded
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
    else:
        raise ValidationError(
            "semisupervised_plan.joblib must contain a SemiSupervisedPlan or a "
            "payload with key 'plan'."
        )
    _validate_plan_contract(plan)
    _validate_summary(plan, meta.get("plan"))
    _validate_summary(plan, meta.get("fit"))
    return plan


def _validate_plan_contract(plan: SemiSupervisedPlan) -> None:
    """Validate available fitted shape, class and missingness contracts."""
    if not isinstance(plan, SemiSupervisedPlan):
        raise ValidationError("Loaded plan object is not a SemiSupervisedPlan")
    if (
        not isinstance(plan.columns, (tuple, list))
        or not plan.columns
        or any(not isinstance(c, str) or not c for c in plan.columns)
        or len(set(plan.columns)) != len(plan.columns)
    ):
        raise ValidationError("Semi-supervised feature columns must be unique non-empty names.")
    if not isinstance(plan.target_column, str) or not plan.target_column:
        raise ValidationError("Semi-supervised target column must be a non-empty name.")
    for name in ("n_train_rows", "n_labeled_train", "n_unlabeled_train"):
        count = getattr(plan, name)
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValidationError(f"Semi-supervised {name} must be a non-negative integer.")
    if (
        plan.n_train_rows < 1
        or plan.n_labeled_train < 1
        or plan.n_labeled_train + plan.n_unlabeled_train != plan.n_train_rows
    ):
        raise ValidationError(
            "Semi-supervised labeled/unlabeled counts must sum to positive train rows."
        )
    classes = getattr(plan.label_encoder_, "classes_", None)
    if classes is None or len(classes) < 2:
        raise ValidationError(
            "Semi-supervised plan requires a fitted multi-class vocabulary encoder."
        )
    # JSON preserves bool/int and string/numeric distinctions that Python's
    # equality alone would collapse, while numpy scalars become native values.
    encoded = classes.tolist()
    if json.dumps(encoded) != json.dumps(list(plan.classes_)):
        raise ValidationError("Semi-supervised label encoder and class vocabulary disagree.")
    model = plan.estimator_
    if not callable(getattr(model, "predict", None)):
        raise ValidationError("Semi-supervised plan requires a fitted predictive estimator.")
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_is_fitted

    if isinstance(model, BaseEstimator):
        try:
            check_is_fitted(model)
        except (ValueError, TypeError) as exc:
            raise ValidationError("Semi-supervised estimator is not fitted.") from exc
        codes = getattr(model, "classes_", None)
        if codes is not None and json.dumps(codes.tolist()) != json.dumps(
            list(range(len(classes)))
        ):
            raise ValidationError("Semi-supervised estimator class codes disagree with vocabulary.")
    width = getattr(model, "n_features_in_", None)
    if plan.modality == "tabular" and width is not None and width != len(plan.columns):
        raise ValidationError(
            "Semi-supervised feature columns disagree with fitted estimator width."
        )


def _validate_summary(plan: SemiSupervisedPlan, summary: Any) -> None:
    """Check present summary fields while retaining legacy summary-free bundles."""
    if summary is None:
        return
    if not isinstance(summary, dict):
        raise ValidationError("Semi-supervised bundle summary must be an object.")
    expected = plan.to_dict()
    for key in (
        "columns",
        "target_column",
        "method",
        "backend",
        "modality",
        "n_train_rows",
        "n_labeled_train",
        "n_unlabeled_train",
        "classes",
    ):
        if key in summary and json.dumps(summary[key]) != json.dumps(expected[key]):
            raise ValidationError(f"Semi-supervised summary {key!r} disagrees with fitted plan.")
