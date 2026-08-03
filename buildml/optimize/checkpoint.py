"""Decision-policy bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.optimize.results import (
    ApplyDecisionsResult,
    DecisionEvalResult,
    DecisionFitResult,
    DecisionPlan,
)
from buildml.optimize.types import CostModel

BUNDLE_FORMAT = "buildml.decision_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Decision bundles and Session checkpoints are complementary, not "
    "interchangeable. A decision bundle (buildml.decision_bundle.v1) stores a "
    "fitted DecisionPlan (threshold / cost matrix / allocation rules). A "
    "Session checkpoint stores data, roles, splits, history, and optionally a "
    "classical FitResult; it does not embed the DecisionPlan. Reload tabular "
    "workflow via checkpoint_load; reload the policy via load_decision_bundle. "
    "Honesty: ML score/cost/allocation decision helpers — not a general OR "
    "platform. Cross-link: classical Session.tune_threshold remains the "
    "diagnostic explorer; fit_decision_policy(method='threshold') persists "
    "the chosen operating point."
)


def save_decision_bundle(
    path: str | Path,
    plan: DecisionPlan,
    *,
    fit_result: DecisionFitResult | None = None,
    eval_result: DecisionEvalResult | None = None,
    apply_result: ApplyDecisionsResult | None = None,
) -> Path:
    """Write a decision-policy bundle directory (``buildml.decision_bundle.v1``).

    Persists the fitted :class:`~buildml.optimize.results.DecisionPlan`,
    optional fit/eval/apply summaries, and auxiliary estimator objects to
    disk. Distinct from Session checkpoints — reload policies via
    :func:`~buildml.optimize.checkpoint.load_decision_bundle`.

    Parameters
    ----------
    path:
        Destination directory; created when missing.
    plan:
        Fitted decision policy to serialise.
    fit_result:
        Optional fit summary stored in ``meta.json``.
    eval_result:
        Optional holdout evaluation summary stored in ``meta.json``.
    apply_result:
        Optional apply summary stored in ``meta.json``.

    Returns
    -------
    pathlib.Path
        Resolved bundle directory containing ``decision_plan.joblib`` and
        ``meta.json``.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None`` or serialisation inputs are invalid.
    """
    if plan is None:
        raise ValidationError("No DecisionPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "plan": plan,
        "cost_matrix": None if plan.cost_matrix_ is None else np.asarray(plan.cost_matrix_),
        "aux_estimator": getattr(plan, "aux_estimator_", None),
    }
    joblib.dump(payload, destination / "decision_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "apply": None if apply_result is None else apply_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_decision_bundle(path: str | Path, *, trusted: bool = False) -> DecisionPlan:
    """Load a decision-policy bundle into a :class:`DecisionPlan`.

    Reads ``meta.json`` and ``decision_plan.joblib`` produced by
    :func:`~buildml.optimize.checkpoint.save_decision_bundle`, rebuilds cost
    matrix and auxiliary estimator fields when needed, and returns a plan
    ready for :func:`~buildml.optimize.apply.apply_decisions`.

    Parameters
    ----------
    path:
        Bundle directory written by :func:`save_decision_bundle`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    DecisionPlan
        Rehydrated policy with cost matrix and auxiliary estimator restored
        when present in the bundle.

    Raises
    ------
    ValidationError
        When files are missing, the format version is unsupported, or the
        payload does not contain a valid :class:`DecisionPlan`.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "decision_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete decision bundle at {root}. "
            f"Expected meta.json and decision_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported decision bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, DecisionPlan):
        plan = loaded
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        if not isinstance(plan, DecisionPlan):
            raise ValidationError("Loaded plan object is not a DecisionPlan")
        if loaded.get("cost_matrix") is not None and plan.cost_matrix_ is None:
            plan.cost_matrix_ = np.asarray(loaded["cost_matrix"], dtype=float)
        if loaded.get("aux_estimator") is not None and plan.aux_estimator_ is None:
            plan.aux_estimator_ = loaded["aux_estimator"]
    else:
        raise ValidationError(
            "decision_plan.joblib must contain a DecisionPlan or a payload "
            "with key 'plan'."
        )

    # Rebuild cost_matrix_ from cost_model when needed
    if plan.cost_matrix_ is None and plan.cost_model is not None:
        if plan.cost_model.matrix is not None:
            plan.cost_matrix_ = np.asarray(plan.cost_model.matrix, dtype=float)
    if plan.cost_model is None and isinstance(meta.get("plan"), dict):
        cm = meta["plan"].get("cost_model")
        if isinstance(cm, dict):
            plan.cost_model = CostModel(
                kind=cm.get("kind", "binary_expected_cost"),
                fp_cost=cm.get("fp_cost"),
                fn_cost=cm.get("fn_cost"),
                tp_benefit=float(cm.get("tp_benefit") or 0.0),
                tn_benefit=float(cm.get("tn_benefit") or 0.0),
                matrix=cm.get("matrix"),
                class_labels=tuple(cm.get("class_labels") or ()),
                formula=str(cm.get("formula") or ""),
                extras=dict(cm.get("extras") or {}),
            )
            if plan.cost_matrix_ is None and plan.cost_model.matrix is not None:
                plan.cost_matrix_ = np.asarray(plan.cost_model.matrix, dtype=float)
    return plan
