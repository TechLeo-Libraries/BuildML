"""Thin Session facades over buildml.ensemble."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from buildml.core.errors import ValidationError
from buildml.ensemble.checkpoint import load_ensemble_bundle, save_ensemble_bundle
from buildml.ensemble.explain_hooks import fit_result_summary
from buildml.ensemble.fit import (
    fit_blending_ensemble,
    fit_stacking_ensemble,
    fit_voting_ensemble,
)
from buildml.ensemble.types import BlendMethod, VotingMethod
from buildml.model.supervised import EvaluateResult, evaluate_estimator

TaskType = Literal["classification", "regression", "auto"]
EstimatorMap = Mapping[str, Any] | Sequence[tuple[str, Any]]


def _attach_ensemble(session, plan, result, fit_result, operation: str, params: dict[str, Any]):
    session._ensemble_plan = plan
    session._ensemble_fit_result = result
    session._fit_result = fit_result
    session._record(
        operation,
        params,
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def fit_voting(
    session,
    estimators: EstimatorMap,
    *,
    voting: VotingMethod = "hard",
    weights: Sequence[float] | None = None,
    task: TaskType = "auto",
) -> Any:
    """Fit a VotingClassifier / VotingRegressor on the train partition only.

    Notes
    -----
    **Leakage:** Requires a split. Fits on train only. Sets Session ``fit_result``
    so classical ``evaluate`` / ``predict`` / ``save_pipeline`` work.
    """
    session.assert_can_fit("train")
    plan, result, fit_result = fit_voting_ensemble(
        session.dataset,
        session._split_plan,
        estimators,
        voting=voting,
        weights=weights,
        task=task,
    )
    return _attach_ensemble(
        session,
        plan,
        result,
        fit_result,
        "fit_voting",
        {
            "strategy": "voting",
            "voting": voting,
            "weights": None if weights is None else list(weights),
            "task": task,
            "estimator_names": list(result.estimator_names),
        },
    )


def fit_stacking(
    session,
    estimators: EstimatorMap,
    *,
    final_estimator: Any | None = None,
    cv: int = 5,
    passthrough: bool = False,
    stack_method: str = "auto",
    task: TaskType = "auto",
) -> Any:
    """Fit a StackingClassifier / StackingRegressor on the train partition only.

    Notes
    -----
    **Leakage:** Stacking CV folds stay inside train. Session test is never used
    for out-of-fold meta features.
    """
    session.assert_can_fit("train")
    plan, result, fit_result = fit_stacking_ensemble(
        session.dataset,
        session._split_plan,
        estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        stack_method=stack_method,
        task=task,
    )
    return _attach_ensemble(
        session,
        plan,
        result,
        fit_result,
        "fit_stacking",
        {
            "strategy": "stacking",
            "cv": cv,
            "passthrough": passthrough,
            "stack_method": stack_method,
            "task": task,
            "final_estimator": None
            if final_estimator is None
            else type(final_estimator).__name__,
            "estimator_names": list(result.estimator_names),
        },
    )


def fit_blending(
    session,
    estimators: EstimatorMap,
    *,
    final_estimator: Any | None = None,
    holdout_fraction: float = 0.2,
    blend_method: BlendMethod = "predict_proba",
    random_state: int | None = 0,
    refit_bases_on_full_train: bool = True,
    passthrough: bool = False,
    task: TaskType = "auto",
) -> Any:
    """Fit a holdout-blend ensemble on the train partition only.

    Notes
    -----
    **Leakage:** The blend holdout is carved from train. Session validation/test
    never enter meta-learner fitting. Prefer stacking when you want CV OOF
    meta features instead of a single holdout.
    """
    session.assert_can_fit("train")
    plan, result, fit_result = fit_blending_ensemble(
        session.dataset,
        session._split_plan,
        estimators,
        final_estimator=final_estimator,
        holdout_fraction=holdout_fraction,
        blend_method=blend_method,
        random_state=random_state,
        refit_bases_on_full_train=refit_bases_on_full_train,
        passthrough=passthrough,
        task=task,
    )
    return _attach_ensemble(
        session,
        plan,
        result,
        fit_result,
        "fit_blending",
        {
            "strategy": "blending",
            "holdout_fraction": holdout_fraction,
            "blend_method": blend_method,
            "random_state": random_state,
            "refit_bases_on_full_train": refit_bases_on_full_train,
            "passthrough": passthrough,
            "task": task,
            "final_estimator": None
            if final_estimator is None
            else type(final_estimator).__name__,
            "estimator_names": list(result.estimator_names),
        },
    )


def evaluate_ensemble(
    session,
    *,
    partition: Literal["train", "validation", "test"] = "test",
) -> EvaluateResult:
    """Evaluate the last native ensemble with classical supervised metrics.

    Requires ``fit_voting`` / ``fit_stacking`` / ``fit_blending`` (or a loaded
    ensemble bundle). Delegates to the same metric path as ``Session.evaluate``.
    """
    if session._fit_result is None:
        raise ValidationError(
            "No fitted ensemble. Call fit_voting / fit_stacking / fit_blending first."
        )
    plan = getattr(session, "_ensemble_plan", None)
    result = evaluate_estimator(
        session.dataset, session._split_plan, session._fit_result, partition=partition
    )
    if plan is not None:
        tips = list(result.recommendations)
        tips.insert(
            0,
            f"Ensemble strategy={plan.strategy}; bases={list(plan.estimator_names)}.",
        )
        for note in plan.disclosures[:3]:
            tips.append(note)
        result.recommendations = tips
        result.diagnostics["ensemble"] = plan.to_dict()
    session._record(
        "evaluate_ensemble",
        {"partition": partition, "strategy": None if plan is None else plan.strategy},
        result_summary=result.to_dict(),
    )
    return result


def save_ensemble_bundle_op(session, path: str | Path) -> Path:
    """Persist the active EnsemblePlan as ``buildml.ensemble_bundle.v1``."""
    plan = getattr(session, "_ensemble_plan", None)
    if plan is None:
        raise ValidationError(
            "No ensemble plan. Call fit_voting / fit_stacking / fit_blending first."
        )
    out = save_ensemble_bundle(
        path,
        plan,
        fit_result=getattr(session, "_fit_result", None),
        ensemble_fit_result=getattr(session, "_ensemble_fit_result", None),
    )
    session._record(
        "save_ensemble_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "strategy": plan.strategy,
            "estimator_names": list(plan.estimator_names),
        },
    )
    return out


def load_ensemble_bundle_op(session, path: str | Path) -> Any:
    """Load an ensemble bundle into this Session."""
    plan, fit_result = load_ensemble_bundle(path)
    session._ensemble_plan = plan
    session._ensemble_fit_result = None
    session._fit_result = fit_result
    session._record(
        "load_ensemble_bundle",
        {"path": str(path), "strategy": plan.strategy},
        result_summary=plan.to_dict(),
    )
    return session
