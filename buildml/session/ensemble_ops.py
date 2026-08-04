"""Thin Session facades over buildml.ensemble."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
    """Fit a voting ensemble on the train partition only.

    Delegates to :func:`buildml.ensemble.fit.fit_voting_ensemble`, stores the
    plan on Session, and sets ``fit_result`` so classical evaluate/predict work.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan.
    estimators:
        Base estimators as a mapping or ``(name, estimator)`` sequence.
    voting:
        Voting strategy (``hard`` or ``soft`` for classifiers).
    weights:
        Optional per-estimator vote weights.
    task:
        Task type override (``classification``, ``regression``, or ``auto``).

    Returns
    -------
    EnsembleFitResult
        Serializable fit summary including base estimator names.

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
    """Fit a stacking ensemble on the train partition only.

    Delegates to :func:`buildml.ensemble.fit.fit_stacking_ensemble` with
    out-of-fold meta features computed inside train only.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan.
    estimators:
        Base estimators as a mapping or ``(name, estimator)`` sequence.
    final_estimator:
        Meta-learner fitted on out-of-fold base predictions.
    cv:
        Number of cross-validation folds inside train for OOF features.
    passthrough:
        When True, include original features in meta-learner input.
    stack_method:
        Base prediction method (``auto``, ``predict_proba``, etc.).
    task:
        Task type override (``classification``, ``regression``, or ``auto``).

    Returns
    -------
    EnsembleFitResult
        Serializable fit summary including CV fold count and base names.

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

    Delegates to :func:`buildml.ensemble.fit.fit_blending_ensemble` with a
    holdout carved from train for meta-learner fitting.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan.
    estimators:
        Base estimators as a mapping or ``(name, estimator)`` sequence.
    final_estimator:
        Meta-learner fitted on holdout base predictions.
    holdout_fraction:
        Fraction of train rows reserved for blend holdout.
    blend_method:
        Base prediction method for blending (``predict_proba``, etc.).
    random_state:
        Seed for holdout split and base estimator initialization.
    refit_bases_on_full_train:
        When True, refit base estimators on all train rows after blending.
    passthrough:
        When True, include original features in meta-learner input.
    task:
        Task type override (``classification``, ``regression``, or ``auto``).

    Returns
    -------
    EnsembleFitResult
        Serializable fit summary including holdout fraction disclosures.

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

    Delegates to the same metric path as ``Session.evaluate``. Requires a
    prior :func:`fit_voting`, :func:`fit_stacking`, or :func:`fit_blending`.

    Parameters
    ----------
    session:
        Active Session with ``fit_result`` from an ensemble fit.
    partition:
        Partition to evaluate (``train``, ``validation``, or ``test``).

    Returns
    -------
    EvaluateResult
        Classical metrics plus ensemble strategy disclosures. Diagnostics also
        include ``base_contributions``, ``diversity``, and ``ensemble_report``
        (predict-only scoring of train-fitted bases on ``partition``).

    Raises
    ------
    ValidationError
        When no fitted ensemble exists on the Session.
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
        from buildml.ensemble.evaluate import build_ensemble_eval_report

        report = build_ensemble_eval_report(
            session.dataset,
            session._split_plan,
            session._fit_result,
            plan,
            partition=partition,
            ensemble_metrics=dict(result.metrics),
        )
        result.diagnostics["ensemble_report"] = report.to_dict()
        result.diagnostics["base_contributions"] = [
            c.to_dict() for c in report.base_contributions
        ]
        result.diagnostics["diversity"] = (
            None if report.diversity is None else report.diversity.to_dict()
        )
        for note in report.disclosures[:2]:
            if note not in result.recommendations:
                result.recommendations.append(note)
        if report.diversity is not None and report.diversity.mean_pairwise_disagreement is not None:
            tips_div = (
                f"Base diversity mean_pairwise_disagreement="
                f"{report.diversity.mean_pairwise_disagreement:.4f}."
            )
            result.recommendations.append(tips_div)
    session._record(
        "evaluate_ensemble",
        {"partition": partition, "strategy": None if plan is None else plan.strategy},
        result_summary=result.to_dict(),
    )
    return result


def save_ensemble_bundle_op(session, path: str | Path) -> Path:
    """Persist the active EnsemblePlan as ``buildml.ensemble_bundle.v1``.

    Delegates to :func:`buildml.ensemble.checkpoint.save_ensemble_bundle`.
    Reload with :func:`load_ensemble_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with an ensemble plan from a prior fit.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no ensemble plan exists on the Session.
    """
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


def load_ensemble_bundle_op(session, path: str | Path, *, trusted: bool = False) -> Any:
    """Load an ensemble bundle into this Session.

    Delegates to :func:`buildml.ensemble.checkpoint.load_ensemble_bundle`
    and restores ``fit_result`` for classical evaluate/predict.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded EnsemblePlan.
    path:
        Path to a ``buildml.ensemble_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with EnsemblePlan and ``fit_result`` attached.
    """
    plan, fit_result = load_ensemble_bundle(path, trusted=trusted)
    session._ensemble_plan = plan
    session._ensemble_fit_result = None
    session._fit_result = fit_result
    session._record(
        "load_ensemble_bundle",
        {"path": str(path), "strategy": plan.strategy},
        result_summary=plan.to_dict(),
    )
    return cast("Session", session)