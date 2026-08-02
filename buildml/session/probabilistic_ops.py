"""Thin Session facades over buildml.probabilistic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.probabilistic.checkpoint import (
    load_probabilistic_bundle,
    save_probabilistic_bundle,
)
from buildml.probabilistic.evaluate import evaluate_probabilistic
from buildml.probabilistic.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    interval_result_summary,
    predict_result_summary,
)
from buildml.probabilistic.fit import fit_probabilistic
from buildml.probabilistic.predict import predict_interval, predict_probabilistic
from buildml.probabilistic.types import (
    IntervalMethod,
    ProbabilisticBackend,
    ProbabilisticEstimator,
    ProbabilisticTask,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_probabilistic_op(
    session,
    *,
    backend: str | None = None,
    estimator: ProbabilisticEstimator = "bayesian_ridge",
    task: ProbabilisticTask | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    alpha: float = 0.1,
    conformal: bool = True,
    conformal_calibration_fraction: float = 0.2,
    interval_method: IntervalMethod | None = None,
    prefer_reduce_components: bool = True,
    n_restarts_optimizer: int = 0,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
) -> Any:
    """Fit a Bayesian / probabilistic estimator on Session train.

    Notes
    -----
    **Backends:** ``native`` (sklearn + in-tree conformal), ``mapie`` and
    ``ngboost`` when ``buildml[probabilistic-industry]`` is installed.

    **Leakage:** Requires a split. Fit and optional split-conformal calibration
    use train only (conformal carve never touches validation/test). Honesty:
    uncertainty quantification for tabular estimators — not PyMC/Stan MCMC.
    Classical ``Session.calibration()`` is unchanged.
    """
    session.assert_can_fit("train")
    plan, result = fit_probabilistic(
        session.dataset,
        session._split_plan,
        backend=backend,  # type: ignore[arg-type]
        estimator=estimator,
        task=task,
        columns=columns,
        random_state=random_state,
        alpha=alpha,
        conformal=conformal,
        conformal_calibration_fraction=conformal_calibration_fraction,
        interval_method=interval_method,
        prefer_reduce_components=prefer_reduce_components,
        n_restarts_optimizer=n_restarts_optimizer,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._probabilistic_plan = plan
    session._probabilistic_fit_result = result
    session._probabilistic_eval_result = None
    session._probabilistic_predict_result = None
    session._probabilistic_interval_result = None
    session._record(
        "fit_probabilistic",
        {
            "backend": backend,
            "estimator": estimator,
            "task": task,
            "columns": columns,
            "random_state": random_state,
            "alpha": alpha,
            "conformal": conformal,
            "conformal_calibration_fraction": conformal_calibration_fraction,
            "interval_method": interval_method,
            "prefer_reduce_components": prefer_reduce_components,
            "n_restarts_optimizer": n_restarts_optimizer,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def evaluate_probabilistic_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    alpha: float | None = None,
) -> Any:
    """Evaluate the probabilistic plan on a holdout partition."""
    plan = getattr(session, "_probabilistic_plan", None)
    if plan is None:
        raise ValidationError("No probabilistic plan. Call fit_probabilistic(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_probabilistic(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        alpha=alpha,
    )
    session._probabilistic_eval_result = result
    session._record(
        "evaluate_probabilistic",
        {"partition": resolved, "alpha": alpha},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def predict_probabilistic_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    return_std: bool = True,
    return_proba: bool = True,
) -> Any:
    """Predict with the probabilistic estimator (no update)."""
    plan = getattr(session, "_probabilistic_plan", None)
    if plan is None:
        raise ValidationError("No probabilistic plan. Call fit_probabilistic(...) first.")
    result = predict_probabilistic(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        return_std=return_std,
        return_proba=return_proba,
    )
    session._probabilistic_predict_result = result
    session._record(
        "predict_probabilistic",
        {
            "partition": partition,
            "return_std": return_std,
            "return_proba": return_proba,
        },
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def predict_interval_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    alpha: float | None = None,
    method: str | None = None,
) -> Any:
    """Predictive intervals (regression) or conformal prediction sets (classification)."""
    plan = getattr(session, "_probabilistic_plan", None)
    if plan is None:
        raise ValidationError("No probabilistic plan. Call fit_probabilistic(...) first.")
    result = predict_interval(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        alpha=alpha,
        method=method,
    )
    session._probabilistic_interval_result = result
    session._record(
        "predict_interval",
        {"partition": partition, "alpha": alpha, "method": method},
        warnings=tuple(result.warnings),
        result_summary=interval_result_summary(result),
    )
    return result


def save_probabilistic_bundle_op(session, path: str | Path) -> Path:
    """Persist the active ProbabilisticPlan as ``buildml.probabilistic_bundle.v1``."""
    plan = getattr(session, "_probabilistic_plan", None)
    if plan is None:
        raise ValidationError("No probabilistic plan. Call fit_probabilistic(...) first.")
    out = save_probabilistic_bundle(
        path,
        plan,
        fit_result=getattr(session, "_probabilistic_fit_result", None),
        eval_result=getattr(session, "_probabilistic_eval_result", None),
    )
    session._record(
        "save_probabilistic_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.probabilistic_bundle.v1"},
    )
    return out


def load_probabilistic_bundle_op(session, path: str | Path):
    """Load a probabilistic bundle into this Session."""
    plan = load_probabilistic_bundle(path)
    session._probabilistic_plan = plan
    session._probabilistic_fit_result = None
    session._probabilistic_eval_result = None
    session._probabilistic_predict_result = None
    session._probabilistic_interval_result = None
    session._record(
        "load_probabilistic_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "estimator_name": plan.estimator_name,
            "task": plan.task,
        },
    )
    return session
