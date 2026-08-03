"""Thin Session facades over buildml.probabilistic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
    """Fit a Bayesian or probabilistic estimator on Session train only.

    Delegates to :func:`buildml.probabilistic.fit.fit_probabilistic`, stores
    the :class:`~buildml.probabilistic.results.ProbabilisticPlan` on Session,
    and records the fit. Follow with :func:`predict_probabilistic_op` or
    :func:`predict_interval_op`.

    Parameters
    ----------
    session:
        Active Session with dataset, split plan, and target column.
    backend:
        Optional backend override (``native``, ``mapie``, ``ngboost``).
    estimator:
        Probabilistic estimator key (``bayesian_ridge``, etc.).
    task:
        Task override; inferred from target when ``None``.
    columns:
        Explicit feature columns; ``None`` auto-selects numerics.
    random_state:
        Seed for stochastic steps.
    alpha:
        Significance level for intervals (e.g. 0.1 for 90% intervals).
    conformal:
        When True, apply split-conformal calibration on train carve-out.
    conformal_calibration_fraction:
        Fraction of train reserved for conformal calibration.
    interval_method:
        Interval construction method override.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    n_restarts_optimizer:
        Restarts for Bayesian ridge optimizer.
    n_estimators:
        Tree count for NGBoost backend.
    learning_rate:
        Learning rate for NGBoost backend.

    Returns
    -------
    ProbabilisticFitResult
        Serializable fit summary including backend and conformal disclosures.

    Notes
    -----
    **Backends:** ``native`` (sklearn + in-tree conformal), ``mapie`` and
    ``ngboost`` when ``buildml[probabilistic-industry]`` is installed.

    **Leakage:** Requires a split. Fit and optional split-conformal calibration
    use train only (conformal carve never touches validation/test). Honesty:
    uncertainty quantification for tabular estimators: not PyMC/Stan MCMC.
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
    """Evaluate the probabilistic plan on a holdout partition.

    Delegates to :func:`buildml.probabilistic.evaluate.evaluate_probabilistic`
    for calibration and interval coverage metrics. Falls back to ``test`` when
    no validation partition exists.

    Parameters
    ----------
    session:
        Active Session with a ProbabilisticPlan from :func:`fit_probabilistic_op`.
    partition:
        Holdout partition (default ``validation``).
    alpha:
        Significance level override for interval metrics.

    Returns
    -------
    ProbabilisticEvalResult
        Calibration, coverage, and sharpness metrics on the partition.

    Raises
    ------
    ValidationError
        When no probabilistic plan exists on the Session.
    """
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
    """Predict with the probabilistic estimator without updating the plan.

    Delegates to :func:`buildml.probabilistic.predict.predict_probabilistic`
    and optionally returns standard deviations or class probabilities.

    Parameters
    ----------
    session:
        Active Session with a ProbabilisticPlan from :func:`fit_probabilistic_op`.
    partition:
        Split partition to predict on (default ``test``).
    return_std:
        When True, include predictive standard deviations (regression).
    return_proba:
        When True, include class probabilities (classification).

    Returns
    -------
    ProbabilisticPredictResult
        Point predictions with optional uncertainty outputs.

    Raises
    ------
    ValidationError
        When no probabilistic plan exists on the Session.
    """
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
    """Predict predictive intervals or conformal prediction sets on a partition.

    Delegates to :func:`buildml.probabilistic.predict.predict_interval`.
    Regression returns lower/upper bounds; classification returns prediction sets.

    Parameters
    ----------
    session:
        Active Session with a ProbabilisticPlan from :func:`fit_probabilistic_op`.
    partition:
        Split partition to score (default ``test``).
    alpha:
        Significance level override for interval width.
    method:
        Interval method override (conformal, native, etc.).

    Returns
    -------
    ProbabilisticIntervalResult
        Interval bounds or conformal sets per row.

    Raises
    ------
    ValidationError
        When no probabilistic plan exists on the Session.
    """
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
    """Persist the active ProbabilisticPlan as ``buildml.probabilistic_bundle.v1``.

    Delegates to :func:`buildml.probabilistic.checkpoint.save_probabilistic_bundle`.
    Reload with :func:`load_probabilistic_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a ProbabilisticPlan from :func:`fit_probabilistic_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no probabilistic plan exists on the Session.
    """
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


def load_probabilistic_bundle_op(session, path: str | Path, *, trusted: bool = False):
    """Load a probabilistic bundle into this Session.

    Delegates to :func:`buildml.probabilistic.checkpoint.load_probabilistic_bundle`
    and clears prior eval/predict/interval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded ProbabilisticPlan.
    path:
        Path to a ``buildml.probabilistic_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with ProbabilisticPlan attached for chaining.
    """
    plan = load_probabilistic_bundle(path, trusted=trusted)
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
    return cast("Session", session)