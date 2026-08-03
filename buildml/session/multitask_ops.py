"""Thin Session facades over buildml.multitask."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.multitask.checkpoint import load_multitask_bundle, save_multitask_bundle
from buildml.multitask.evaluate import evaluate_multitask
from buildml.multitask.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
)
from buildml.multitask.fit import fit_multitask
from buildml.multitask.predict import predict_multitask
from buildml.multitask.types import (
    MultiTaskBackend,
    MultiTaskBaseEstimator,
    MultiTaskMethod,
    MultiTaskTask,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_multitask_op(
    session,
    *,
    backend: MultiTaskBackend | None = None,
    method: MultiTaskMethod = "multi_output",
    task: MultiTaskTask = "auto",
    targets: Sequence[str] | None = None,
    columns: list[str] | None = None,
    base_estimator: MultiTaskBaseEstimator | str = "logistic_regression",
    random_state: int | None = 0,
    order: Sequence[str] | None = None,
    prefer_reduce_components: bool = True,
    prediction_prefix: str = "multitask_pred",
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> Any:
    """Fit a multi-target estimator on the train partition only.

    Delegates to :func:`buildml.multitask.fit.fit_multitask`, stores the
    :class:`~buildml.multitask.results.MultiTaskPlan` on Session, and records
    the fit. Follow with :func:`predict_multitask_op` or
    :func:`evaluate_multitask_op`.

    Parameters
    ----------
    session:
        Active Session with dataset, split plan, and at least two targets.
    backend:
        Optional backend override (``sklearn``, ``industry``, ``torch``).
    method:
        Multi-task strategy (``multi_output``, ``chain``, ``torch_multihead``).
    task:
        Task mix (``auto``, ``classification``, ``regression``, ``mixed``).
    targets:
        Optional explicit target column names (roles or list).
    columns:
        Optional explicit feature columns.
    base_estimator:
        Base estimator key for sklearn/industry backends.
    random_state:
        Seed for stochastic steps.
    order:
        Optional target column order for chained strategies.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    prediction_prefix:
        Prefix for attached prediction column names.
    epochs:
        Training epochs for torch multi-head backend.
    batch_size:
        Minibatch size for torch backend.
    learning_rate:
        Optimizer learning rate for torch backend.
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    MultiTaskFitResult
        Serializable fit summary per target and backend disclosures.

    Notes
    -----
    **Leakage:** Requires a split. Fit uses train only. Validation/test are
    never used for fitting. Needs ``>= 2`` target columns (roles or
    ``targets=``). Sklearn/industry require same-type tasks; torch supports
    mixed cls+reg. Classical ``Session.fit`` remains single-target.
    """
    session.assert_can_fit("train")
    plan, result = fit_multitask(
        session.dataset,
        session._split_plan,
        backend=backend,
        method=method,
        task=task,
        targets=targets,
        columns=columns,
        base_estimator=base_estimator,
        random_state=random_state,
        order=order,
        prefer_reduce_components=prefer_reduce_components,
        prediction_prefix=prediction_prefix,
        reduce_plan=getattr(session, "_reduce_plan", None),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    session._multitask_plan = plan
    session._multitask_fit_result = result
    session._multitask_predict_result = None
    session._multitask_eval_result = None
    session._record(
        "fit_multitask",
        {
            "backend": backend,
            "method": method,
            "task": task,
            "targets": None if targets is None else list(targets),
            "columns": columns,
            "base_estimator": base_estimator,
            "random_state": random_state,
            "order": None if order is None else list(order),
            "prefer_reduce_components": prefer_reduce_components,
            "prediction_prefix": prediction_prefix,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def predict_multitask_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
    prediction_prefix: str | None = None,
) -> Any:
    """Predict all targets with the frozen multi-task plan without refitting.

    Delegates to :func:`buildml.multitask.predict.predict_multitask`. When
    ``attach=True``, prediction columns are merged into Session dataset.

    Parameters
    ----------
    session:
        Active Session with a multi-task plan from :func:`fit_multitask_op`.
    partition:
        Partition to score (``train``, ``validation``, ``test``, or ``all``).
    attach:
        When True, attach prediction columns to the Session dataset frame.
    prediction_prefix:
        Optional override for attached column name prefix.

    Returns
    -------
    MultiTaskPredictResult
        Per-target predictions and optional attached column metadata.

    Raises
    ------
    ValidationError
        When no multi-task plan exists on the Session.
    """
    plan = getattr(session, "_multitask_plan", None)
    if plan is None:
        raise ValidationError("No multi-task plan. Call fit_multitask(...) first.")
    new_dataset, result = predict_multitask(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        attach=attach,
        prediction_prefix=prediction_prefix,
    )
    if new_dataset is not None:
        session._dataset = new_dataset
    session._multitask_predict_result = result
    session._record(
        "predict_multitask",
        {
            "partition": partition,
            "attach": attach,
            "prediction_prefix": prediction_prefix,
        },
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def evaluate_multitask_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Evaluate the multi-task plan on a holdout partition without refitting.

    Delegates to :func:`buildml.multitask.evaluate.evaluate_multitask`.
    Holdout partitions are never used during fit.

    Parameters
    ----------
    session:
        Active Session with a multi-task plan from :func:`fit_multitask_op`.
    partition:
        Holdout partition to score. Validation falls back to test when absent.

    Returns
    -------
    MultiTaskEvalResult
        Per-target and aggregated holdout metrics.

    Raises
    ------
    ValidationError
        When no multi-task plan exists on the Session.
    """
    plan = getattr(session, "_multitask_plan", None)
    if plan is None:
        raise ValidationError("No multi-task plan. Call fit_multitask(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_multitask(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
    )
    session._multitask_eval_result = result
    session._record(
        "evaluate_multitask",
        {"partition": resolved},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_multitask_bundle_op(session, path: str | Path) -> Path:
    """Persist the active multi-task plan as ``buildml.multitask_bundle.v1``.

    Delegates to :func:`buildml.multitask.checkpoint.save_multitask_bundle`.
    Reload with :func:`load_multitask_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a multi-task plan from :func:`fit_multitask_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no multi-task plan exists on the Session.
    """
    plan = getattr(session, "_multitask_plan", None)
    if plan is None:
        raise ValidationError("No multi-task plan. Call fit_multitask(...) first.")
    out = save_multitask_bundle(
        path,
        plan,
        fit_result=getattr(session, "_multitask_fit_result", None),
        eval_result=getattr(session, "_multitask_eval_result", None),
    )
    session._record(
        "save_multitask_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "backend": plan.backend,
            "method": plan.method,
            "task": plan.task,
            "n_tasks": len(plan.target_columns),
            "target_columns": list(plan.target_columns),
        },
    )
    return out


def load_multitask_bundle_op(session, path: str | Path) -> Any:
    """Load a multi-task bundle into this Session.

    Delegates to :func:`buildml.multitask.checkpoint.load_multitask_bundle`
    and clears prior fit/eval/predict results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded multi-task plan.
    path:
        Path to a ``buildml.multitask_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with multi-task plan attached for chaining.
    """
    plan = load_multitask_bundle(path)
    session._multitask_plan = plan
    session._multitask_fit_result = None
    session._multitask_predict_result = None
    session._multitask_eval_result = None
    session._record(
        "load_multitask_bundle",
        {
            "path": str(path),
            "backend": plan.backend,
            "method": plan.method,
            "task": plan.task,
            "n_tasks": len(plan.target_columns),
        },
        result_summary=plan.to_dict(),
    )
    return session
