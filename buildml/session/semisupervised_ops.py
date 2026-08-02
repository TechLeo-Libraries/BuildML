"""Thin Session facades over buildml.semisupervised."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.semisupervised.checkpoint import (
    load_semisupervised_bundle,
    save_semisupervised_bundle,
)
from buildml.semisupervised.evaluate import evaluate_semisupervised
from buildml.semisupervised.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
)
from buildml.semisupervised.fit import fit_semisupervised
from buildml.semisupervised.predict import predict_semisupervised
from buildml.semisupervised.types import SemiSupervisedMethod

PartitionOrAll = PartitionName | Literal["all"]


def fit_semisupervised_op(
    session,
    *,
    method: SemiSupervisedMethod = "label_propagation",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    kernel: str = "knn",
    n_neighbors: int = 7,
    max_iter: int = 1000,
    alpha: float = 0.2,
    base_estimator: str = "logistic_regression",
    threshold: float = 0.75,
    criterion: str = "threshold",
    k_best: int = 10,
    max_self_train_iter: int = 10,
    unlabeled_marker: Any = None,
    prefer_reduce_components: bool = True,
) -> Any:
    """Fit a semi-supervised classifier on the train partition only.

    Notes
    -----
    **Leakage:** Requires a split. Fit uses train only. Unlabeled rows are
    target NaNs (default). Validation/test never invent labels for selection.
    """
    session.assert_can_fit("train")
    plan, result = fit_semisupervised(
        session.dataset,
        session._split_plan,
        method=method,
        columns=columns,
        random_state=random_state,
        kernel=kernel,
        n_neighbors=n_neighbors,
        max_iter=max_iter,
        alpha=alpha,
        base_estimator=base_estimator,
        threshold=threshold,
        criterion=criterion,
        k_best=k_best,
        max_self_train_iter=max_self_train_iter,
        unlabeled_marker=unlabeled_marker,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._semisupervised_plan = plan
    session._semisupervised_fit_result = result
    session._semisupervised_predict_result = None
    session._semisupervised_eval_result = None
    session._record(
        "fit_semisupervised",
        {
            "method": method,
            "columns": columns,
            "kernel": kernel,
            "n_neighbors": n_neighbors,
            "max_iter": max_iter,
            "alpha": alpha,
            "base_estimator": base_estimator,
            "threshold": threshold,
            "criterion": criterion,
            "k_best": k_best,
            "max_self_train_iter": max_self_train_iter,
            "unlabeled_marker": unlabeled_marker,
            "prefer_reduce_components": prefer_reduce_components,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def predict_semisupervised_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
    prediction_column: str = "semisupervised_prediction",
) -> Any:
    """Predict with the train-fitted semi-supervised plan (no refit)."""
    plan = getattr(session, "_semisupervised_plan", None)
    if plan is None:
        raise ValidationError("No semi-supervised plan. Call fit_semisupervised(...) first.")
    new_dataset, result = predict_semisupervised(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        attach=attach,
        prediction_column=prediction_column,
    )
    if new_dataset is not None:
        session._dataset = new_dataset
    session._semisupervised_predict_result = result
    session._record(
        "predict_semisupervised",
        {
            "partition": partition,
            "attach": attach,
            "prediction_column": prediction_column,
        },
        result_summary=predict_result_summary(result),
    )
    return result


def evaluate_semisupervised_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> Any:
    """Evaluate the last semi-supervised plan on labeled rows of a partition."""
    plan = getattr(session, "_semisupervised_plan", None)
    if plan is None:
        raise ValidationError("No semi-supervised plan. Call fit_semisupervised(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_semisupervised(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        unlabeled_marker=unlabeled_marker,
    )
    session._semisupervised_eval_result = result
    session._record(
        "evaluate_semisupervised",
        {"partition": resolved, "unlabeled_marker": unlabeled_marker},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_semisupervised_bundle_op(session, path: str | Path) -> Path:
    """Persist the active SemiSupervisedPlan as ``buildml.semisupervised_bundle.v1``."""
    plan = getattr(session, "_semisupervised_plan", None)
    if plan is None:
        raise ValidationError("No semi-supervised plan. Call fit_semisupervised(...) first.")
    out = save_semisupervised_bundle(
        path,
        plan,
        fit_result=getattr(session, "_semisupervised_fit_result", None),
        eval_result=getattr(session, "_semisupervised_eval_result", None),
    )
    session._record(
        "save_semisupervised_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "method": plan.method,
            "n_labeled_train": plan.n_labeled_train,
            "n_unlabeled_train": plan.n_unlabeled_train,
        },
    )
    return out


def load_semisupervised_bundle_op(session, path: str | Path) -> Any:
    """Load a semi-supervised bundle into this Session."""
    plan = load_semisupervised_bundle(path)
    session._semisupervised_plan = plan
    session._semisupervised_fit_result = None
    session._semisupervised_predict_result = None
    session._semisupervised_eval_result = None
    session._record(
        "load_semisupervised_bundle",
        {"path": str(path), "method": plan.method},
        result_summary=plan.to_dict(),
    )
    return session
