"""Thin Session facades over buildml.semisupervised."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
from buildml.semisupervised.types import SemiSupervisedBackend, SemiSupervisedMethod

PartitionOrAll = PartitionName | Literal["all"]


def fit_semisupervised_op(
    session,
    *,
    backend: SemiSupervisedBackend | None = None,
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
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    consistency_weight: float = 1.0,
    mixup_alpha: float = 0.75,
    device: str = "cpu",
    text_column: str | None = None,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    unlabeled_marker: Any = None,
    prefer_reduce_components: bool = True,
) -> Any:
    """Fit a semi-supervised classifier on labeled and unlabeled train rows.

    Delegates to :func:`buildml.semisupervised.fit.fit_semisupervised`, stores
    the :class:`~buildml.semisupervised.results.SemiSupervisedPlan` on Session,
    and records the fit. Follow with :func:`predict_semisupervised_op` or
    :func:`evaluate_semisupervised_op`.

    Parameters
    ----------
    session:
        Active Session with dataset, split plan, and labeled/unlabeled train rows.
    backend:
        Optional backend override (``sklearn``, ``industry``, ``torch``, ``text``).
    method:
        Semi-supervised method key (``label_propagation``, ``self_training``, etc.).
    columns:
        Optional explicit feature columns for tabular backends.
    random_state:
        Seed for stochastic steps.
    kernel:
        Kernel or affinity type for graph-based methods.
    n_neighbors:
        Neighborhood size for kNN graph construction.
    max_iter:
        Maximum iterations for iterative label propagation methods.
    alpha:
        Clamping factor for label propagation (labeled vs propagated mass).
    base_estimator:
        Base classifier for self-training and pseudo-label methods.
    threshold:
        Confidence threshold for pseudo-label acceptance.
    criterion:
        Pseudo-label selection criterion (``threshold`` or ``k_best``).
    k_best:
        Top-k pseudo-labels per iteration when ``criterion='k_best'``.
    max_self_train_iter:
        Maximum self-training rounds for pseudo-label methods.
    epochs:
        Training epochs for torch consistency-regularization backend.
    batch_size:
        Minibatch size for torch backend.
    learning_rate:
        Optimizer learning rate for torch backend.
    consistency_weight:
        Weight on unlabeled consistency loss for torch backend.
    mixup_alpha:
        Mixup alpha for torch consistency backend.
    device:
        Torch device string (``cpu`` or ``cuda``).
    text_column:
        Text column for HF embedding semi-supervised backend.
    text_model_name:
        Sentence-transformer model name for text backend.
    unlabeled_marker:
        Value treated as unlabeled in the train target column.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.

    Returns
    -------
    SemiSupervisedFitResult
        Serializable fit summary including labeled/unlabeled train counts.

    Notes
    -----
    **Leakage:** Requires a split. Fit uses train only. Unlabeled rows are
    target NaNs (default). Validation/test never invent labels for selection.
    """
    session.assert_can_fit("train")
    plan, result = fit_semisupervised(
        session.dataset,
        session._split_plan,
        backend=backend,
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
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        consistency_weight=consistency_weight,
        mixup_alpha=mixup_alpha,
        device=device,
        text_column=text_column,
        text_model_name=text_model_name,
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
            "backend": backend,
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
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "consistency_weight": consistency_weight,
            "mixup_alpha": mixup_alpha,
            "device": device,
            "text_column": text_column,
            "text_model_name": text_model_name,
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
    """Predict with the train-fitted semi-supervised plan without refitting.

    Delegates to :func:`buildml.semisupervised.predict.predict_semisupervised`.
    When ``attach=True``, predictions are merged into Session dataset.

    Parameters
    ----------
    session:
        Active Session with a semi-supervised plan from
        :func:`fit_semisupervised_op`.
    partition:
        Partition to score (``train``, ``validation``, ``test``, or ``all``).
    attach:
        When True, attach prediction column to the Session dataset frame.
    prediction_column:
        Column name used when ``attach=True``.

    Returns
    -------
    SemiSupervisedPredictResult
        Predictions and optional probabilities for the requested partition.

    Raises
    ------
    ValidationError
        When no semi-supervised plan exists on the Session.
    """
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
    """Evaluate the semi-supervised plan on labeled rows of a holdout partition.

    Delegates to :func:`buildml.semisupervised.evaluate.evaluate_semisupervised`.
    Unlabeled holdout rows are skipped; holdout data is never used during fit.

    Parameters
    ----------
    session:
        Active Session with a semi-supervised plan from
        :func:`fit_semisupervised_op`.
    partition:
        Holdout partition to score. Validation falls back to test when absent.
    unlabeled_marker:
        Value treated as unlabeled when scoring labeled rows only.

    Returns
    -------
    SemiSupervisedEvalResult
        Holdout metrics computed on labeled rows only.

    Raises
    ------
    ValidationError
        When no semi-supervised plan exists on the Session.
    """
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
    """Persist the semi-supervised plan as ``buildml.semisupervised_bundle.v1``.

    Delegates to :func:`buildml.semisupervised.checkpoint.save_semisupervised_bundle`.
    Reload with :func:`load_semisupervised_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a semi-supervised plan from
        :func:`fit_semisupervised_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no semi-supervised plan exists on the Session.
    """
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
            "backend": getattr(plan, "backend", "sklearn"),
            "n_labeled_train": plan.n_labeled_train,
            "n_unlabeled_train": plan.n_unlabeled_train,
        },
    )
    return out


def load_semisupervised_bundle_op(session, path: str | Path, *, trusted: bool = False) -> Any:
    """Load a semi-supervised bundle into this Session.

    Delegates to :func:`buildml.semisupervised.checkpoint.load_semisupervised_bundle`
    and clears prior fit/predict/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded semi-supervised plan.
    path:
        Path to a ``buildml.semisupervised_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with semi-supervised plan attached for chaining.
    """
    plan = load_semisupervised_bundle(path, trusted=trusted)
    session._semisupervised_plan = plan
    session._semisupervised_fit_result = None
    session._semisupervised_predict_result = None
    session._semisupervised_eval_result = None
    session._record(
        "load_semisupervised_bundle",
        {"path": str(path), "method": plan.method, "backend": getattr(plan, "backend", "sklearn")},
        result_summary=plan.to_dict(),
    )
    return cast("Session", session)