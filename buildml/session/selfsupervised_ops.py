"""Thin Session facades over buildml.selfsupervised."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.selfsupervised.checkpoint import load_ssl_bundle, save_ssl_bundle
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    head_fit_result_summary,
    transform_result_summary,
)
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.fit import fit_ssl_pretext
from buildml.selfsupervised.transform import transform_ssl
from buildml.selfsupervised.types import SelfSupervisedMethod, SSLHeadEstimator

PartitionOrAll = PartitionName | Literal["all"]


def fit_ssl_pretext_op(
    session,
    *,
    method: SelfSupervisedMethod | None = None,
    columns: list[str] | None = None,
    text_column: str | None = None,
    image_column: str | None = None,
    random_state: int | None = 0,
    latent_dim: int = 16,
    hidden: tuple[int, ...] | list[int] = (64,),
    mask_ratio: float = 0.15,
    n_mask_views: int = 3,
    max_iter: int = 200,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    temperature: float = 0.5,
    projector_dim: int = 32,
    projector_hidden: tuple[int, ...] | list[int] = (64,),
    prefer_reduce_components: bool = True,
    representation_prefix: str = "ssl_emb",
    backbone: str = "resnet18",
    weight_mode: str = "mock",
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Any:
    """Fit a self-supervised pretext encoder on the train partition only.

    Delegates to :func:`buildml.selfsupervised.fit.fit_ssl_pretext`, stores the
    :class:`~buildml.selfsupervised.results.SSLPlan` on Session, and records
    the fit. Follow with :func:`transform_ssl_op` or :func:`finetune_ssl_head_op`.

    Parameters
    ----------
    session:
        Active Session with tabular, text, or image columns and a split plan.
    method:
        Self-supervised method override; inferred from modality when ``None``.
    columns:
        Tabular feature columns for pretext training.
    text_column:
        Text column for language-model or contrastive text methods.
    image_column:
        Image path/bytes column for vision methods.
    random_state:
        Seed for augmentations and initialization.
    latent_dim:
        Output embedding dimensionality.
    hidden:
        Hidden layer sizes for tabular encoders.
    mask_ratio:
        Fraction of features masked in masked-modeling pretext tasks.
    n_mask_views:
        Number of masked views per sample for contrastive objectives.
    max_iter:
        Maximum iterations for sklearn-style encoders.
    epochs:
        Training epochs for torch backends.
    batch_size:
        Minibatch size for torch training.
    learning_rate:
        Optimizer learning rate for torch backends.
    temperature:
        Temperature for contrastive loss scaling.
    projector_dim:
        Projector head dimension for contrastive methods.
    projector_hidden:
        Hidden sizes for the contrastive projector MLP.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    representation_prefix:
        Column prefix when attaching embeddings to the dataset.
    backbone:
        Vision backbone architecture name for image methods.
    weight_mode:
        Weight initialization mode for mock/demo vision backends.
    hf_model_name:
        HuggingFace model name for text embedding methods.
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    SSLFitResult
        Serializable fit summary including method, modality, and disclosures.
    """
    session.assert_can_fit("train")
    plan, result = fit_ssl_pretext(
        session.dataset,
        session._split_plan,
        method=method,
        columns=columns,
        text_column=text_column,
        image_column=image_column,
        random_state=random_state,
        latent_dim=latent_dim,
        hidden=hidden,
        mask_ratio=mask_ratio,
        n_mask_views=n_mask_views,
        max_iter=max_iter,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        projector_dim=projector_dim,
        projector_hidden=projector_hidden,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
        representation_prefix=representation_prefix,
        backbone=backbone,
        weight_mode=weight_mode,
        hf_model_name=hf_model_name,
        device=device,
    )
    session._ssl_plan = plan
    session._ssl_fit_result = result
    session._ssl_transform_result = None
    session._ssl_head_plan = None
    session._ssl_head_fit_result = None
    session._ssl_eval_result = None
    session._record(
        "fit_ssl_pretext",
        {
            "method": result.method,
            "modality": getattr(result, "modality", "tabular"),
            "columns": columns,
            "text_column": text_column,
            "image_column": image_column,
            "latent_dim": latent_dim,
            "hidden": list(hidden),
            "mask_ratio": mask_ratio,
            "n_mask_views": n_mask_views,
            "max_iter": max_iter,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "projector_dim": projector_dim,
            "prefer_reduce_components": prefer_reduce_components,
            "representation_prefix": representation_prefix,
            "backbone": backbone,
            "weight_mode": weight_mode,
            "hf_model_name": hf_model_name,
            "device": device,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def transform_ssl_op(
    session,
    *,
    partition: PartitionOrAll = "train",
    attach: bool = False,
) -> Any:
    """Export SSL representations with the train-fitted pretext encoder.

    Delegates to :func:`buildml.selfsupervised.transform.transform_ssl`
    without refitting. Optionally attaches embedding columns to Session dataset.

    Parameters
    ----------
    session:
        Active Session with an SSLPlan from :func:`fit_ssl_pretext_op`.
    partition:
        Split partition to encode (default ``train``).
    attach:
        When True, merge embedding columns into the Session dataset frame.

    Returns
    -------
    SSLTransformResult
        Embedding matrix metadata and optional attached column names.

    Raises
    ------
    ValidationError
        When no SSL plan exists on the Session.
    """
    plan = getattr(session, "_ssl_plan", None)
    if plan is None:
        raise ValidationError("No SSL plan. Call fit_ssl_pretext(...) first.")
    new_dataset, result, _emb = transform_ssl(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        attach=attach,
    )
    if new_dataset is not None:
        session._dataset = new_dataset
    session._ssl_transform_result = result
    session._record(
        "transform_ssl",
        {"partition": partition, "attach": attach},
        result_summary=transform_result_summary(result),
    )
    return result


def finetune_ssl_head_op(
    session,
    *,
    estimator: SSLHeadEstimator = "logistic_regression",
    random_state: int | None = 0,
    unlabeled_marker: Any = None,
) -> Any:
    """Fit a supervised head on frozen SSL embeddings using labeled train rows.

    Delegates to :func:`buildml.selfsupervised.finetune.finetune_ssl_head`.
    Requires a prior :func:`fit_ssl_pretext_op`.

    Parameters
    ----------
    session:
        Active Session with an SSLPlan from :func:`fit_ssl_pretext_op`.
    estimator:
        Supervised head estimator (``logistic_regression``, etc.).
    random_state:
        Seed for head fitting.
    unlabeled_marker:
        Value marking unlabeled rows to exclude from head training.

    Returns
    -------
    SSLHeadFitResult
        Head fit summary including labeled row counts and disclosures.

    Raises
    ------
    ValidationError
        When no SSL plan exists on the Session.
    """
    plan = getattr(session, "_ssl_plan", None)
    if plan is None:
        raise ValidationError("No SSL plan. Call fit_ssl_pretext(...) first.")
    session.assert_can_fit("train")
    head_plan, result = finetune_ssl_head(
        session.dataset,
        session._split_plan,
        plan,
        estimator=estimator,
        random_state=random_state,
        unlabeled_marker=unlabeled_marker,
    )
    session._ssl_head_plan = head_plan
    session._ssl_head_fit_result = result
    session._ssl_eval_result = None
    session._record(
        "finetune_ssl_head",
        {
            "estimator": estimator,
            "random_state": random_state,
            "unlabeled_marker": unlabeled_marker,
        },
        warnings=tuple(result.warnings),
        result_summary=head_fit_result_summary(result),
    )
    return result


def evaluate_ssl_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> Any:
    """Evaluate frozen SSL pretext encoder and head on a labeled partition.

    Delegates to :func:`buildml.selfsupervised.evaluate.evaluate_ssl`.
    Requires both :func:`fit_ssl_pretext_op` and :func:`finetune_ssl_head_op`.
    Falls back to ``test`` when no validation partition exists.

    Parameters
    ----------
    session:
        Active Session with SSL and head plans from prior fit steps.
    partition:
        Holdout partition for evaluation (default ``validation``).
    unlabeled_marker:
        Value marking unlabeled rows to exclude from evaluation.

    Returns
    -------
    SSLEvalResult
        Holdout metrics for the frozen pretext + head pipeline.

    Raises
    ------
    ValidationError
        When SSL or head plans are missing on the Session.
    """
    ssl_plan = getattr(session, "_ssl_plan", None)
    head_plan = getattr(session, "_ssl_head_plan", None)
    if ssl_plan is None or head_plan is None:
        raise ValidationError(
            "evaluate_ssl requires fit_ssl_pretext(...) and finetune_ssl_head(...)."
        )
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_ssl(
        session.dataset,
        ssl_plan,
        head_plan,
        session._split_plan,
        partition=resolved,
        unlabeled_marker=unlabeled_marker,
    )
    session._ssl_eval_result = result
    session._record(
        "evaluate_ssl",
        {"partition": resolved, "unlabeled_marker": unlabeled_marker},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_ssl_bundle_op(session, path: str | Path) -> Path:
    """Persist the active SSL plan as ``buildml.ssl_bundle.v2``.

    Delegates to :func:`buildml.selfsupervised.checkpoint.save_ssl_bundle`.
    Reload with :func:`load_ssl_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with an SSLPlan from :func:`fit_ssl_pretext_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no SSL plan exists on the Session.
    """
    plan = getattr(session, "_ssl_plan", None)
    if plan is None:
        raise ValidationError("No SSL plan. Call fit_ssl_pretext(...) first.")
    out = save_ssl_bundle(
        path,
        plan,
        fit_result=getattr(session, "_ssl_fit_result", None),
        head_plan=getattr(session, "_ssl_head_plan", None),
        head_fit_result=getattr(session, "_ssl_head_fit_result", None),
        eval_result=getattr(session, "_ssl_eval_result", None),
    )
    session._record(
        "save_ssl_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "method": plan.method,
            "latent_dim": plan.latent_dim,
            "has_head": getattr(session, "_ssl_head_plan", None) is not None,
        },
    )
    return out


def load_ssl_bundle_op(session, path: str | Path) -> Any:
    """Load a self-supervised bundle into this Session.

    Delegates to :func:`buildml.selfsupervised.checkpoint.load_ssl_bundle`
    and clears prior transform/head/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded SSLPlan.
    path:
        Path to a ``buildml.ssl_bundle.v2`` directory.

    Returns
    -------
    Session
        ``session`` with SSLPlan and optional head plan attached.
    """
    plan, head = load_ssl_bundle(path)
    session._ssl_plan = plan
    session._ssl_head_plan = head
    session._ssl_fit_result = None
    session._ssl_transform_result = None
    session._ssl_head_fit_result = None
    session._ssl_eval_result = None
    session._record(
        "load_ssl_bundle",
        {
            "path": str(path),
            "method": plan.method,
            "has_head": head is not None,
        },
        result_summary=plan.to_dict(),
    )
    return session
