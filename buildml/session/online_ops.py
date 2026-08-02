"""Thin Session facades over buildml.online."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.online.checkpoint import load_online_bundle, save_online_bundle
from buildml.online.evaluate import evaluate_online
from buildml.online.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
    update_result_summary,
)
from buildml.online.fit import fit_online
from buildml.online.predict import predict_online
from buildml.online.types import OnlineBackend, OnlineDriftDetector, OnlineEstimator, OnlineTask
from buildml.online.update import partial_fit_online

PartitionOrAll = PartitionName | Literal["all"]


def fit_online_op(
    session,
    *,
    backend: OnlineBackend | None = None,
    estimator: OnlineEstimator | str = "sgd_classifier",
    task: OnlineTask | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    chunk_size: int = 50,
    n_init: int | None = None,
    indices: Sequence[Any] | None = None,
    classes: Sequence[Any] | None = None,
    prefer_reduce_components: bool = True,
    allow_refit_fallback: bool = False,
    drift_disclose: bool = True,
    drift_detector: OnlineDriftDetector | None = None,
    buffer_size: int = 512,
    epochs_per_update: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> Any:
    """Warm-start an incremental estimator on the first train chunk.

    Notes
    -----
    **Leakage:** Requires a split. Init and later updates use train chunks only
    (or role-aligned external frames). Validation/test are never used for
    updates. Classifiers need a ``classes`` vocabulary (explicit or discovered
    from the full train target column — labels only).
    """
    session.assert_can_fit("train")
    plan, result = fit_online(
        session.dataset,
        session._split_plan,
        backend=backend,
        estimator=str(estimator),
        task=task,
        columns=columns,
        random_state=random_state,
        chunk_size=chunk_size,
        n_init=n_init,
        indices=indices,
        classes=classes,
        prefer_reduce_components=prefer_reduce_components,
        allow_refit_fallback=allow_refit_fallback,
        drift_disclose=drift_disclose,
        drift_detector=drift_detector,
        buffer_size=buffer_size,
        epochs_per_update=epochs_per_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
        hidden_dim=hidden_dim,
        device=device,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._online_plan = plan
    session._online_fit_result = result
    session._online_update_result = None
    session._online_eval_result = None
    session._online_predict_result = None
    session._record(
        "fit_online",
        {
            "backend": plan.backend,
            "estimator": estimator,
            "task": task,
            "columns": columns,
            "random_state": random_state,
            "chunk_size": chunk_size,
            "n_init": n_init,
            "indices": None if indices is None else list(indices),
            "classes": None if classes is None else list(classes),
            "prefer_reduce_components": prefer_reduce_components,
            "allow_refit_fallback": allow_refit_fallback,
            "drift_disclose": drift_disclose,
            "drift_detector": (plan.config or {}).get("drift_detector"),
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def partial_fit_online_op(
    session,
    *,
    n_rows: int | None = None,
    indices: Sequence[Any] | None = None,
    frame: pd.DataFrame | None = None,
) -> Any:
    """Incremental ``partial_fit`` update on the next train chunk or frame."""
    plan = getattr(session, "_online_plan", None)
    if plan is None:
        raise ValidationError("No online plan. Call fit_online(...) first.")
    new_plan, result = partial_fit_online(
        session.dataset,
        plan,
        session._split_plan,
        n_rows=n_rows,
        indices=indices,
        frame=frame,
    )
    session._online_plan = new_plan
    session._online_update_result = result
    session._record(
        "partial_fit_online",
        {
            "n_rows": n_rows,
            "n_indices": None if indices is None else len(list(indices)),
            "external_frame": frame is not None,
            "update_mode": result.update_mode,
        },
        warnings=tuple(result.warnings),
        result_summary=update_result_summary(result),
    )
    return result


def evaluate_online_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    drift_check: bool = True,
) -> Any:
    """Evaluate the online learner on a holdout partition (never for updates)."""
    plan = getattr(session, "_online_plan", None)
    if plan is None:
        raise ValidationError("No online plan. Call fit_online(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_online(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        drift_check=drift_check,
    )
    session._online_eval_result = result
    session._record(
        "evaluate_online",
        {"partition": resolved, "drift_detected": result.drift_detected},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def predict_online_op(
    session,
    *,
    partition: PartitionOrAll = "test",
) -> Any:
    """Predict with the incremental online estimator (no update)."""
    plan = getattr(session, "_online_plan", None)
    if plan is None:
        raise ValidationError("No online plan. Call fit_online(...) first.")
    result = predict_online(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
    )
    session._online_predict_result = result
    session._record(
        "predict_online",
        {"partition": partition},
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def save_online_bundle_op(session, path: str | Path) -> Path:
    """Persist the active OnlinePlan as ``buildml.online_bundle.v1``."""
    plan = getattr(session, "_online_plan", None)
    if plan is None:
        raise ValidationError("No online plan. Call fit_online(...) first.")
    out = save_online_bundle(
        path,
        plan,
        fit_result=getattr(session, "_online_fit_result", None),
        eval_result=getattr(session, "_online_eval_result", None),
    )
    session._record(
        "save_online_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "backend": plan.backend,
            "estimator_name": plan.estimator_name,
            "n_seen_rows": plan.n_seen_rows,
            "n_updates": plan.n_updates,
        },
    )
    return out


def load_online_bundle_op(session, path: str | Path) -> Any:
    """Load an online-learning bundle into this Session."""
    plan = load_online_bundle(path)
    session._online_plan = plan
    session._online_fit_result = None
    session._online_update_result = None
    session._online_eval_result = None
    session._online_predict_result = None
    session._record(
        "load_online_bundle",
        {"path": str(path), "backend": plan.backend, "estimator_name": plan.estimator_name},
        result_summary=plan.to_dict(),
    )
    return session
