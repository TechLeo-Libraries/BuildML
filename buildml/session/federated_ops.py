"""Thin Session facades over buildml.federated."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.federated.checkpoint import (
    load_federated_bundle,
    save_federated_bundle,
)
from buildml.federated.evaluate import evaluate_federated
from buildml.federated.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
)
from buildml.federated.fit import fit_federated
from buildml.federated.predict import predict_federated
from buildml.federated.results import export_round_history
from buildml.federated.types import (
    FederatedBackend,
    FederatedEstimator,
    FederatedMethod,
    FederatedTask,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_federated_op(
    session,
    *,
    backend: FederatedBackend | None = None,
    method: FederatedMethod = "fedavg",
    estimator: FederatedEstimator = "sgd_classifier",
    task: FederatedTask | None = None,
    client_column: str | None = None,
    columns: list[str] | None = None,
    n_rounds: int = 5,
    local_epochs: int = 1,
    client_fraction: float = 1.0,
    mu: float = 0.0,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    min_client_rows: int = 2,
) -> Any:
    """Simulate federated averaging on Session train clients.

    Delegates to :func:`buildml.federated.fit.fit_federated`, stores the global
    :class:`~buildml.federated.results.FederatedPlan` on Session, and records
    the fit. Follow with :func:`evaluate_federated_op` or
    :func:`predict_federated_op` on holdout partitions.

    Parameters
    ----------
    session:
        Active Session with dataset, split plan, and client/group column.
    backend:
        Optional backend override (``native`` or ``flower``).
    method:
        Federated aggregation method (``fedavg`` or ``fedprox``).
    estimator:
        Sklearn linear/SGD estimator key for local and global models.
    task:
        Optional task override; inferred from ``estimator`` when ``None``.
    client_column:
        Optional explicit client/group column.
    columns:
        Optional explicit feature columns.
    n_rounds:
        Number of federated communication rounds.
    local_epochs:
        Local training epochs per selected client per round.
    client_fraction:
        Fraction of eligible clients sampled each round.
    mu:
        FedProx proximal strength (required when ``method='fedprox'``).
    random_state:
        Seed for client sampling and estimator initialization.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    min_client_rows:
        Minimum train rows required for a client to participate.

    Returns
    -------
    FederatedFitResult
        Serializable fit summary including rounds, clients, and disclosures.
        Use :func:`evaluate_federated_op` or :func:`predict_federated_op` next.

    Notes
    -----
    **Leakage:** Requires a split. Local client updates use train only.
    Validation/test are never used for training. Needs a client/group column
    (role or ``client_column=``) and exactly one ``role='target'``. Honesty:
    local FedAvg-style simulation — ``backend='flower'`` uses Flower libraries
    but still runs in-process unless you deploy Flower separately; not
    cryptographic secure aggregation.
    """
    session.assert_can_fit("train")
    plan, result = fit_federated(
        session.dataset,
        session._split_plan,
        backend=backend,
        method=method,
        estimator=estimator,
        task=task,
        client_column=client_column,
        columns=columns,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        mu=mu,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        min_client_rows=min_client_rows,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._federated_plan = plan
    session._federated_fit_result = result
    session._federated_eval_result = None
    session._federated_predict_result = None
    session._record(
        "fit_federated",
        {
            "backend": result.backend,
            "method": method,
            "estimator": estimator,
            "task": task,
            "client_column": client_column,
            "columns": columns,
            "n_rounds": n_rounds,
            "local_epochs": local_epochs,
            "client_fraction": client_fraction,
            "mu": mu,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "min_client_rows": min_client_rows,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def evaluate_federated_op(
    session,
    *,
    backend: FederatedBackend | None = None,
    partition: PartitionOrAll = "validation",
    per_client: bool = True,
) -> Any:
    """Evaluate the global federated model on a holdout partition.

    Delegates to :func:`buildml.federated.evaluate.evaluate_federated`.
    Holdout data is never used for federated training rounds.

    Parameters
    ----------
    session:
        Active Session with a federated plan from :func:`fit_federated_op`.
    backend:
        Optional backend override for evaluation adapters.
    partition:
        Holdout partition to score. Validation falls back to test when absent.
    per_client:
        When True, include per-client holdout metrics in the result.

    Returns
    -------
    FederatedEvalResult
        Global and optional per-client holdout metrics.

    Raises
    ------
    ValidationError
        When no federated plan exists on the Session.
    """
    plan = getattr(session, "_federated_plan", None)
    if plan is None:
        raise ValidationError(
            "No federated plan. Call fit_federated(...) first."
        )
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_federated(
        session.dataset,
        plan,
        session._split_plan,
        backend=backend,
        partition=resolved,
        per_client=per_client,
    )
    session._federated_eval_result = result
    session._record(
        "evaluate_federated",
        {
            "backend": backend,
            "partition": resolved,
            "per_client": per_client,
        },
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def predict_federated_op(
    session,
    *,
    backend: FederatedBackend | None = None,
    partition: PartitionOrAll = "test",
) -> Any:
    """Predict with the global federated model without local updates.

    Delegates to :func:`buildml.federated.predict.predict_federated` and
    stores predictions on Session.

    Parameters
    ----------
    session:
        Active Session with a federated plan from :func:`fit_federated_op`.
    backend:
        Optional backend override for prediction adapters.
    partition:
        Partition to score (``train``, ``validation``, ``test``, or ``all``).

    Returns
    -------
    FederatedPredictResult
        Predictions from the aggregated global model.

    Raises
    ------
    ValidationError
        When no federated plan exists on the Session.
    """
    plan = getattr(session, "_federated_plan", None)
    if plan is None:
        raise ValidationError(
            "No federated plan. Call fit_federated(...) first."
        )
    result = predict_federated(
        session.dataset,
        plan,
        session._split_plan,
        backend=backend,
        partition=partition,
    )
    session._federated_predict_result = result
    session._record(
        "predict_federated",
        {"backend": backend, "partition": partition},
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def export_round_history_op(
    session,
    path: str | Path,
    *,
    include_disclosures: bool = False,
) -> Path:
    """Export federated round metrics to JSON for audit and teaching overlays.

    Delegates to :func:`buildml.federated.results.export_round_history` using
    the active :class:`~buildml.federated.results.FederatedPlan` on Session.

    Parameters
    ----------
    session:
        Active Session with a federated plan from :func:`fit_federated_op`.
    path:
        Destination JSON file path (parent directories are created).
    include_disclosures:
        When ``True``, embed plan disclosures and warnings in the payload.

    Returns
    -------
    pathlib.Path
        Resolved output file path.

    Raises
    ------
    ValidationError
        When no federated plan exists on the Session.
    """
    plan = getattr(session, "_federated_plan", None)
    if plan is None:
        raise ValidationError(
            "No federated plan. Call fit_federated(...) first."
        )
    out = export_round_history(
        plan,
        path,
        include_disclosures=include_disclosures,
    )
    session._record(
        "export_round_history",
        {
            "path": str(out),
            "include_disclosures": include_disclosures,
            "n_rounds_completed": len(plan.round_history),
        },
        result_summary={
            "path": str(out),
            "backend": getattr(plan, "backend", "native"),
            "method": plan.method,
            "n_rounds_completed": len(plan.round_history),
        },
    )
    return out


def save_federated_bundle_op(session, path: str | Path) -> Path:
    """Persist the active federated plan as ``buildml.federated_bundle.v1``.

    Delegates to :func:`buildml.federated.checkpoint.save_federated_bundle`.
    Reload with :func:`load_federated_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a federated plan from :func:`fit_federated_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no federated plan exists on the Session.
    """
    plan = getattr(session, "_federated_plan", None)
    if plan is None:
        raise ValidationError(
            "No federated plan. Call fit_federated(...) first."
        )
    out = save_federated_bundle(
        path,
        plan,
        fit_result=getattr(session, "_federated_fit_result", None),
        eval_result=getattr(session, "_federated_eval_result", None),
    )
    session._record(
        "save_federated_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "backend": getattr(plan, "backend", "native"),
            "method": plan.method,
            "estimator_name": plan.estimator_name,
            "client_column": plan.client_column,
            "n_clients": len(plan.client_ids),
        },
    )
    return out


def load_federated_bundle_op(session, path: str | Path, *, trusted: bool = False) -> Any:
    """Load a federated-learning bundle into this Session.

    Delegates to :func:`buildml.federated.checkpoint.load_federated_bundle`
    and clears prior fit/eval/predict results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded federated plan.
    path:
        Path to a ``buildml.federated_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with federated plan attached for chaining.
    """
    plan = load_federated_bundle(path, trusted=trusted)
    session._federated_plan = plan
    session._federated_fit_result = None
    session._federated_eval_result = None
    session._federated_predict_result = None
    session._record(
        "load_federated_bundle",
        {
            "path": str(path),
            "backend": getattr(plan, "backend", "native"),
            "method": plan.method,
            "estimator_name": plan.estimator_name,
            "client_column": plan.client_column,
            "n_clients": len(plan.client_ids),
        },
        result_summary=plan.to_dict(),
    )
    return cast("Session", session)