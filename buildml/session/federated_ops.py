"""Thin Session facades over buildml.federated."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
    """Holdout evaluation of the global federated model (never for training)."""
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
    """Predict with the global federated model (no update)."""
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


def save_federated_bundle_op(session, path: str | Path) -> Path:
    """Persist the active FederatedPlan as ``buildml.federated_bundle.v1``."""
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


def load_federated_bundle_op(session, path: str | Path) -> Any:
    """Load a federated-learning bundle into this Session."""
    plan = load_federated_bundle(path)
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
    return session
