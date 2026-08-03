"""Train-only federated learning simulation (native FedAvg / FedProx or Flower)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.federated.catalog import resolve_backend
from buildml.federated.features import (
    average_linear_params,
    client_ids_in_frame,
    encode_labels,
    extract_linear_params,
    frame_for_client,
    matrix_from_frame,
    resolve_client_column,
    resolve_federated_columns,
    resolve_target_column,
    set_linear_params,
)
from buildml.federated.results import FederatedFitResult, FederatedPlan
from buildml.federated.types import (
    FederatedBackend,
    FederatedConfig,
    FederatedEstimator,
    FederatedMethod,
    FederatedTask,
)

_CLASSIFIERS = {"sgd_classifier", "logistic_regression"}
_REGRESSORS = {"sgd_regressor", "ridge", "linear_regression"}
_PARTIAL_FIT = {"sgd_classifier", "sgd_regressor"}


@dataclass
class _FederatedContext:
    method_key: str
    est_key: str
    resolved_task: str
    client_col: str
    target_col: str
    cols: list[str]
    used_reduce: bool
    train: Any
    eligible: list[Any]
    label_encoder: Any
    classes_tuple: tuple[Any, ...] | None
    global_est: Any
    mu: float
    disclosures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def fit_federated(
    dataset: Dataset,
    split_plan: SplitPlan | None,
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
    reduce_plan: Any | None = None,
) -> tuple[FederatedPlan, FederatedFitResult]:
    """Simulate federated averaging on Session train clients.

    Backends
    --------
    native (default):
        In-process weighted coef_/intercept_ aggregation (FedAvg / FedProx).
    flower (``buildml[federated-industry]``):
        Flower NumPyClient wrappers over Session partitions + flwr weighted
        aggregation — still local simulation unless you deploy Flower yourself.

    Honesty
    -------
    Local FedAvg-style (or FedProx) orchestration on rows partitioned by a
    client/group column. Each "client" is a slice of the Session train
    partition — not a networked FL runtime unless you operate one separately.
    No cryptographic secure aggregation; model updates are averaged in-process
    with clear privacy limits (the orchestrator sees client updates).
    Validation/test partitions are never used for local training.

    Parameters
    ----------
    dataset:
        BuildML dataset with features, target, and client columns.
    split_plan:
        Train/validation/test split; train partition is used for local updates.
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
        Prefer reduced component columns when a reduce plan exists.
    min_client_rows:
        Minimum train rows required for a client to participate.
    reduce_plan:
        Optional preprocess reduce plan from Session.

    Returns
    -------
    tuple[FederatedPlan, FederatedFitResult]
        Fitted plan with global estimator and a serializable fit summary.

    Raises
    ------
    ValidationError
        When split, column, client, or hyperparameter preconditions fail.
    MissingExtraError
        When ``backend='flower'`` requires ``federated-industry`` and it is
        missing.
    """
    resolved_backend = resolve_backend(backend, method=method)
    if resolved_backend == "flower":
        from buildml.federated.adapters.flower import fit_flower

        return fit_flower(
            dataset,
            split_plan,
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
            reduce_plan=reduce_plan,
        )
    return _fit_native(
        dataset,
        split_plan,
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
        reduce_plan=reduce_plan,
    )


def _prepare_federated_context(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
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
    reduce_plan: Any | None = None,
) -> _FederatedContext:
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    method_key = str(method).lower().replace("-", "_")
    if method_key not in {"fedavg", "fedprox"}:
        raise ValidationError(
            f"Unknown federated method={method!r}. Supported: 'fedavg', 'fedprox'."
        )
    est_key = str(estimator).lower().replace("-", "_")
    if est_key not in _CLASSIFIERS | _REGRESSORS:
        raise ValidationError(
            f"Unknown federated estimator={estimator!r}. Supported: "
            f"{sorted(_CLASSIFIERS | _REGRESSORS)}."
        )
    if int(n_rounds) < 1:
        raise ValidationError("n_rounds must be >= 1.")
    if int(local_epochs) < 1:
        raise ValidationError("local_epochs must be >= 1.")
    if not (0.0 < float(client_fraction) <= 1.0):
        raise ValidationError("client_fraction must be in (0, 1].")
    if float(mu) < 0.0:
        raise ValidationError("mu (FedProx proximal strength) must be >= 0.")
    if method_key == "fedprox" and float(mu) == 0.0:
        raise ValidationError(
            "method='fedprox' requires mu > 0 (proximal strength toward the "
            "global model). Use method='fedavg' for plain averaging."
        )
    if method_key == "fedavg":
        mu = 0.0
    if int(min_client_rows) < 1:
        raise ValidationError("min_client_rows must be >= 1.")

    resolved_task = _resolve_task(est_key, task)
    disclosures: list[str] = []
    warnings: list[str] = []

    client_col, client_notes = resolve_client_column(dataset, client_column)
    disclosures.extend(client_notes)
    target_col, target_notes = resolve_target_column(dataset)
    disclosures.extend(target_notes)

    train = frame_for_partition(dataset, split_plan, "train")
    cols, used_reduce, col_notes = resolve_federated_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_col,
        client_column=client_col,
    )
    disclosures.extend(col_notes)

    all_client_ids = client_ids_in_frame(train, client_col)
    eligible: list[Any] = []
    skipped_small = 0
    for cid in all_client_ids:
        n = int((train[client_col] == cid).sum())
        if n >= int(min_client_rows):
            eligible.append(cid)
        else:
            skipped_small += 1
    if len(eligible) < 2:
        raise ValidationError(
            "Federated learning needs at least 2 clients with "
            f">= min_client_rows={min_client_rows} train rows "
            f"(found {len(eligible)} eligible of {len(all_client_ids)} "
            f"distinct client ids via column {client_col!r})."
        )
    if skipped_small:
        warnings.append(
            f"Skipped {skipped_small} client(s) with fewer than "
            f"{min_client_rows} train row(s)."
        )

    label_encoder = None
    classes_tuple: tuple[Any, ...] | None = None
    if resolved_task == "classification":
        _, label_encoder, classes_tuple = encode_labels(train[target_col])
        disclosures.append(
            "Classification class vocabulary discovered from the full train "
            "target column (labels only — client features used only during "
            f"that client's local updates). classes={list(classes_tuple)}."
        )

    global_est = _make_estimator(est_key, random_state)
    global_est = _initialize_global(
        global_est,
        train,
        eligible_clients=eligible,
        client_column=client_col,
        target_column=target_col,
        columns=cols,
        task=resolved_task,
        estimator_key=est_key,
        label_encoder=label_encoder,
        classes=classes_tuple,
        random_state=random_state,
    )

    return _FederatedContext(
        method_key=method_key,
        est_key=est_key,
        resolved_task=resolved_task,
        client_col=client_col,
        target_col=target_col,
        cols=cols,
        used_reduce=used_reduce,
        train=train,
        eligible=eligible,
        label_encoder=label_encoder,
        classes_tuple=classes_tuple,
        global_est=global_est,
        mu=float(mu),
        disclosures=disclosures,
        warnings=warnings,
    )


def _fit_native(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
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
    reduce_plan: Any | None = None,
) -> tuple[FederatedPlan, FederatedFitResult]:
    ctx = _prepare_federated_context(
        dataset,
        split_plan,
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
        reduce_plan=reduce_plan,
    )

    rng = np.random.default_rng(random_state)
    round_history: list[dict[str, Any]] = []
    final_metric: float | None = None

    for round_idx in range(int(n_rounds)):
        selected = _sample_clients(ctx.eligible, float(client_fraction), rng)
        client_params: list[dict[str, np.ndarray]] = []
        weights: list[float] = []
        client_metrics: list[float] = []
        participated: list[Any] = []
        global_params = extract_linear_params(ctx.global_est)

        for cid in selected:
            client_frame = frame_for_client(ctx.train, ctx.client_col, cid)
            n_rows = int(len(client_frame))
            if n_rows < int(min_client_rows):
                continue
            local_est, local_metric, local_notes = _local_update(
                ctx.global_est,
                global_params,
                client_frame,
                columns=ctx.cols,
                target_column=ctx.target_col,
                task=ctx.resolved_task,
                estimator_key=ctx.est_key,
                local_epochs=int(local_epochs),
                mu=float(ctx.mu),
                label_encoder=ctx.label_encoder,
                classes=ctx.classes_tuple,
                random_state=random_state,
            )
            if local_notes:
                for note in local_notes:
                    if note not in ctx.disclosures:
                        ctx.disclosures.append(note)
            client_params.append(extract_linear_params(local_est))
            weights.append(float(n_rows))
            participated.append(cid)
            if local_metric is not None:
                client_metrics.append(float(local_metric))

        if not client_params:
            ctx.warnings.append(
                f"Round {round_idx + 1}: no clients produced updates; "
                "stopping early."
            )
            break

        aggregated = average_linear_params(client_params, weights)
        set_linear_params(ctx.global_est, aggregated)
        if ctx.classes_tuple is not None and hasattr(ctx.global_est, "classes_"):
            ctx.global_est.classes_ = np.asarray(ctx.label_encoder.classes_)

        mean_client = float(np.mean(client_metrics)) if client_metrics else None
        final_metric = mean_client
        total_weight = float(sum(weights))
        round_history.append(
            {
                "round": round_idx + 1,
                "backend": "native",
                "n_clients": len(client_params),
                "n_samples": int(sum(weights)),
                "total_weight": total_weight,
                "weighting": "sample_size",
                "aggregation": "weighted coef_/intercept_ average",
                "mean_client_train_metric": mean_client,
                "client_ids": [str(c) for c in participated],
                "client_weights": {
                    str(cid): float(w)
                    for cid, w in zip(participated, weights, strict=True)
                },
            }
        )

    ctx.disclosures.extend(
        [
            "Federated fit uses the train partition only; validation/test "
            "are never used for local client updates.",
            "Each client sees only its own train rows during local updates.",
            "Aggregation is in-process weighted coefficient averaging "
            "(FedAvg / weighted-by-n) — not cryptographic secure aggregation.",
            "Honesty: local FL simulation for research/teaching/workflows — "
            "not a distributed FL platform unless you deploy one separately.",
            f"backend=native, method={ctx.method_key}, estimator={ctx.est_key}, "
            f"n_clients={len(ctx.eligible)}, n_rounds={n_rounds}, "
            f"local_epochs={local_epochs}, client_fraction={client_fraction}, "
            f"mu={ctx.mu}, n_train_rows={len(ctx.train)}.",
        ]
    )
    if ctx.method_key == "fedprox":
        ctx.disclosures.append(
            f"FedProx proximal pull applied after each local epoch "
            f"(mu={ctx.mu}): coef ← coef − mu·(coef − global)."
        )

    config = FederatedConfig(
        backend="native",
        method=ctx.method_key,  # type: ignore[arg-type]
        estimator=ctx.est_key,  # type: ignore[arg-type]
        task=ctx.resolved_task,  # type: ignore[arg-type]
        client_column=ctx.client_col,
        columns=tuple(ctx.cols),
        n_rounds=int(n_rounds),
        local_epochs=int(local_epochs),
        client_fraction=float(client_fraction),
        mu=float(ctx.mu),
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        min_client_rows=int(min_client_rows),
    )
    return _build_fit_outputs(
        ctx,
        round_history=round_history,
        final_metric=final_metric,
        backend="native",
        config=config,
        n_rounds=int(n_rounds),
        local_epochs=int(local_epochs),
        client_fraction=float(client_fraction),
    )


def _build_fit_outputs(
    ctx: _FederatedContext,
    *,
    round_history: list[dict[str, Any]],
    final_metric: float | None,
    backend: str,
    config: FederatedConfig,
    n_rounds: int | None = None,
    local_epochs: int | None = None,
    client_fraction: float | None = None,
) -> tuple[FederatedPlan, FederatedFitResult]:
    rounds = int(n_rounds if n_rounds is not None else config.n_rounds)
    epochs = int(local_epochs if local_epochs is not None else config.local_epochs)
    fraction = float(
        client_fraction if client_fraction is not None else config.client_fraction
    )
    plan = FederatedPlan(
        backend=backend,
        method=ctx.method_key,
        estimator_name=ctx.est_key,
        task=ctx.resolved_task,
        columns=tuple(ctx.cols),
        target_column=ctx.target_col,
        client_column=ctx.client_col,
        client_ids=tuple(ctx.eligible),
        n_train_rows=int(len(ctx.train)),
        n_rounds=rounds,
        local_epochs=epochs,
        client_fraction=fraction,
        mu=float(ctx.mu),
        classes_=ctx.classes_tuple,
        round_history=tuple(round_history),
        estimator_=ctx.global_est,
        label_encoder_=ctx.label_encoder,
        disclosures=tuple(ctx.disclosures),
        warnings=tuple(ctx.warnings),
        used_reduce_components=ctx.used_reduce,
        config=config.to_dict(),
    )
    result = FederatedFitResult(
        backend=backend,
        method=ctx.method_key,
        estimator_name=ctx.est_key,
        task=ctx.resolved_task,
        n_train_rows=int(len(ctx.train)),
        n_clients=len(ctx.eligible),
        n_rounds=rounds,
        local_epochs=epochs,
        client_column=ctx.client_col,
        columns=tuple(ctx.cols),
        target_column=ctx.target_col,
        final_train_metric=final_metric,
        round_history=tuple(round_history),
        used_reduce_components=ctx.used_reduce,
        disclosures=tuple(ctx.disclosures),
        warnings=tuple(ctx.warnings),
    )
    return plan, result


def _linear_ndarrays(estimator: Any) -> list[np.ndarray]:
    params = extract_linear_params(estimator)
    return [
        np.asarray(params["coef_"], dtype=float).ravel().copy(),
        np.asarray(params["intercept_"], dtype=float).copy(),
    ]


def _ndarrays_to_linear(
    ndarrays: list[np.ndarray],
    *,
    template: Any,
) -> dict[str, np.ndarray]:
    template_params = extract_linear_params(template)
    coef = np.asarray(ndarrays[0], dtype=float).reshape(template_params["coef_"].shape)
    intercept = np.asarray(ndarrays[1], dtype=float).reshape(
        template_params["intercept_"].shape
    )
    return {"coef_": coef, "intercept_": intercept}


def _resolve_task(est_key: str, task: FederatedTask | None) -> str:
    if est_key in _CLASSIFIERS:
        if task is not None and task != "classification":
            raise ValidationError(
                f"estimator={est_key!r} only supports task='classification'."
            )
        return "classification"
    if est_key in _REGRESSORS:
        if task is not None and task != "regression":
            raise ValidationError(
                f"estimator={est_key!r} only supports task='regression'."
            )
        return "regression"
    raise ValidationError(f"Cannot resolve task for estimator={est_key!r}.")


def _make_estimator(est_key: str, random_state: int | None) -> Any:
    if est_key == "sgd_classifier":
        return SGDClassifier(
            loss="log_loss",
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=random_state,
            learning_rate="optimal",
        )
    if est_key == "sgd_regressor":
        return SGDRegressor(
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=random_state,
            learning_rate="optimal",
        )
    if est_key == "logistic_regression":
        return LogisticRegression(
            max_iter=200,
            warm_start=True,
            random_state=random_state,
        )
    if est_key == "ridge":
        return Ridge(random_state=random_state)
    if est_key == "linear_regression":
        return LinearRegression()
    raise ValidationError(f"Unknown estimator factory for {est_key!r}.")


def _initialize_global(
    estimator: Any,
    train: Any,
    *,
    eligible_clients: list[Any],
    client_column: str,
    target_column: str,
    columns: list[str],
    task: str,
    estimator_key: str,
    label_encoder: Any,
    classes: tuple[Any, ...] | None,
    random_state: int | None,
) -> Any:
    """Seed global coef_ shapes via a small train-only init fit."""
    sizes = [
        (cid, int((train[client_column] == cid).sum())) for cid in eligible_clients
    ]
    sizes.sort(key=lambda t: t[1], reverse=True)
    init_cid = sizes[0][0]
    init_frame = frame_for_client(train, client_column, init_cid)
    x = matrix_from_frame(init_frame, columns)
    if task == "classification":
        y, _, _ = encode_labels(init_frame[target_column], label_encoder=label_encoder)
        if estimator_key in _PARTIAL_FIT:
            estimator.partial_fit(x, y, classes=np.arange(len(classes or ())))
        else:
            estimator.fit(x, y)
    else:
        y = init_frame[target_column].to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValidationError(
                "Federated regression targets contain nulls on the init client; "
                "impute or drop before fit_federated."
            )
        if estimator_key in _PARTIAL_FIT:
            estimator.partial_fit(x, y)
        else:
            estimator.fit(x, y)
    _ = random_state
    return estimator


def _sample_clients(
    eligible: list[Any],
    client_fraction: float,
    rng: np.random.Generator,
) -> list[Any]:
    n = max(1, int(np.ceil(len(eligible) * float(client_fraction))))
    n = min(n, len(eligible))
    if n == len(eligible):
        return list(eligible)
    idx = rng.choice(len(eligible), size=n, replace=False)
    return [eligible[int(i)] for i in idx]


def _local_update(
    global_est: Any,
    global_params: dict[str, np.ndarray],
    client_frame: Any,
    *,
    columns: list[str],
    target_column: str,
    task: str,
    estimator_key: str,
    local_epochs: int,
    mu: float,
    label_encoder: Any,
    classes: tuple[Any, ...] | None,
    random_state: int | None,
) -> tuple[Any, float | None, list[str]]:
    """Train a client model starting from global parameters."""
    notes: list[str] = []
    x = matrix_from_frame(client_frame, columns)
    if task == "classification":
        y, _, _ = encode_labels(
            client_frame[target_column], label_encoder=label_encoder
        )
    else:
        y = client_frame[target_column].to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValidationError(
                "Federated regression targets contain nulls in a client "
                "partition; impute or drop before fit_federated."
            )

    local = clone(global_est)
    set_linear_params(local, global_params)
    if classes is not None and hasattr(local, "classes_"):
        local.classes_ = np.asarray(label_encoder.classes_)
    for attr in ("n_features_in_", "feature_names_in_", "t_"):
        if hasattr(global_est, attr):
            setattr(local, attr, deepcopy(getattr(global_est, attr)))

    if estimator_key in _PARTIAL_FIT:
        for _epoch in range(local_epochs):
            if task == "classification":
                local.partial_fit(x, y, classes=np.arange(len(classes or ())))
            else:
                local.partial_fit(x, y)
            if mu > 0.0:
                _apply_proximal_pull(local, global_params, mu)
        notes.append(
            f"Local updates used sklearn partial_fit for {estimator_key} "
            f"({local_epochs} epoch(s) per selected client)."
        )
    else:
        for _epoch in range(local_epochs):
            if hasattr(local, "warm_start"):
                local.warm_start = True
            if hasattr(local, "n_iter_") and local.n_iter_ is None:
                local.n_iter_ = np.array([0])
            try:
                local.fit(x, y)
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(
                    f"Local federated fit failed for estimator={estimator_key!r}: "
                    f"{exc}"
                ) from exc
            if mu > 0.0:
                _apply_proximal_pull(local, global_params, mu)
        notes.append(
            f"Local updates used sklearn .fit for {estimator_key} "
            f"({local_epochs} pass(es) per selected client)."
        )

    metric = _score_local(local, x, y, task=task)
    _ = random_state
    return local, metric, notes


def _apply_proximal_pull(
    estimator: Any,
    global_params: dict[str, np.ndarray],
    mu: float,
) -> None:
    """Apply FedProx-style proximal pull toward the global parameters."""
    params = extract_linear_params(estimator)
    params["coef_"] = params["coef_"] - float(mu) * (
        params["coef_"] - global_params["coef_"]
    )
    params["intercept_"] = params["intercept_"] - float(mu) * (
        params["intercept_"] - global_params["intercept_"]
    )
    set_linear_params(estimator, params)


def _score_local(
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
) -> float | None:
    try:
        pred = estimator.predict(x)
    except Exception:  # noqa: BLE001
        return None
    if task == "classification":
        return float(accuracy_score(y, pred))
    try:
        return float(r2_score(y, pred))
    except Exception:  # noqa: BLE001
        return float(-mean_squared_error(y, pred))
