"""Flower (flwr) adapter: NumPyClient wrappers + weighted aggregation (local sim)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from flwr.client import NumPyClient
from flwr.server.strategy.aggregate import aggregate

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.federated.extras import require_flwr
from buildml.federated.features import (
    extract_linear_params,
    frame_for_client,
    matrix_from_frame,
    set_linear_params,
)
from buildml.federated.fit import (
    _build_fit_outputs,
    _linear_ndarrays,
    _ndarrays_to_linear,
    _prepare_federated_context,
    _sample_clients,
    _score_local,
)
from buildml.federated.results import FederatedFitResult, FederatedPlan
from buildml.federated.types import (
    FederatedConfig,
    FederatedEstimator,
    FederatedMethod,
    FederatedTask,
)


@dataclass(slots=True)
class _ClientPartition:
    client_id: Any
    frame: Any
    n_rows: int


class _BuildMLSklearnClient(NumPyClient):
    """Flower NumPyClient delegating local fit to Session client partitions."""

    def __init__(
        self,
        partition: _ClientPartition,
        *,
        global_template: Any,
        columns: list[str],
        target_column: str,
        task: str,
        estimator_key: str,
        local_epochs: int,
        mu: float,
        label_encoder: Any,
        classes: tuple[Any, ...] | None,
        random_state: int | None,
    ) -> None:
        """Configure a Flower NumPyClient for one Session client partition.

        Stores partition data and training hyperparameters so Flower can invoke
        local fit and evaluate callbacks against a single client slice.

        Parameters
        ----------
        partition:
            Client slice with frame and row count metadata.
        global_template:
            Global sklearn estimator template for parameter shape.
        columns:
            Feature column names for local updates.
        target_column:
            Target column name on the client frame.
        task:
            ``classification`` or ``regression`` task mode.
        estimator_key:
            Sklearn estimator factory key.
        local_epochs:
            Local training epochs per fit call.
        mu:
            FedProx proximal strength applied after local epochs.
        label_encoder:
            Optional fitted label encoder for classification.
        classes:
            Optional class tuple for partial_fit classifiers.
        random_state:
            Seed forwarded to sklearn estimators.
        """
        self.partition = partition
        self.global_template = global_template
        self.columns = columns
        self.target_column = target_column
        self.task = task
        self.estimator_key = estimator_key
        self.local_epochs = local_epochs
        self.mu = mu
        self.label_encoder = label_encoder
        self.classes = classes
        self.random_state = random_state
        self.last_metric: float | None = None

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return global linear parameters as Flower ndarray payloads.

        Serialises the current global template coefficients for Flower server
        round initialization and client synchronization.

        Parameters
        ----------
        config:
            Flower server config dict (ignored).

        Returns
        -------
        list[numpy.ndarray]
            Flattened ``coef_`` and ``intercept_`` arrays for aggregation.
        """
        _ = config
        return _linear_ndarrays(self.global_template)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Run a local client update and return Flower fit results.

        Applies local SGD or full fit starting from server parameters, optionally
        with FedProx proximal pull, and reports sample-weighted metrics.

        Parameters
        ----------
        parameters:
            Global model parameters from the Flower server.
        config:
            Flower server config dict (ignored).

        Returns
        -------
        tuple[list[numpy.ndarray], int, dict[str, float]]
            Updated parameters, example count, and optional train metrics.
        """
        from buildml.federated.fit import _local_update

        _ = config
        global_params = _ndarrays_to_linear(
            parameters,
            template=self.global_template,
        )
        local_est, metric, _notes = _local_update(
            self.global_template,
            global_params,
            self.partition.frame,
            columns=self.columns,
            target_column=self.target_column,
            task=self.task,
            estimator_key=self.estimator_key,
            local_epochs=self.local_epochs,
            mu=self.mu,
            label_encoder=self.label_encoder,
            classes=self.classes,
            random_state=self.random_state,
        )
        self.last_metric = metric
        ndarrays = _linear_ndarrays(local_est)
        metrics: dict[str, float] = {}
        if metric is not None:
            metrics["train_metric"] = float(metric)
        return ndarrays, self.partition.n_rows, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Evaluate the global model on the client holdout slice.

        Returns Flower loss as ``1 - local_score`` so higher accuracy yields
        lower reported loss.

        Parameters
        ----------
        parameters:
            Global model parameters from the Flower server.
        config:
            Flower server config dict (ignored).

        Returns
        -------
        tuple[float, int, dict[str, float]]
            Loss value, example count, and an empty metrics dict.
        """
        from buildml.federated.fit import _make_estimator
        from buildml.federated.features import encode_labels

        _ = config
        params = _ndarrays_to_linear(parameters, template=self.global_template)
        est = _make_estimator(self.estimator_key, self.random_state)
        set_linear_params(est, params)
        if self.classes is not None and hasattr(est, "classes_"):
            est.classes_ = np.asarray(self.label_encoder.classes_)
        x = matrix_from_frame(self.partition.frame, self.columns)
        if self.task == "classification":
            y, _, _ = encode_labels(
                self.partition.frame[self.target_column],
                label_encoder=self.label_encoder,
            )
        else:
            y = self.partition.frame[self.target_column].to_numpy(dtype=float)
        loss = 1.0 - (_score_local(est, x, y, task=self.task) or 0.0)
        return float(loss), self.partition.n_rows, {}


def fit_flower(
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
    """Run federated rounds via Flower NumPyClient + flwr aggregation (local sim).

    Honesty
    -------
    Uses ``flwr`` NumPyClient wrappers over Session client partitions and
    Flower's weighted ``aggregate`` helper. This still executes in-process on
    Session data — not a networked Flower deployment unless you operate one
    separately. No cryptographic secure aggregation.

    Parameters
    ----------
    dataset:
        BuildML dataset with features, target, and client columns.
    split_plan:
        Train/validation/test split; train partition is used for local updates.
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
        When ``flwr`` is not installed (``federated-industry`` extra).
    """
    require_flwr(feature="Flower federated backend")
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

    partitions = [
        _ClientPartition(
            client_id=cid,
            frame=frame_for_client(ctx.train, ctx.client_col, cid),
            n_rows=int((ctx.train[ctx.client_col] == cid).sum()),
        )
        for cid in ctx.eligible
    ]
    partition_by_id = {p.client_id: p for p in partitions}

    rng = np.random.default_rng(random_state)
    round_history: list[dict[str, Any]] = []
    final_metric: float | None = None
    global_ndarrays = _linear_ndarrays(ctx.global_est)

    for round_idx in range(int(n_rounds)):
        selected = _sample_clients(ctx.eligible, float(client_fraction), rng)
        fit_results: list[tuple[list[np.ndarray], int]] = []
        client_metrics: list[float] = []
        participated: list[Any] = []
        weights: list[float] = []

        for cid in selected:
            part = partition_by_id.get(cid)
            if part is None or part.n_rows < int(min_client_rows):
                continue
            client = _BuildMLSklearnClient(
                part,
                global_template=ctx.global_est,
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
            updated, n_examples, metrics = client.fit(global_ndarrays, {})
            fit_results.append((updated, int(n_examples)))
            participated.append(cid)
            weights.append(float(n_examples))
            if "train_metric" in metrics:
                client_metrics.append(float(metrics["train_metric"]))

        if not fit_results:
            ctx.warnings.append(
                f"Round {round_idx + 1}: no clients produced updates; stopping early."
            )
            break

        global_ndarrays = aggregate(fit_results)
        set_linear_params(
            ctx.global_est,
            _ndarrays_to_linear(global_ndarrays, template=ctx.global_est),
        )
        if ctx.classes_tuple is not None and hasattr(ctx.global_est, "classes_"):
            ctx.global_est.classes_ = np.asarray(ctx.label_encoder.classes_)

        mean_client = float(np.mean(client_metrics)) if client_metrics else None
        final_metric = mean_client
        total_weight = float(sum(weights))
        round_history.append(
            {
                "round": round_idx + 1,
                "backend": "flower",
                "n_clients": len(fit_results),
                "n_samples": int(sum(weights)),
                "total_weight": total_weight,
                "weighting": "sample_size",
                "aggregation": "flwr.server.strategy.aggregate",
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
            "Flower backend: NumPyClient local fit on Session client partitions "
            "+ flwr weighted ndarray aggregation.",
            "Honesty: still an in-process simulation on Session data — not a "
            "networked Flower deployment unless you operate one separately.",
            "No cryptographic secure aggregation; orchestrator sees client updates.",
            f"backend=flower, flwr aggregation rounds={len(round_history)}.",
        ]
    )
    if ctx.method_key == "fedprox":
        ctx.disclosures.append(
            f"FedProx proximal pull applied locally after each client epoch "
            f"(mu={ctx.mu}); aggregation uses Flower weighted averaging."
        )

    config = FederatedConfig(
        backend="flower",
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
        backend="flower",
        config=config,
    )
