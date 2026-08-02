"""Thin Session facades over buildml.graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.graph.checkpoint import load_graph_bundle, save_graph_bundle
from buildml.graph.data import build_graph_spec
from buildml.graph.evaluate import evaluate_graph
from buildml.graph.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    graph_spec_summary,
    predict_result_summary,
)
from buildml.graph.fit import fit_graph
from buildml.graph.predict import predict_graph
from buildml.graph.types import (
    ClassicalEstimator,
    GraphMethod,
    GraphMode,
    GraphSpec,
    GraphTask,
)

PartitionOrAll = PartitionName | Literal["all"]


def set_graph_op(
    session,
    edges: pd.DataFrame | Sequence[tuple[Any, Any]],
    *,
    source_col: str = "source",
    target_col: str = "target",
    node_id_col: str = "node_id",
    directed: bool = False,
) -> GraphSpec:
    """Attach an edge list to the Session (nodes = dataset rows).

    Notes
    -----
    Dataset rows are nodes. ``node_id_col`` must uniquely identify rows and
    match edge endpoints. Splits created with :meth:`Session.split` are node
    partitions. Call this before :meth:`Session.fit_graph`.
    """
    if session.dataset is None:
        raise ValidationError("Ingest a dataset before set_graph.")
    spec = build_graph_spec(
        edges,
        source_col=source_col,
        target_col=target_col,
        node_id_col=node_id_col,
        directed=directed,
    )
    # Validate node ids exist / unique early; snapshot values so later
    # preprocess (e.g. scale on numeric id) cannot break edge matching.
    from buildml.graph.data import node_id_series

    series = node_id_series(session.dataset, node_id_col)
    spec.node_ids_ = tuple(series.tolist())
    session._graph_spec = spec
    session._graph_plan = None
    session._graph_fit_result = None
    session._graph_predict_result = None
    session._graph_eval_result = None
    session._record(
        "set_graph",
        {
            "source_col": source_col,
            "target_col": target_col,
            "node_id_col": node_id_col,
            "directed": directed,
            "n_edges": spec.n_edges,
            "n_nodes_in_edges": spec.n_nodes_in_edges,
        },
        warnings=(),
        result_summary=graph_spec_summary(spec),
    )
    return spec


def fit_graph_op(
    session,
    *,
    method: GraphMethod = "classical",
    task: GraphTask = "node_classification",
    mode: GraphMode = "inductive",
    columns: Sequence[str] | None = None,
    classical_estimator: ClassicalEstimator = "logistic_regression",
    hidden_dim: int = 32,
    n_layers: int = 2,
    epochs: int = 80,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.1,
    random_state: int | None = 0,
    include_graph_metrics: bool = True,
) -> Any:
    """Fit graph node classification on Session train nodes.

    Notes
    -----
    **Leakage:** Requires a split. Labels from train only.
    **Inductive (default):** train-induced subgraph for fit.
    **Transductive:** full topology; train-label-only supervision (disclosed).
    Classical needs ``buildml[graph]``; GCN needs ``buildml[torch]``.
    """
    spec = getattr(session, "_graph_spec", None)
    if spec is None:
        raise ValidationError(
            "No GraphSpec. Call set_graph(edges, node_id_col=...) first."
        )
    session.assert_can_fit("train")
    plan, result = fit_graph(
        session.dataset,
        session._split_plan,
        spec,
        method=method,
        task=task,
        mode=mode,
        columns=columns,
        classical_estimator=classical_estimator,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        random_state=random_state,
        include_graph_metrics=include_graph_metrics,
    )
    session._graph_plan = plan
    session._graph_fit_result = result
    session._graph_predict_result = None
    session._graph_eval_result = None
    session._record(
        "fit_graph",
        {
            "method": method,
            "task": task,
            "mode": mode,
            "columns": None if columns is None else list(columns),
            "classical_estimator": classical_estimator,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "random_state": random_state,
            "include_graph_metrics": include_graph_metrics,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def predict_graph_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Predict node labels with the fitted GraphPlan."""
    plan = getattr(session, "_graph_plan", None)
    if plan is None:
        raise ValidationError("No graph plan. Call fit_graph(...) first.")
    result = predict_graph(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
    )
    session._graph_predict_result = result
    session._record(
        "predict_graph",
        {"partition": partition},
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def evaluate_graph_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Evaluate node classification on a holdout partition."""
    plan = getattr(session, "_graph_plan", None)
    if plan is None:
        raise ValidationError("No graph plan. Call fit_graph(...) first.")
    result = evaluate_graph(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
    )
    session._graph_eval_result = result
    session._record(
        "evaluate_graph",
        {"partition": partition},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_graph_bundle_op(session, path: str | Path) -> Path:
    """Persist the active GraphPlan as ``buildml.graph_bundle.v1``."""
    plan = getattr(session, "_graph_plan", None)
    if plan is None:
        raise ValidationError("No graph plan. Call fit_graph(...) first.")
    out = save_graph_bundle(
        path,
        plan,
        fit_result=getattr(session, "_graph_fit_result", None),
        eval_result=getattr(session, "_graph_eval_result", None),
    )
    session._record(
        "save_graph_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.graph_bundle.v1"},
    )
    return out


def load_graph_bundle_op(session, path: str | Path):
    """Load a graph bundle into this Session."""
    plan = load_graph_bundle(path)
    session._graph_plan = plan
    session._graph_spec = plan.graph_spec
    session._graph_fit_result = None
    session._graph_predict_result = None
    session._graph_eval_result = None
    session._record(
        "load_graph_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "method": plan.method,
            "mode": plan.mode,
            "n_train_nodes": plan.n_train_nodes,
        },
    )
    return session
