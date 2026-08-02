"""Typed results for Session-facing Graph ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.graph.types import GraphSpec


@dataclass(slots=True)
class GraphPlan:
    """Fitted graph-learning plan (classical or GCN).

    Persist via ``buildml.graph_bundle.v1``. Distinct from Session checkpoints.
    Honesty: node classification with classical NetworkX metrics + sklearn
    and/or a small pure-Torch GCN and/or PyG conv layers — not Neo4j/KG.
    """

    method: str
    task: str
    mode: str
    node_id_col: str
    feature_columns: tuple[str, ...]
    graph_metric_names: tuple[str, ...]
    design_feature_names: tuple[str, ...]
    target_column: str
    classes_: tuple[Any, ...]
    n_train_nodes: int
    n_edges_fit: int
    directed: bool
    estimator_name: str
    estimator_: Any = field(repr=False, default=None)
    label_encoder_: Any = field(repr=False, default=None)
    graph_spec: GraphSpec | None = field(repr=False, default=None)
    adj_norm_fit_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "mode": self.mode,
            "node_id_col": self.node_id_col,
            "feature_columns": list(self.feature_columns),
            "graph_metric_names": list(self.graph_metric_names),
            "design_feature_names": list(self.design_feature_names),
            "target_column": self.target_column,
            "classes_": list(self.classes_),
            "n_train_nodes": self.n_train_nodes,
            "n_edges_fit": self.n_edges_fit,
            "directed": self.directed,
            "estimator_name": self.estimator_name,
            "graph_spec": None if self.graph_spec is None else self.graph_spec.to_dict(),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class GraphFitResult:
    """Outcome of fitting a graph learner on Session train nodes."""

    method: str
    mode: str
    task: str
    n_train_nodes: int
    n_edges_fit: int
    n_classes: int
    train_accuracy: float | None = None
    train_loss_last: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mode": self.mode,
            "task": self.task,
            "n_train_nodes": self.n_train_nodes,
            "n_edges_fit": self.n_edges_fit,
            "n_classes": self.n_classes,
            "train_accuracy": self.train_accuracy,
            "train_loss_last": self.train_loss_last,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class GraphPredictResult:
    """Node predictions for a partition."""

    partition: str
    method: str
    mode: str
    n_nodes: int
    predictions: tuple[Any, ...]
    probabilities: tuple[tuple[float, ...], ...] | None = None
    classes_: tuple[Any, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "mode": self.mode,
            "n_nodes": self.n_nodes,
            "predictions": list(self.predictions),
            "probabilities": None
            if self.probabilities is None
            else [list(row) for row in self.probabilities],
            "classes_": list(self.classes_),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class GraphEvalResult:
    """Holdout evaluation for node classification."""

    partition: str
    method: str
    mode: str
    n_nodes: int
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "mode": self.mode,
            "n_nodes": self.n_nodes,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
