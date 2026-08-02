"""Configuration and graph-data types for Session-facing Graph ML.

Conventions
-----------
- Session rows are **nodes**. One row per node.
- ``node_id_col`` identifies nodes (role ``id`` recommended; otherwise a
  feature/ignore column). Values must be unique and match edge endpoints.
- Edges are a separate table with ``source_col`` / ``target_col`` referencing
  ``node_id`` values (not row positions).
- Train / validation / test splits are **node** splits via Session.split.
- Graph structure is attached with :meth:`Session.set_graph` before fit.

Inductive vs transductive
-------------------------
- ``inductive`` (default): message-passing / NetworkX metrics for fitting use
  only edges whose **both** endpoints are train nodes. Holdout scoring may use
  edges into the train set (train↔holdout) but never holdout labels for fit.
- ``transductive``: the full graph topology participates in aggregation /
  feature computation; supervision remains train-label-only. Documented as
  allowing holdout node features to influence train embeddings via edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import ValidationError

GraphMethod = Literal["classical", "gcn"]
GraphTask = Literal["node_classification"]
GraphMode = Literal["inductive", "transductive"]
ClassicalEstimator = Literal["logistic_regression", "random_forest"]


@dataclass(slots=True)
class GraphSpec:
    """Normalized edge list + node-id convention attached to a Session."""

    edges: pd.DataFrame = field(repr=False)
    source_col: str = "source"
    target_col: str = "target"
    node_id_col: str = "node_id"
    directed: bool = False
    n_edges: int = 0
    n_nodes_in_edges: int = 0
    # Snapshot of node_id values at set_graph time (row-aligned). Survives
    # later preprocess that accidentally mutates the id column (e.g. scale).
    node_ids_: tuple[Any, ...] = field(default=(), repr=False)
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_col": self.source_col,
            "target_col": self.target_col,
            "node_id_col": self.node_id_col,
            "directed": self.directed,
            "n_edges": self.n_edges,
            "n_nodes_in_edges": self.n_nodes_in_edges,
            "n_node_ids_snapshot": len(self.node_ids_),
            "disclosures": list(self.disclosures),
        }

    def validate(self) -> None:
        if self.edges is None or self.edges.empty:
            raise ValidationError("GraphSpec.edges must be a non-empty DataFrame.")
        for col in (self.source_col, self.target_col):
            if col not in self.edges.columns:
                raise ValidationError(
                    f"GraphSpec edges missing column {col!r}. "
                    "Pass source_col/target_col matching the edge table."
                )
        if self.n_edges <= 0:
            raise ValidationError("GraphSpec has zero edges after normalization.")
        if not str(self.node_id_col).strip():
            raise ValidationError("GraphSpec.node_id_col is required.")


@dataclass(slots=True)
class GraphConfig:
    """User-facing Graph ML knobs (serializable summary)."""

    method: GraphMethod = "classical"
    task: GraphTask = "node_classification"
    mode: GraphMode = "inductive"
    columns: tuple[str, ...] | None = None
    classical_estimator: ClassicalEstimator = "logistic_regression"
    hidden_dim: int = 32
    n_layers: int = 2
    epochs: int = 80
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.1
    random_state: int | None = 0
    include_graph_metrics: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "mode": self.mode,
            "columns": None if self.columns is None else list(self.columns),
            "classical_estimator": self.classical_estimator,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "random_state": self.random_state,
            "include_graph_metrics": self.include_graph_metrics,
        }
