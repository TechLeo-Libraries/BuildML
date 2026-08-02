"""Fit Graph ML learners (classical NetworkX+sklearn or pure-Torch GCN)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.graph.data import (
    build_adjacency,
    edge_pairs_as_indices,
    filter_edges_for_mode,
    matrix_from_frame,
    node_index_map,
    normalize_adjacency,
    partition_node_mask,
    resolve_feature_columns,
    target_array,
)
from buildml.graph.features import build_classical_design, compute_graph_metrics
from buildml.graph.gnn import GCNClassifier
from buildml.graph.results import GraphFitResult, GraphPlan
from buildml.graph.types import (
    ClassicalEstimator,
    GraphConfig,
    GraphMethod,
    GraphMode,
    GraphSpec,
    GraphTask,
)


def fit_graph(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    graph_spec: GraphSpec,
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
) -> tuple[GraphPlan, GraphFitResult]:
    """Fit a node classifier with leakage-aware graph structure.

    Honesty
    -------
    Requires ``set_graph`` (:class:`GraphSpec`) and a Session split. Labels for
    supervision come from the train partition only. Inductive mode fits on the
    train-induced subgraph; transductive mode uses full topology with
    train-label-only loss / sklearn fit rows.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    graph_spec.validate()

    method_key = str(method).lower().replace("-", "_")
    if method_key not in {"classical", "gcn"}:
        raise ValidationError(
            f"Unknown graph method={method!r}. Supported: classical, gcn."
        )
    if task != "node_classification":
        raise ValidationError(
            f"Unsupported graph task={task!r}. "
            "This surface currently ships node_classification only."
        )
    if str(mode) not in {"inductive", "transductive"}:
        raise ValidationError(
            f"Unknown graph mode={mode!r}. Supported: inductive, transductive."
        )
    mode_key: GraphMode = "inductive" if str(mode) == "inductive" else "transductive"

    frame = dataset._ensure_pandas()
    targets = dataset.role_columns(ColumnRole.TARGET)
    if len(targets) != 1:
        raise ValidationError(
            "Graph node classification requires exactly one target column."
        )
    target_column = str(targets[0])
    n_nodes = int(len(frame))

    id_map = node_index_map(
        dataset,
        graph_spec.node_id_col,
        node_ids_snapshot=graph_spec.node_ids_ or None,
    )
    src_all, dst_all, map_disc = edge_pairs_as_indices(graph_spec, id_map)
    train_mask = partition_node_mask(n_nodes, split_plan, "train")
    src_fit, dst_fit, mode_disc = filter_edges_for_mode(
        src_all,
        dst_all,
        train_mask=train_mask,
        mode=mode_key,
        for_fit=True,
    )
    if len(src_fit) == 0:
        raise ValidationError(
            "No edges remain for fit under the chosen mode. "
            "Inductive mode needs train–train edges."
        )

    feat_cols, feat_disc = resolve_feature_columns(
        dataset,
        frame,
        None if columns is None else list(columns),
        node_id_col=graph_spec.node_id_col,
        target_column=target_column,
    )
    y_all = target_array(frame, target_column)
    y_train_raw = y_all[train_mask]
    if len(np.unique(y_train_raw)) < 2:
        raise ValidationError(
            "Train partition must contain at least two target classes."
        )

    disclosures: list[str] = list(graph_spec.disclosures)
    disclosures.extend(map_disc)
    disclosures.extend(mode_disc)
    disclosures.extend(feat_disc)
    disclosures.extend(
        [
            f"method={method_key}; mode={mode_key}; task={task}.",
            "Supervision uses train-node labels only.",
            "Validation/test labels are never used for fitting.",
        ]
    )
    warnings: list[str] = []

    encoder = LabelEncoder()
    encoder.fit(y_train_raw)
    classes = tuple(encoder.classes_.tolist())

    config = GraphConfig(
        method=method_key,  # type: ignore[arg-type]
        task=task,
        mode=mode_key,
        columns=None if columns is None else tuple(str(c) for c in columns),
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

    if method_key == "classical":
        return _fit_classical(
            frame=frame,
            feat_cols=feat_cols,
            target_column=target_column,
            y_all=y_all,
            train_mask=train_mask,
            src_fit=src_fit,
            dst_fit=dst_fit,
            graph_spec=graph_spec,
            mode_key=mode_key,
            encoder=encoder,
            classes=classes,
            classical_estimator=classical_estimator,
            include_graph_metrics=include_graph_metrics,
            random_state=random_state,
            n_edges_fit=len(src_fit),
            disclosures=disclosures,
            warnings=warnings,
            config=config,
        )

    return _fit_gcn(
        frame=frame,
        feat_cols=feat_cols,
        target_column=target_column,
        y_all=y_all,
        train_mask=train_mask,
        src_fit=src_fit,
        dst_fit=dst_fit,
        graph_spec=graph_spec,
        mode_key=mode_key,
        encoder=encoder,
        classes=classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        random_state=random_state,
        n_edges_fit=len(src_fit),
        disclosures=disclosures,
        warnings=warnings,
        config=config,
    )


def _fit_classical(
    *,
    frame,
    feat_cols: list[str],
    target_column: str,
    y_all: np.ndarray,
    train_mask: np.ndarray,
    src_fit: np.ndarray,
    dst_fit: np.ndarray,
    graph_spec: GraphSpec,
    mode_key: GraphMode,
    encoder: LabelEncoder,
    classes: tuple[Any, ...],
    classical_estimator: ClassicalEstimator,
    include_graph_metrics: bool,
    random_state: int | None,
    n_edges_fit: int,
    disclosures: list[str],
    warnings: list[str],
    config: GraphConfig,
) -> tuple[GraphPlan, GraphFitResult]:
    n_nodes = len(frame)
    tabular = matrix_from_frame(frame, feat_cols)
    graph_feats = None
    graph_names: list[str] = []
    if include_graph_metrics:
        graph_feats, graph_names, g_disc = compute_graph_metrics(
            n_nodes,
            src_fit,
            dst_fit,
            directed=graph_spec.directed,
            mode=mode_key,
            train_mask=train_mask,
        )
        disclosures.extend(g_disc)
    else:
        disclosures.append("include_graph_metrics=False; tabular features only.")

    X, design_names = build_classical_design(
        tabular, feat_cols, graph_feats, graph_names
    )
    X_train = X[train_mask]
    y_train = encoder.transform(y_all[train_mask])

    est_name = str(classical_estimator)
    if est_name == "random_forest":
        model: Any = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=random_state),
        )
        est_name = "logistic_regression"
    model.fit(X_train, y_train)
    train_acc = float(np.mean(model.predict(X_train) == y_train))

    plan = GraphPlan(
        method="classical",
        task="node_classification",
        mode=mode_key,
        node_id_col=graph_spec.node_id_col,
        feature_columns=tuple(feat_cols),
        graph_metric_names=tuple(graph_names),
        design_feature_names=tuple(design_names),
        target_column=target_column,
        classes_=classes,
        n_train_nodes=int(train_mask.sum()),
        n_edges_fit=n_edges_fit,
        directed=graph_spec.directed,
        estimator_name=est_name,
        estimator_=model,
        label_encoder_=encoder,
        graph_spec=graph_spec,
        adj_norm_fit_=None,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    fit_result = GraphFitResult(
        method="classical",
        mode=mode_key,
        task="node_classification",
        n_train_nodes=int(train_mask.sum()),
        n_edges_fit=n_edges_fit,
        n_classes=len(classes),
        train_accuracy=train_acc,
        train_loss_last=None,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, fit_result


def _fit_gcn(
    *,
    frame,
    feat_cols: list[str],
    target_column: str,
    y_all: np.ndarray,
    train_mask: np.ndarray,
    src_fit: np.ndarray,
    dst_fit: np.ndarray,
    graph_spec: GraphSpec,
    mode_key: GraphMode,
    encoder: LabelEncoder,
    classes: tuple[Any, ...],
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    random_state: int | None,
    n_edges_fit: int,
    disclosures: list[str],
    warnings: list[str],
    config: GraphConfig,
) -> tuple[GraphPlan, GraphFitResult]:
    n_nodes = len(frame)
    if not feat_cols:
        raise ValidationError(
            "GCN requires at least one numeric tabular node feature column. "
            "Encode/scale features first, or use method='classical' with "
            "include_graph_metrics=True."
        )
    x = matrix_from_frame(frame, feat_cols)
    adj = build_adjacency(
        n_nodes, src_fit, dst_fit, directed=graph_spec.directed
    )
    adj_norm = normalize_adjacency(adj)
    disclosures.append(
        "Pure-Torch GCN (no PyTorch Geometric): symmetric normalized "
        "adjacency with self-loops; train-mask cross-entropy only."
    )

    class_to_index = {c: i for i, c in enumerate(classes)}
    gcn = GCNClassifier(
        in_dim=x.shape[1],
        n_classes=len(classes),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        random_state=random_state,
    )
    gcn.fit(x, y_all, adj_norm, train_mask, class_to_index)
    pred_idx = gcn.predict(x, adj_norm)
    y_train_idx = np.asarray(
        [class_to_index[v] for v in y_all[train_mask].tolist()], dtype=np.int64
    )
    train_acc = float(np.mean(pred_idx[train_mask] == y_train_idx))
    train_loss_last = (
        None if not gcn.train_losses_ else float(gcn.train_losses_[-1])
    )

    plan = GraphPlan(
        method="gcn",
        task="node_classification",
        mode=mode_key,
        node_id_col=graph_spec.node_id_col,
        feature_columns=tuple(feat_cols),
        graph_metric_names=(),
        design_feature_names=tuple(feat_cols),
        target_column=target_column,
        classes_=classes,
        n_train_nodes=int(train_mask.sum()),
        n_edges_fit=n_edges_fit,
        directed=graph_spec.directed,
        estimator_name="gcn",
        estimator_=gcn,
        label_encoder_=encoder,
        graph_spec=graph_spec,
        adj_norm_fit_=adj_norm,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    fit_result = GraphFitResult(
        method="gcn",
        mode=mode_key,
        task="node_classification",
        n_train_nodes=int(train_mask.sum()),
        n_edges_fit=n_edges_fit,
        n_classes=len(classes),
        train_accuracy=train_acc,
        train_loss_last=train_loss_last,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, fit_result
