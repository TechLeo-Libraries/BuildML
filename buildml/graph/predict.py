"""Predict with a fitted GraphPlan."""

from __future__ import annotations

from typing import Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.graph.data import (
    build_adjacency,
    edge_pairs_as_indices,
    filter_edges_for_mode,
    matrix_from_frame,
    node_index_map,
    normalize_adjacency,
    partition_node_mask,
)
from buildml.graph.features import build_classical_design, compute_graph_metrics
from buildml.graph.results import GraphPlan, GraphPredictResult

PartitionOrAll = Literal["train", "validation", "test", "all"]


def predict_graph(
    dataset: Dataset,
    plan: GraphPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> GraphPredictResult:
    """Score nodes in ``partition`` with a fitted :class:`GraphPlan`.

    Applies mode-aware edge filtering at score time and never refits the
    learner. Labels on holdout nodes are not used during forward passes.

    Parameters
    ----------
    dataset:
        Session dataset holding the node table.
    plan:
        Fitted :class:`GraphPlan` from :func:`fit_graph`.
    split_plan:
        Session split plan defining partition indices.
    partition:
        Partition to score, or ``"all"`` for every node.

    Returns
    -------
    GraphPredictResult
        Predictions, optional probabilities, and honesty disclosures.

    Raises
    ------
    ValidationError
        When no plan exists, split is missing, partition is empty, or the
        backend is unknown.
    """
    if plan is None:
        raise ValidationError("No GraphPlan. Call fit_graph(...) first.")
    if plan.graph_spec is None:
        raise ValidationError("GraphPlan is missing graph_spec; cannot predict.")
    if split_plan is None and partition != "all":
        raise ValidationError("No split plan; pass partition='all' or split first.")

    frame = dataset._ensure_pandas()
    n_nodes = int(len(frame))
    if partition == "all":
        score_mask = np.ones(n_nodes, dtype=bool)
    else:
        assert split_plan is not None
        score_mask = partition_node_mask(n_nodes, split_plan, partition)
    if int(score_mask.sum()) == 0:
        raise ValidationError(f"Partition {partition!r} is empty.")

    train_mask = (
        np.ones(n_nodes, dtype=bool)
        if split_plan is None
        else partition_node_mask(n_nodes, split_plan, "train")
    )
    snapshot = None
    if plan.graph_spec is not None and plan.graph_spec.node_ids_:
        snapshot = plan.graph_spec.node_ids_
    id_map = node_index_map(
        dataset, plan.node_id_col, node_ids_snapshot=snapshot
    )
    src_all, dst_all, map_disc = edge_pairs_as_indices(plan.graph_spec, id_map)
    src, dst, mode_disc = filter_edges_for_mode(
        src_all,
        dst_all,
        train_mask=train_mask,
        mode=plan.mode,  # type: ignore[arg-type]
        for_fit=False if plan.mode == "inductive" else True,
    )
    # Transductive scoring should use full edges; for_fit=True with
    # transductive returns all edges (filter short-circuits). For inductive
    # scoring for_fit=False keeps train↔holdout.
    if plan.mode == "transductive":
        src, dst = src_all, dst_all

    disclosures: list[str] = list(map_disc) + list(mode_disc)
    warnings: list[str] = []

    if plan.method == "classical":
        preds, proba, more_disc = _predict_classical(
            frame, plan, src, dst, train_mask, score_mask
        )
        disclosures.extend(more_disc)
    elif plan.method == "gcn":
        preds, proba, more_disc = _predict_gcn(
            frame, plan, src, dst, score_mask
        )
        disclosures.extend(more_disc)
    elif plan.method == "pyg":
        preds, proba, more_disc = _predict_pyg(
            frame, plan, src, dst, score_mask
        )
        disclosures.extend(more_disc)
    else:
        raise ValidationError(f"Unknown plan.method={plan.method!r}.")

    return GraphPredictResult(
        partition=partition,
        method=plan.method,
        mode=plan.mode,
        n_nodes=int(score_mask.sum()),
        predictions=tuple(preds.tolist()),
        probabilities=None
        if proba is None
        else tuple(tuple(float(v) for v in row) for row in proba.tolist()),
        classes_=plan.classes_,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _predict_classical(
    frame,
    plan: GraphPlan,
    src: np.ndarray,
    dst: np.ndarray,
    train_mask: np.ndarray,
    score_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    disclosures: list[str] = []
    n_nodes = len(frame)
    tabular = matrix_from_frame(frame, list(plan.feature_columns))
    graph_feats = None
    graph_names: list[str] = list(plan.graph_metric_names)
    if plan.graph_metric_names:
        graph_feats, names, g_disc = compute_graph_metrics(
            n_nodes,
            src,
            dst,
            directed=plan.directed,
            mode=plan.mode,  # type: ignore[arg-type]
            train_mask=train_mask,
        )
        disclosures.extend(g_disc)
        # Align to training metric order when possible.
        if list(names) != list(plan.graph_metric_names):
            disclosures.append(
                "Graph metric names at score time differ from fit; "
                "using score-time order aligned by name."
            )
            name_to_col = {n: graph_feats[:, i] for i, n in enumerate(names)}
            graph_feats = np.column_stack(
                [name_to_col[n] for n in plan.graph_metric_names]
            )
            graph_names = list(plan.graph_metric_names)
    X, _ = build_classical_design(
        tabular, list(plan.feature_columns), graph_feats, graph_names
    )
    if list(plan.design_feature_names) and X.shape[1] != len(plan.design_feature_names):
        raise ValidationError(
            "Classical design width at predict time does not match the plan. "
            f"expected={len(plan.design_feature_names)} got={X.shape[1]}."
        )
    model = plan.estimator_
    encoder = plan.label_encoder_
    X_score = X[score_mask]
    pred_idx = model.predict(X_score)
    preds = encoder.inverse_transform(pred_idx)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_score), dtype=np.float64)
    return preds, proba, disclosures


def _predict_gcn(
    frame,
    plan: GraphPlan,
    src: np.ndarray,
    dst: np.ndarray,
    score_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    disclosures = [
        "GCN forward pass on mode-filtered adjacency; labels unused.",
    ]
    n_nodes = len(frame)
    x = matrix_from_frame(frame, list(plan.feature_columns))
    adj = build_adjacency(n_nodes, src, dst, directed=plan.directed)
    adj_norm = normalize_adjacency(adj)
    gcn = plan.estimator_
    proba_all = gcn.predict_proba(x, adj_norm)
    pred_idx_all = proba_all.argmax(axis=1)
    classes = list(plan.classes_)
    preds = np.asarray([classes[int(i)] for i in pred_idx_all[score_mask]], dtype=object)
    proba = proba_all[score_mask]
    return preds, proba, disclosures


def _predict_pyg(
    frame,
    plan: GraphPlan,
    src: np.ndarray,
    dst: np.ndarray,
    score_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    pyg_model = plan.config.get("pyg_model", "gcn")
    disclosures = [
        f"PyG {pyg_model} forward on mode-filtered edge_index; labels unused.",
    ]
    x = matrix_from_frame(frame, list(plan.feature_columns))
    pyg_clf = plan.estimator_
    proba_all = pyg_clf.predict_proba(
        x, src, dst, directed=plan.directed
    )
    pred_idx_all = proba_all.argmax(axis=1)
    classes = list(plan.classes_)
    preds = np.asarray(
        [classes[int(i)] for i in pred_idx_all[score_mask]], dtype=object
    )
    proba = proba_all[score_mask]
    return preds, proba, disclosures
