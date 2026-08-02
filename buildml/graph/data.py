"""Graph data helpers: edge normalization, node maps, adjacency, masks."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.graph.types import GraphMode, GraphSpec

__all__ = [
    "normalize_edges",
    "build_graph_spec",
    "node_id_series",
    "node_index_map",
    "partition_node_mask",
    "edge_pairs_as_indices",
    "filter_edges_for_mode",
    "build_adjacency",
    "normalize_adjacency",
    "resolve_feature_columns",
    "matrix_from_frame",
    "target_array",
    "train_partition_frame",
    "partition_frame",
]


def normalize_edges(
    edges: pd.DataFrame | Sequence[tuple[Any, Any]] | np.ndarray,
    *,
    source_col: str = "source",
    target_col: str = "target",
    directed: bool = False,
) -> pd.DataFrame:
    """Normalize an edge list to columns ``source_col`` / ``target_col``."""
    if isinstance(edges, pd.DataFrame):
        if source_col not in edges.columns or target_col not in edges.columns:
            raise ValidationError(
                f"Edge DataFrame must contain columns {source_col!r} and "
                f"{target_col!r}."
            )
        frame = edges[[source_col, target_col]].copy()
    else:
        arr = np.asarray(edges, dtype=object)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValidationError(
                "Edges must be a DataFrame or an (n_edges, >=2) array/sequence "
                "of (source, target) pairs."
            )
        frame = pd.DataFrame(
            {source_col: arr[:, 0], target_col: arr[:, 1]},
        )
    frame = frame.dropna()
    if frame.empty:
        raise ValidationError("Edge list is empty after dropping null endpoints.")
    # Drop self-loops for undirected / message-passing stability (disclose).
    self_loops = frame[source_col] == frame[target_col]
    n_loops = int(self_loops.sum())
    if n_loops:
        frame = frame.loc[~self_loops].copy()
    if not directed:
        a = frame[source_col].astype(str)
        b = frame[target_col].astype(str)
        src = np.where(a <= b, frame[source_col], frame[target_col])
        dst = np.where(a <= b, frame[target_col], frame[source_col])
        frame = pd.DataFrame({source_col: src, target_col: dst})
        frame = frame.drop_duplicates()
    else:
        frame = frame.drop_duplicates()
    if frame.empty:
        raise ValidationError("No edges remain after normalization.")
    frame.attrs["dropped_self_loops"] = n_loops
    return frame.reset_index(drop=True)


def build_graph_spec(
    edges: pd.DataFrame | Sequence[tuple[Any, Any]] | np.ndarray,
    *,
    source_col: str = "source",
    target_col: str = "target",
    node_id_col: str = "node_id",
    directed: bool = False,
) -> GraphSpec:
    """Build a validated :class:`GraphSpec` from a raw edge list."""
    frame = normalize_edges(
        edges, source_col=source_col, target_col=target_col, directed=directed
    )
    nodes = pd.unique(
        pd.concat([frame[source_col], frame[target_col]], ignore_index=True)
    )
    disclosures = [
        "Edges normalized to unique endpoint pairs; self-loops dropped.",
        f"directed={directed}; node_id_col={node_id_col!r}.",
        "Session rows are nodes; splits are node partitions.",
    ]
    n_loops = int(frame.attrs.get("dropped_self_loops", 0) or 0)
    if n_loops:
        disclosures.append(f"Dropped {n_loops} self-loop edge(s).")
    spec = GraphSpec(
        edges=frame,
        source_col=source_col,
        target_col=target_col,
        node_id_col=node_id_col,
        directed=directed,
        n_edges=int(len(frame)),
        n_nodes_in_edges=int(len(nodes)),
        disclosures=tuple(disclosures),
    )
    spec.validate()
    return spec


def node_id_series(dataset: Dataset, node_id_col: str) -> pd.Series:
    """Return the node-id series; enforce uniqueness."""
    frame = dataset._ensure_pandas()
    if node_id_col not in frame.columns:
        raise ValidationError(
            f"node_id_col={node_id_col!r} not found in the Session dataset. "
            "Add an id column or pass the correct node_id_col to set_graph."
        )
    series = frame[node_id_col]
    if series.isna().any():
        raise ValidationError(f"node_id_col={node_id_col!r} contains nulls.")
    if series.duplicated().any():
        raise ValidationError(
            f"node_id_col={node_id_col!r} must be unique (one row per node)."
        )
    return series


def node_index_map(
    dataset: Dataset,
    node_id_col: str,
    *,
    node_ids_snapshot: tuple[Any, ...] | None = None,
) -> dict[Any, int]:
    """Map node_id value → positional row index in the Session frame.

    Prefer ``node_ids_snapshot`` from :class:`GraphSpec` when present so edge
    matching survives later preprocess that mutates the id column.
    """
    if node_ids_snapshot is not None and len(node_ids_snapshot) > 0:
        frame = dataset._ensure_pandas()
        if len(node_ids_snapshot) != len(frame):
            raise ValidationError(
                "GraphSpec node_ids_ snapshot length does not match the "
                f"current dataset ({len(node_ids_snapshot)} vs {len(frame)}). "
                "Call set_graph again after row-level dataset changes."
            )
        return {v: int(i) for i, v in enumerate(node_ids_snapshot)}
    series = node_id_series(dataset, node_id_col)
    return {v: int(i) for i, v in enumerate(series.tolist())}


def partition_node_mask(
    n_nodes: int,
    split_plan: SplitPlan,
    partition: str,
) -> np.ndarray:
    """Boolean mask over all Session rows for a partition (or all)."""
    mask = np.zeros(n_nodes, dtype=bool)
    if partition == "all":
        mask[:] = True
        return mask
    indices = split_plan.indices_for(partition)  # type: ignore[arg-type]
    if not indices and partition == "validation":
        raise ValidationError("No validation partition exists on this split plan.")
    for i in indices:
        if 0 <= int(i) < n_nodes:
            mask[int(i)] = True
    return mask


def edge_pairs_as_indices(
    spec: GraphSpec,
    id_to_index: dict[Any, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert edge endpoints to row indices; drop edges with unknown nodes."""
    src_vals = spec.edges[spec.source_col].tolist()
    dst_vals = spec.edges[spec.target_col].tolist()
    src_idx: list[int] = []
    dst_idx: list[int] = []
    skipped = 0
    for s, t in zip(src_vals, dst_vals, strict=True):
        if s not in id_to_index or t not in id_to_index:
            skipped += 1
            continue
        src_idx.append(id_to_index[s])
        dst_idx.append(id_to_index[t])
    disclosures: list[str] = []
    if skipped:
        disclosures.append(
            f"Skipped {skipped} edge(s) whose endpoints are not in the "
            "Session node table."
        )
    if not src_idx:
        raise ValidationError(
            "No edges map to Session node ids. Ensure edge endpoints match "
            f"values in node_id_col={spec.node_id_col!r}."
        )
    return (
        np.asarray(src_idx, dtype=np.int64),
        np.asarray(dst_idx, dtype=np.int64),
        disclosures,
    )


def filter_edges_for_mode(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    train_mask: np.ndarray,
    mode: GraphMode,
    for_fit: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Filter edges according to inductive / transductive leakage rules.

    Parameters
    ----------
    for_fit:
        When True and mode is inductive, keep only train–train edges.
        When False (scoring) and inductive, keep edges with at least one
        train endpoint or both endpoints in the active graph (train∪score).
        Transductive keeps all edges always.
    """
    disclosures: list[str] = []
    if mode == "transductive":
        disclosures.append(
            "Transductive mode: full graph topology used for aggregation / "
            "graph features; supervision remains train-label-only."
        )
        return src, dst, disclosures

    both_train = train_mask[src] & train_mask[dst]
    if for_fit:
        disclosures.append(
            "Inductive fit: using only edges with both endpoints in the "
            "train partition (train-induced subgraph)."
        )
        return src[both_train], dst[both_train], disclosures

    # Scoring: allow train–train and train–holdout edges; drop holdout–holdout
    # so holdout topology alone cannot invent structure unseen at fit for
    # classical metrics / message passing into unlabeled cliques.
    keep = train_mask[src] | train_mask[dst]
    disclosures.append(
        "Inductive score: using edges with at least one train endpoint "
        "(train↔holdout allowed; holdout↔holdout dropped)."
    )
    return src[keep], dst[keep], disclosures


def build_adjacency(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    directed: bool,
) -> np.ndarray:
    """Dense float adjacency (small graphs). Self-loops added later for GCN."""
    if n_nodes <= 0:
        raise ValidationError("n_nodes must be positive.")
    if len(src) != len(dst):
        raise ValidationError("src/dst length mismatch.")
    # Guard memory for accidental huge graphs in this Session surface.
    if n_nodes > 5000:
        raise ValidationError(
            f"Graph has {n_nodes} nodes; this Session surface currently "
            "materializes a dense adjacency for clarity and is limited to "
            "5000 nodes. Filter or sample the graph for larger problems."
        )
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in zip(src.tolist(), dst.tolist(), strict=True):
        adj[i, j] = 1.0
        if not directed:
            adj[j, i] = 1.0
    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalized adjacency with self-loops: D^{-1/2}(A+I)D^{-1/2}."""
    a_hat = adj + np.eye(adj.shape[0], dtype=np.float64)
    deg = a_hat.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d = np.diag(deg_inv_sqrt)
    return d @ a_hat @ d


def resolve_feature_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
    *,
    node_id_col: str,
    target_column: str,
) -> tuple[list[str], list[str]]:
    """Resolve numeric tabular node-feature columns."""
    disclosures: list[str] = []
    protected = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    exclude = {target_column, node_id_col}
    if columns is not None:
        names = [str(c) for c in columns if str(c) not in exclude]
        missing = [c for c in names if c not in frame.columns]
        if missing:
            raise ValidationError(f"Unknown feature columns: {missing}.")
        names = [
            c
            for c in names
            if dataset.roles.get(c) not in protected
        ]
    else:
        feature_roles = dataset.role_columns(ColumnRole.FEATURE)
        candidates = feature_roles or [
            str(c)
            for c in frame.columns
            if dataset.roles.get(str(c)) not in protected
        ]
        names = [
            str(c)
            for c in candidates
            if c in frame.columns
            and c not in exclude
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        disclosures.append(
            f"Resolved {len(names)} numeric tabular node feature column(s)."
        )
    if not names:
        # Graph-metrics-only path is allowed when include_graph_metrics=True.
        disclosures.append(
            "No tabular node features resolved; graph metrics alone may be used."
        )
    else:
        for name in names:
            if frame[name].isna().any():
                raise ValidationError(
                    f"Feature column {name!r} contains nulls. Impute/scale first."
                )
            if not pd.api.types.is_numeric_dtype(frame[name]):
                raise ValidationError(
                    f"Feature column {name!r} must be numeric for Graph ML."
                )
    return names, disclosures


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a float design matrix from columns (empty → (n, 0))."""
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.float64)
    return frame[columns].to_numpy(dtype=np.float64, copy=True)


def target_array(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    if target_column not in frame.columns:
        raise ValidationError(f"Target column {target_column!r} missing.")
    series = frame[target_column]
    if series.isna().any():
        raise ValidationError("Target column contains nulls.")
    return series.to_numpy()


def train_partition_frame(dataset: Dataset, split_plan: SplitPlan) -> pd.DataFrame:
    assert_fit_partition(split_plan, "train")
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: str,
) -> pd.DataFrame:
    if partition == "all":
        return dataset._ensure_pandas().copy()
    if split_plan is None:
        raise ValidationError("No split plan; cannot select a partition.")
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]
