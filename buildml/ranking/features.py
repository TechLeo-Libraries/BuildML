"""Column resolution, feature matrices, and graded ranking metrics for LTR."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition

__all__ = [
    "resolve_ranking_columns",
    "train_partition_frame",
    "partition_frame",
    "feature_matrix",
    "standardize_fit",
    "standardize_apply",
    "query_group_sizes",
    "disclose_query_split",
    "ndcg_at_k_graded",
    "average_precision_at_k",
    "mrr_at_k",
    "mean_metric_over_queries",
]


def train_partition_frame(dataset: Dataset, split_plan: SplitPlan) -> pd.DataFrame:
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset, split_plan: SplitPlan | None, partition: str
) -> pd.DataFrame:
    if partition == "all":
        return dataset.frame.copy()
    if split_plan is None:
        raise ValidationError(
            "A SplitPlan is required for partitioned ranker evaluation."
        )
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def resolve_ranking_columns(
    dataset: Dataset,
    *,
    query_column: str | None,
    item_column: str | None,
    relevance_column: str | None,
    feature_columns: Sequence[str] | None,
) -> tuple[str, str, str, tuple[str, ...], list[str]]:
    """Resolve query / item / relevance / feature columns for tabular LTR.

    Conventions
    -----------
    - ``query_column`` / ``item_column`` are explicit kwargs (entity ids).
      Prefer ``role='group'`` on the query id so ``group_split`` can isolate
      queries across partitions.
    - ``relevance_column`` defaults to the Session target role (graded or
      binary relevance labels).
    - Feature columns default to all ``feature``-role numeric columns.
    - This path is **tabular learning-to-rank**, not RAG retrieve/generate
      and not Session recommenders (user–item CF).
    """
    frame = dataset.frame
    disclosures: list[str] = []

    if query_column is None or item_column is None:
        raise ValidationError(
            "fit_ranker requires query_column= and item_column= "
            "(query/item ids are not inferred from ColumnRole)."
        )
    query_col = str(query_column)
    item_col = str(item_column)
    if query_col not in frame.columns:
        raise ValidationError(f"query_column {query_col!r} not in dataset.")
    if item_col not in frame.columns:
        raise ValidationError(f"item_column {item_col!r} not in dataset.")
    if query_col == item_col:
        raise ValidationError("query_column and item_column must differ.")

    role_map = dataset.roles
    q_role = role_map.get(query_col)
    if q_role is None:
        disclosures.append(
            f"query_column={query_col!r} has no role; prefer role='group' "
            "and Session.group_split(...) so holdout queries never appear in train."
        )
    elif q_role == ColumnRole.GROUP:
        disclosures.append(
            f"query_column={query_col!r} has role='group' (preferred for "
            "query-grouped splits)."
        )
    elif q_role not in {ColumnRole.ID, ColumnRole.IGNORE}:
        disclosures.append(
            f"query_column={query_col!r} has role={q_role.value!r}; "
            "prefer role='group' or role='id'."
        )

    i_role = role_map.get(item_col)
    if i_role is None:
        disclosures.append(
            f"item_column={item_col!r} has no role; consider role='id' or "
            "role='ignore' so classical fit() does not treat it as a feature."
        )
    elif i_role not in {ColumnRole.ID, ColumnRole.IGNORE, ColumnRole.GROUP}:
        disclosures.append(
            f"item_column={item_col!r} has role={i_role.value!r}; "
            "prefer role='id' or role='ignore'."
        )

    rel_col = relevance_column
    if rel_col is None:
        try:
            rel_col = dataset.require_target()
            disclosures.append(
                f"relevance_column defaulted to Session target {rel_col!r}."
            )
        except ValidationError as exc:
            raise ValidationError(
                "fit_ranker requires relevance_column= or a Session target role."
            ) from exc
    else:
        rel_col = str(rel_col)
    if rel_col not in frame.columns:
        raise ValidationError(f"relevance_column {rel_col!r} not in dataset.")
    if not pd.api.types.is_numeric_dtype(frame[rel_col]):
        raise ValidationError(
            f"relevance_column {rel_col!r} must be numeric "
            "(graded or binary relevance)."
        )

    if feature_columns is None:
        feat_cols = [
            c
            for c in dataset.role_columns(ColumnRole.FEATURE)
            if c not in {query_col, item_col, rel_col}
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if not feat_cols:
            raise ValidationError(
                "No numeric feature-role columns found; pass feature_columns= "
                "explicitly for tabular LTR."
            )
        disclosures.append(
            f"feature_columns defaulted to numeric feature roles: {feat_cols}."
        )
    else:
        feat_cols = [str(c) for c in feature_columns]
        missing = [c for c in feat_cols if c not in frame.columns]
        if missing:
            raise ValidationError(f"feature_columns missing from dataset: {missing}")
        for col in feat_cols:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                raise ValidationError(
                    f"LTR feature {col!r} must be numeric; got dtype "
                    f"{frame[col].dtype}."
                )
    if not feat_cols:
        raise ValidationError("At least one numeric feature column is required.")
    reserved = {query_col, item_col, rel_col}
    overlap = reserved & set(feat_cols)
    if overlap:
        raise ValidationError(
            f"feature_columns must not include query/item/relevance ids: {sorted(overlap)}"
        )

    disclosures.append(
        "Tabular LTR: each row is a (query, item, features, relevance) judgment. "
        "Distinct from RAG chunk retrieve and from recommender user-item CF."
    )
    return query_col, item_col, rel_col, tuple(feat_cols), disclosures


def feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, len(feature_columns)), dtype=float)
    return frame.loc[:, list(feature_columns)].to_numpy(dtype=float)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return X.copy(), np.zeros(0), np.ones(0)
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X - mean) / scale, mean, scale


def standardize_apply(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    if X.size == 0:
        return X.copy()
    return (X - mean) / scale


def query_group_sizes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Stable-sort rows by query group and return contiguous group sizes."""
    if len(groups) == 0:
        return X.copy(), y.copy(), groups.copy(), []
    order = np.argsort(groups, kind="mergesort")
    groups_sorted = groups[order]
    _, counts = np.unique(groups_sorted, return_counts=True)
    return (
        np.asarray(X[order], dtype=float),
        np.asarray(y[order], dtype=float),
        groups_sorted,
        [int(c) for c in counts.tolist()],
    )


def disclose_query_split(
    dataset: Dataset,
    split_plan: SplitPlan,
    query_column: str,
) -> tuple[bool, list[str], list[str]]:
    """Disclose whether queries are disjoint across partitions."""
    disclosures: list[str] = []
    warnings: list[str] = []
    frame = dataset.frame
    train_q = set(frame.iloc[list(split_plan.train_indices)][query_column].tolist())
    test_q = set(frame.iloc[list(split_plan.test_indices)][query_column].tolist())
    valid_q: set[Any] = set()
    if split_plan.validation_indices:
        valid_q = set(
            frame.iloc[list(split_plan.validation_indices)][query_column].tolist()
        )

    overlap_test = train_q & test_q
    overlap_valid = train_q & valid_q
    group_ok = (
        split_plan.kind == "group"
        and not overlap_test
        and not overlap_valid
    )

    if split_plan.kind == "group" and not overlap_test and not overlap_valid:
        disclosures.append(
            f"Split kind='group' with disjoint queries "
            f"(train={len(train_q)}, test={len(test_q)}, "
            f"validation={len(valid_q)})."
        )
    else:
        if split_plan.kind != "group":
            warnings.append(
                f"Split kind={split_plan.kind!r} is not query-grouped. "
                "Prefer Session.group_split(group_column=query_column) so test "
                "queries (and their labels) never appear in train."
            )
        if overlap_test:
            warnings.append(
                f"{len(overlap_test)} query id(s) appear in both train and test; "
                "label leakage risk for learning-to-rank. Use group_split."
            )
        if overlap_valid:
            warnings.append(
                f"{len(overlap_valid)} query id(s) appear in both train and "
                "validation; prefer group_split."
            )
        disclosures.append(
            "Query-split honesty: holdout labels are never used at fit, but "
            "overlapping query ids across partitions can still leak ranking "
            "structure. Disclose and prefer group_split."
        )
    return group_ok, disclosures, warnings


def ndcg_at_k_graded(relevances_in_rank_order: Sequence[float], k: int) -> float:
    """Graded nDCG@K (gain = 2^rel - 1)."""
    if k <= 0:
        return 0.0
    rels = [float(r) for r in relevances_in_rank_order[:k]]
    if not rels or max(rels) <= 0:
        return 0.0
    dcg = 0.0
    for rank, rel in enumerate(rels, start=1):
        if rel > 0:
            dcg += (2.0**rel - 1.0) / np.log2(rank + 1.0)
    ideal = sorted(
        (float(r) for r in relevances_in_rank_order if float(r) > 0),
        reverse=True,
    )[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal, start=1):
        idcg += (2.0**rel - 1.0) / np.log2(rank + 1.0)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def average_precision_at_k(
    relevances_in_rank_order: Sequence[float],
    k: int,
    *,
    threshold: float = 0.0,
) -> float:
    """AP@K with binary relevance = grade > threshold."""
    if k <= 0:
        return 0.0
    top = [float(r) for r in relevances_in_rank_order[:k]]
    n_relevant = sum(1 for r in relevances_in_rank_order if float(r) > threshold)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, rel in enumerate(top, start=1):
        if rel > threshold:
            hits += 1
            precision_sum += float(hits) / float(rank)
    if hits == 0:
        return 0.0
    return float(precision_sum / min(n_relevant, k))


def mrr_at_k(
    relevances_in_rank_order: Sequence[float],
    k: int,
    *,
    threshold: float = 0.0,
) -> float:
    """MRR@K: reciprocal rank of first relevant item (grade > threshold)."""
    if k <= 0:
        return 0.0
    for rank, rel in enumerate(relevances_in_rank_order[:k], start=1):
        if float(rel) > threshold:
            return float(1.0 / rank)
    return 0.0


def mean_metric_over_queries(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(list(values)))
