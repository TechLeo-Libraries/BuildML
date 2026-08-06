"""Column resolution, feature matrices, and graded ranking metrics for LTR."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    """Return the Session train partition as a pandas DataFrame.

    Thin wrapper around :func:`buildml.data.splits.frame_for_partition` for
    ranker fit paths that must never read holdout rows.

    Parameters
    ----------
    dataset:
        Session dataset containing ranking judgment rows.
    split_plan:
        Split plan defining the train index set.

    Returns
    -------
    pandas.DataFrame
        Copy of rows belonging to the train partition.
    """
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset, split_plan: SplitPlan | None, partition: str
) -> pd.DataFrame:
    """Return a named data partition for ranker evaluate/rank paths.

    Supports ``all`` for the full frame or train/test/validation slices from
    a Session split plan.

    Parameters
    ----------
    dataset:
        Session dataset containing ranking judgment rows.
    split_plan:
        Split plan defining partition indices; required unless ``partition``
        is ``all``.
    partition:
        Partition name such as ``train``, ``test``, ``validation``, or ``all``.

    Returns
    -------
    pandas.DataFrame
        Rows belonging to the requested partition.

    Raises
    ------
    ValidationError
        When a named partition is requested but ``split_plan`` is ``None``.
    """
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
    """Resolve query, item, relevance, and feature columns for tabular LTR.

    Validates explicit query and item ids, defaults relevance to the Session
    target role, and selects numeric feature-role columns when not provided.

    Parameters
    ----------
    dataset:
        Session dataset with judgment rows and column roles.
    query_column:
        Query id column; required and not inferred from roles.
    item_column:
        Item id column; required and not inferred from roles.
    relevance_column:
        Graded or binary relevance labels; defaults to Session target.
    feature_columns:
        Numeric feature columns; defaults to numeric ``feature`` roles.

    Returns
    -------
    tuple[str, str, str, tuple[str, ...], list[str]]
        Query column, item column, relevance column, feature tuple, and honesty
        disclosure strings.

    Raises
    ------
    ValidationError
        When required columns are missing, non-numeric, or overlap reserved ids.

    Notes
    -----
    Prefer ``role='group'`` on the query id so ``group_split`` can isolate
    queries across partitions. This path is tabular learning-to-rank, not RAG
    retrieve/generate and not Session recommenders (user–item CF).
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
    """Materialize a numeric feature matrix from a judgment frame.

    Returns a zero-row matrix with the correct column count when the frame is
    empty so downstream rankers can handle edge cases uniformly.

    Parameters
    ----------
    frame:
        Judgment rows containing the requested feature columns.
    feature_columns:
        Numeric columns to extract in declaration order.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_features)`` float array of feature values.
    """
    if frame.empty:
        return np.zeros((0, len(feature_columns)), dtype=float)
    return frame.loc[:, list(feature_columns)].to_numpy(dtype=float)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-feature z-score standardization on train rows only.

    Near-zero standard deviations are clamped to 1.0 so constant features do
    not explode at apply time.

    Parameters
    ----------
    X:
        Train feature matrix of shape ``(n_rows, n_features)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Standardized matrix, per-feature means, and per-feature scales.
    """
    if X.size == 0:
        return X.copy(), np.zeros(0), np.ones(0)
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X - mean) / scale, mean, scale


def standardize_apply(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply train-fitted z-score standardization to new rows.

    Uses the mean and scale vectors stored on a frozen
    :class:`~buildml.ranking.results.RankerPlan`.

    Parameters
    ----------
    X:
        Feature matrix to transform.
    mean:
        Per-feature means from :func:`standardize_fit`.
    scale:
        Per-feature scales from :func:`standardize_fit`.

    Returns
    -------
    numpy.ndarray
        Standardized feature matrix with the same shape as ``X``.
    """
    if X.size == 0:
        return X.copy()
    return (X - mean) / scale


def query_group_sizes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Stable-sort rows by query group and return contiguous group sizes.

    Industry GBDT rankers require rows grouped by query with monotonic group
    size arrays; this helper prepares that layout from arbitrary row order.

    Parameters
    ----------
    X:
        Feature matrix aligned with ``y`` and ``groups``.
    y:
        Relevance labels aligned with ``groups``.
    groups:
        Query id array with one entry per row.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list[int]]
        Sorted features, sorted labels, sorted groups, and contiguous group
        sizes for LightGBM/XGBoost training APIs.
    """
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
    """Disclose whether queries are disjoint across train and holdout partitions.

    Compares query id sets across train, validation, and test indices and
    records warnings when splits are not query-grouped or overlap exists.

    Parameters
    ----------
    dataset:
        Session dataset containing the query id column.
    split_plan:
        Split plan whose indices define partition membership.
    query_column:
        Query id column used for group-split honesty checks.

    Returns
    -------
    tuple[bool, list[str], list[str]]
        ``True`` when ``split_plan.kind=='group'`` with disjoint queries,
        disclosure strings, and warning strings.
    """
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
    """Compute graded nDCG@K with gain ``2^rel - 1``.

    Expects relevances already sorted by descending ranker score within a
    single query. Returns ``0.0`` when the ideal DCG is zero.

    Parameters
    ----------
    relevances_in_rank_order:
        Graded relevance labels in rank order (best item first).
    k:
        Cutoff for discounted cumulative gain.

    Returns
    -------
    float
        Normalized discounted cumulative gain in ``[0, 1]``.
    """
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
    """Compute average precision at K with binary relevance.

    Treats an item as relevant when its graded label exceeds ``threshold``.
    Uses the standard AP formula averaged over relevant items in the top-K list.

    Parameters
    ----------
    relevances_in_rank_order:
        Graded relevance labels in rank order (best item first).
    k:
        Cutoff for precision accumulation.
    threshold:
        Grades strictly above this value count as relevant.

    Returns
    -------
    float
        Average precision at K for the ranked list.
    """
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
    """Compute mean reciprocal rank at K for a single query.

    Returns the reciprocal rank of the first item whose graded label exceeds
    ``threshold``, or ``0.0`` when no relevant item appears in the top-K list.

    Parameters
    ----------
    relevances_in_rank_order:
        Graded relevance labels in rank order (best item first).
    k:
        Cutoff for the reciprocal-rank search.
    threshold:
        Grades strictly above this value count as relevant.

    Returns
    -------
    float
        Reciprocal rank of the first relevant item, or ``0.0``.
    """
    if k <= 0:
        return 0.0
    for rank, rel in enumerate(relevances_in_rank_order[:k], start=1):
        if float(rel) > threshold:
            return float(1.0 / rank)
    return 0.0


def mean_metric_over_queries(values: Sequence[float]) -> float:
    """Macro-average a per-query metric list for holdout evaluation.

    Returns ``0.0`` for an empty input so callers can safely aggregate skipped
    query lists without extra guards.

    Parameters
    ----------
    values:
        Per-query metric values to average.

    Returns
    -------
    float
        Arithmetic mean of ``values``, or ``0.0`` when empty.
    """
    if not values:
        return 0.0
    return float(np.mean(list(values)))
