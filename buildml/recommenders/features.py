"""Column resolution, interaction matrices, and ranking metrics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition

__all__ = [
    "resolve_interaction_columns",
    "train_partition_frame",
    "partition_frame",
    "build_interactions",
    "build_user_item_matrix",
    "item_feature_matrix",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision_at_k",
    "mean_average_precision",
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
            "A SplitPlan is required for partitioned recommender evaluation."
        )
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def resolve_interaction_columns(
    dataset: Dataset,
    *,
    user_column: str | None,
    item_column: str | None,
    rating_column: str | None,
    feedback: str,
) -> tuple[str, str, str | None, list[str]]:
    """Resolve user / item / rating columns with documented role conventions.

    Conventions
    -----------
    - ``user_column`` / ``item_column`` are **explicit kwargs** (no dedicated
      ColumnRole). Common practice: mark them ``role='id'`` or ``role='ignore'``
      so they are not treated as classical features.
    - ``rating_column`` defaults to the Session target role for explicit
      feedback; omit / set ``feedback='implicit'`` for presence-only signals.
    - This path is **not** RAG retrieval and **not** diagnostic EDA
      ``Recommendation`` Finding objects.
    """
    frame = dataset.frame
    disclosures: list[str] = []

    user_col = user_column
    item_col = item_column
    if user_col is None or item_col is None:
        raise ValidationError(
            "fit_recommender requires user_column= and item_column= "
            "(entity ids are not inferred from ColumnRole)."
        )
    if user_col not in frame.columns:
        raise ValidationError(f"user_column {user_col!r} not in dataset.")
    if item_col not in frame.columns:
        raise ValidationError(f"item_column {item_col!r} not in dataset.")
    if user_col == item_col:
        raise ValidationError("user_column and item_column must differ.")

    role_map = dataset.roles
    for col, label in ((user_col, "user"), (item_col, "item")):
        role = role_map.get(col)
        if role is None:
            disclosures.append(
                f"{label}_column={col!r} has no role; consider role='id' or "
                "role='ignore' so classical fit() does not treat it as a feature."
            )
        elif role not in {ColumnRole.ID, ColumnRole.IGNORE, ColumnRole.GROUP}:
            disclosures.append(
                f"{label}_column={col!r} has role={role.value!r}; "
                "prefer role='id' or role='ignore' for entity identifiers."
            )

    rating_col = rating_column
    if feedback == "implicit":
        if rating_col is not None:
            disclosures.append(
                "feedback='implicit': rating_column is ignored; "
                "presence of a row is the positive signal."
            )
        rating_col = None
        disclosures.append(
            "Implicit feedback: each train interaction is treated as a positive "
            "(value 1.0)."
        )
    else:
        if rating_col is None:
            try:
                rating_col = dataset.require_target()
                disclosures.append(
                    f"rating_column defaulted to Session target {rating_col!r}."
                )
            except ValidationError as exc:
                raise ValidationError(
                    "feedback='explicit' requires rating_column= or a Session "
                    "target role."
                ) from exc
        if rating_col not in frame.columns:
            raise ValidationError(f"rating_column {rating_col!r} not in dataset.")

    disclosures.append(
        "Recommenders are not RAG (document retrieve/generate) and not EDA "
        "Recommendation Findings (teaching advice objects)."
    )
    return user_col, item_col, rating_col, disclosures


def build_interactions(
    frame: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
    rating_column: str | None,
    feedback: str,
    min_rating: float | None = None,
) -> pd.DataFrame:
    """Normalize interaction rows; drop null entity ids."""
    needed = [user_column, item_column]
    if rating_column is not None:
        needed.append(rating_column)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValidationError(f"Interaction frame missing columns: {missing}")

    out = frame[needed].copy()
    out = out.dropna(subset=[user_column, item_column])
    if out.empty:
        raise ValidationError("No interactions remain after dropping null user/item ids.")

    if feedback == "implicit" or rating_column is None:
        out = out[[user_column, item_column]].drop_duplicates()
        out["__rating__"] = 1.0
    else:
        out = out.dropna(subset=[rating_column])
        out["__rating__"] = pd.to_numeric(out[rating_column], errors="coerce")
        out = out.dropna(subset=["__rating__"])
        if min_rating is not None:
            out = out[out["__rating__"] >= float(min_rating)]
        if out.empty:
            raise ValidationError(
                "No interactions remain after rating filters / null drops."
            )
        # Aggregate duplicate user-item pairs by mean rating
        out = (
            out.groupby([user_column, item_column], as_index=False)["__rating__"]
            .mean()
        )
    return out.reset_index(drop=True)


def build_user_item_matrix(
    interactions: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
) -> tuple[np.ndarray, tuple[Any, ...], tuple[Any, ...], dict[Any, int], dict[Any, int]]:
    """Build a dense user×item rating matrix from interaction rows."""
    users = tuple(pd.unique(interactions[user_column]))
    items = tuple(pd.unique(interactions[item_column]))
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {it: i for i, it in enumerate(items)}
    matrix = np.zeros((len(users), len(items)), dtype=float)
    u_idx = interactions[user_column].map(user_index).to_numpy()
    i_idx = interactions[item_column].map(item_index).to_numpy()
    ratings = interactions["__rating__"].to_numpy(dtype=float)
    # Last write wins for any residual duplicates (already aggregated above)
    matrix[u_idx, i_idx] = ratings
    return matrix, users, items, user_index, item_index


def item_feature_matrix(
    frame: pd.DataFrame,
    *,
    item_column: str,
    item_ids: tuple[Any, ...],
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per train item: mean of numeric feature columns (train rows)."""
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"item_feature_columns missing: {missing}")
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValidationError(
                f"Content recommender requires numeric item features; "
                f"{col!r} is not numeric."
            )
    rows: list[np.ndarray] = []
    for item in item_ids:
        sub = frame.loc[frame[item_column] == item, feature_columns]
        if sub.empty:
            rows.append(np.zeros(len(feature_columns), dtype=float))
        else:
            rows.append(sub.to_numpy(dtype=float).mean(axis=0))
    feats = np.vstack(rows) if rows else np.zeros((0, len(feature_columns)))
    mean = np.mean(feats, axis=0) if len(feats) else np.zeros(len(feature_columns))
    scale = np.std(feats, axis=0) if len(feats) else np.ones(len(feature_columns))
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (feats - mean) / scale
    return standardized, mean, scale


def precision_at_k(recommended: list[Any], relevant: set[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return float(hits) / float(k)


def recall_at_k(recommended: list[Any], relevant: set[Any], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for item in top if item in relevant)
    return float(hits) / float(len(relevant))


def ndcg_at_k(recommended: list[Any], relevant: set[Any], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    dcg = 0.0
    for rank, item in enumerate(top, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1.0)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def average_precision_at_k(
    recommended: list[Any], relevant: set[Any], k: int
) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top, start=1):
        if item in relevant:
            hits += 1
            precision_sum += float(hits) / float(rank)
    if hits == 0:
        return 0.0
    return float(precision_sum / min(len(relevant), k))


def mean_average_precision(
    per_user_recommended: list[list[Any]],
    per_user_relevant: list[set[Any]],
    k: int,
) -> float:
    if not per_user_recommended:
        return 0.0
    scores = [
        average_precision_at_k(rec, rel, k)
        for rec, rel in zip(per_user_recommended, per_user_relevant)
        if rel
    ]
    if not scores:
        return 0.0
    return float(np.mean(scores))
