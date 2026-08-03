"""Triple column resolution, train-only materialization, adjacency helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition

__all__ = [
    "resolve_triple_columns",
    "train_partition_frame",
    "partition_frame",
    "build_triples",
    "build_vocabularies",
    "encode_triples",
    "build_adjacency",
    "triple_set",
    "mrr_from_ranks",
    "hits_at_k",
]


def train_partition_frame(dataset: Dataset, split_plan: SplitPlan) -> pd.DataFrame:
    """Return the Session train partition as a pandas DataFrame.

    Convenience wrapper used by fit paths before triple materialization.

    Parameters
    ----------
    dataset:
        Session dataset.
    split_plan:
        Split plan with train indices.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by ``split_plan.train_indices``.
    """
    return frame_for_partition(dataset, split_plan, "train")


def partition_frame(
    dataset: Dataset, split_plan: SplitPlan | None, partition: str
) -> pd.DataFrame:
    """Return a holdout or full-dataset frame for KG scoring and evaluation.

    Wraps split-plan partition selection so score and evaluate paths share the
    same frame contract.

    Parameters
    ----------
    dataset:
        Session dataset.
    split_plan:
        Split plan required unless ``partition='all'``.
    partition:
        ``train``, ``validation``, ``test``, or ``all``.

    Returns
    -------
    pandas.DataFrame
        Rows for the requested partition.

    Raises
    ------
    ValidationError
        When a named partition is requested without a split plan.
    """
    if partition == "all":
        return dataset.frame.copy()
    if split_plan is None:
        raise ValidationError(
            "A SplitPlan is required for partitioned knowledge-graph evaluation."
        )
    return frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]


def resolve_triple_columns(
    dataset: Dataset,
    *,
    head_column: str | None,
    relation_column: str | None,
    tail_column: str | None,
) -> tuple[str, str, str, list[str]]:
    """Resolve head, relation, and tail columns for KG fit.

    All three columns must be supplied explicitly; roles are checked for
    teaching disclosures but are not inferred automatically.

    Parameters
    ----------
    dataset:
        Session dataset containing triple columns.
    head_column, relation_column, tail_column:
        Column names for the (head, relation, tail) triple store.

    Returns
    -------
    tuple[str, str, str, list[str]]
        Resolved column names and honesty disclosures.

    Raises
    ------
    ValidationError
        When any column is missing or the three names are not distinct.

    Notes
    -----
    Prefer ``role='id'`` or ``role='ignore'`` on triple columns so classical
    ``fit()`` does not treat them as numeric features. Distinct from Graph ML
    adjacency features and from RAG chunk embeddings.
    """
    frame = dataset.frame
    disclosures: list[str] = []

    if head_column is None or relation_column is None or tail_column is None:
        raise ValidationError(
            "fit_kg requires head_column=, relation_column=, and tail_column= "
            "(triple columns are not inferred from ColumnRole)."
        )
    for name, col in (
        ("head_column", head_column),
        ("relation_column", relation_column),
        ("tail_column", tail_column),
    ):
        if col not in frame.columns:
            raise ValidationError(f"{name} {col!r} not in dataset.")
    if len({head_column, relation_column, tail_column}) < 3:
        raise ValidationError(
            "head_column, relation_column, and tail_column must be three "
            "distinct column names."
        )

    role_map = dataset.roles
    for col, label in (
        (head_column, "head"),
        (relation_column, "relation"),
        (tail_column, "tail"),
    ):
        role = role_map.get(col)
        if role is None:
            disclosures.append(
                f"{label}_column={col!r} has no role; consider role='id' or "
                "role='ignore' so classical fit() does not treat it as a feature."
            )
        elif role not in {ColumnRole.ID, ColumnRole.IGNORE, ColumnRole.GROUP}:
            disclosures.append(
                f"{label}_column={col!r} has role={role.value!r}; "
                "prefer role='id' or role='ignore' for KG identifiers."
            )

    disclosures.append(
        "KG path: Session rows are (head, relation, tail) triples. "
        "Distinct from Graph ML node classification and from RAG."
    )
    return head_column, relation_column, tail_column, disclosures


def build_triples(
    frame: pd.DataFrame,
    *,
    head_column: str,
    relation_column: str,
    tail_column: str,
) -> pd.DataFrame:
    """Materialize unique non-null triples from a partition frame.

    Drops null cells and duplicate rows before vocabulary construction on
    Session train only.

    Parameters
    ----------
    frame:
        Partition frame containing triple columns.
    head_column, relation_column, tail_column:
        Column names defining each triple.

    Returns
    -------
    pandas.DataFrame
        Unique triple rows with duplicates removed.

    Raises
    ------
    ValidationError
        When the frame is empty, columns are missing, or all rows are null.
    """
    if frame.empty:
        raise ValidationError("No rows available to build KG triples.")
    for col in (head_column, relation_column, tail_column):
        if col not in frame.columns:
            raise ValidationError(f"Triple column {col!r} missing from frame.")
    triples = frame[[head_column, relation_column, tail_column]].copy()
    triples = triples.dropna()
    if triples.empty:
        raise ValidationError("All triples were null after dropna.")
    before = len(triples)
    triples = triples.drop_duplicates()
    triples.attrs["n_duplicate_dropped"] = int(before - len(triples))
    return triples.reset_index(drop=True)


def build_vocabularies(
    triples: pd.DataFrame,
    *,
    head_column: str,
    relation_column: str,
    tail_column: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[Any, int], dict[Any, int]]:
    """Build entity and relation catalogs from train triples only.

    Entity vocabulary unions heads and tails; relations are taken from the
    relation column only.

    Parameters
    ----------
    triples:
        Materialized triple frame from :func:`build_triples`.
    head_column, relation_column, tail_column:
        Column names defining each triple.

    Returns
    -------
    tuple[tuple, tuple, dict, dict]
        ``(entity_ids, relation_ids, entity_index, relation_index)``.
    """
    entities = pd.unique(
        pd.concat(
            [triples[head_column], triples[tail_column]],
            ignore_index=True,
        )
    )
    relations = pd.unique(triples[relation_column])
    entity_ids = tuple(entities.tolist())
    relation_ids = tuple(relations.tolist())
    entity_index = {e: i for i, e in enumerate(entity_ids)}
    relation_index = {r: i for i, r in enumerate(relation_ids)}
    return entity_ids, relation_ids, entity_index, relation_index


def encode_triples(
    triples: pd.DataFrame,
    *,
    head_column: str,
    relation_column: str,
    tail_column: str,
    entity_index: dict[Any, int],
    relation_index: dict[Any, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map raw triple identifiers to dense integer index arrays.

    Rows with out-of-vocabulary heads, relations, or tails are skipped silently
    so holdout evaluation can report skipped counts separately.

    Parameters
    ----------
    triples:
        Triple frame to encode.
    head_column, relation_column, tail_column:
        Column names defining each triple.
    entity_index, relation_index:
        Vocabulary maps from :func:`build_vocabularies`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Parallel ``heads``, ``relations``, and ``tails`` integer arrays.

    Raises
    ------
    ValidationError
        When no triples remain encodable against the vocabulary.
    """
    heads: list[int] = []
    rels: list[int] = []
    tails: list[int] = []
    for h, r, t in zip(
        triples[head_column].tolist(),
        triples[relation_column].tolist(),
        triples[tail_column].tolist(),
        strict=True,
    ):
        if h not in entity_index or t not in entity_index or r not in relation_index:
            continue
        heads.append(entity_index[h])
        rels.append(relation_index[r])
        tails.append(entity_index[t])
    if not heads:
        raise ValidationError(
            "No triples remain after vocabulary encoding "
            "(all heads/relations/tails were out-of-vocabulary)."
        )
    return (
        np.asarray(heads, dtype=np.int64),
        np.asarray(rels, dtype=np.int64),
        np.asarray(tails, dtype=np.int64),
    )


def build_adjacency(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build out/in adjacency lists for symbolic KG query.

    Each entity maps to ``(relation_id, neighbor_id)`` pairs derived from train
    triples only.

    Parameters
    ----------
    heads, relations, tails:
        Encoded train triple arrays.

    Returns
    -------
    tuple[dict, dict]
        ``(out_edges, in_edges)`` adjacency maps for :func:`query_kg`.
    """
    out_edges: dict[int, list[tuple[int, int]]] = {}
    in_edges: dict[int, list[tuple[int, int]]] = {}
    for h, r, t in zip(heads.tolist(), relations.tolist(), tails.tolist(), strict=True):
        out_edges.setdefault(h, []).append((r, t))
        in_edges.setdefault(t, []).append((r, h))
    return out_edges, in_edges


def triple_set(
    heads: np.ndarray, relations: np.ndarray, tails: np.ndarray
) -> frozenset[tuple[int, int, int]]:
    """Return the set of encoded train triples for filtered ranking.

    Built from train-only triple arrays and stored on the plan for evaluate and
    predict filtered-ranking paths.

    Parameters
    ----------
    heads, relations, tails:
        Encoded triple arrays.

    Returns
    -------
    frozenset[tuple[int, int, int]]
        Known true triples used to filter candidate ranks during evaluation.
    """
    return frozenset(
        zip(heads.tolist(), relations.tolist(), tails.tolist(), strict=True)
    )


def mrr_from_ranks(ranks: list[int] | np.ndarray) -> float:
    """Compute mean reciprocal rank from 1-indexed rank lists.

    Used by :func:`buildml.kg.evaluate.evaluate_kg` after filtered tail/head
    ranking on holdout triples.

    Parameters
    ----------
    ranks:
        1-indexed ranks of the true entity or relation fill-in.

    Returns
    -------
    float
        Mean of ``1 / rank`` across queries, or ``0.0`` when empty.
    """
    arr = np.asarray(ranks, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(1.0 / arr))


def hits_at_k(ranks: list[int] | np.ndarray, k: int) -> float:
    """Compute Hits@K from 1-indexed rank lists.

    Returns the fraction of queries whose true fill-in ranks at or below ``k``.

    Parameters
    ----------
    ranks:
        1-indexed ranks of the true fill-in.
    k:
        Cutoff for the hit rate.

    Returns
    -------
    float
        Fraction of ranks less than or equal to ``k``, or ``0.0`` when empty.
    """
    arr = np.asarray(ranks, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr <= float(k)))
