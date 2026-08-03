"""Route a neighbour query to whichever backend the plan was fitted with.

The four backends search different spaces. Exact search works over the raw
numeric features. The embedding backend searches sentence-transformer vectors,
so the query has to be embedded first. The torch backend searches a learned
representation, so the query has to pass through the encoder. The industry
backend searches an approximate index rather than scanning.

Two invariants hold across all of them. A query is always transformed the same
way the cases were, since neighbours in mismatched spaces are meaningless. And
where an approximate index is unavailable, the same search matrix is scanned
exactly: the result is identical, just slower, so an unavailable index never
changes the answer.

See Also
--------
buildml.cbr.retrieval_build.build_search_artifacts : Building what is searched.
buildml.cbr.cases.pairwise_distances : The exact-search arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.cbr.adapters.industry_ann import query_ann_index
from buildml.cbr.adapters.sklearn_retrieval import batch_neighbor_orders
from buildml.cbr.adapters.text_embed import embed_text_queries
from buildml.cbr.cases import pairwise_distances, top_k_indices
from buildml.cbr.results import CbrPlan
from buildml.core.errors import ValidationError


def retrieve_neighbor_batches(
    plan: CbrPlan,
    q_num: np.ndarray,
    q_cat: np.ndarray,
    *,
    k: int,
    query_frame: Any | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Find each query's nearest cases, using the plan's backend.

    Transforms the queries into the space the case base was indexed in, then
    searches: through the approximate index when one exists, by exact scan
    otherwise.

    Parameters
    ----------
    plan:
        The fitted reasoner, supplying the backend, metric, and case memory.
    q_num:
        Query numeric features, already encoded with the train-fitted
        transforms.
    q_cat:
        Query categorical codes, used by the mixed metric.
    k:
        Neighbours per query.
    query_frame:
        The raw query frame. Required by the embedding backend, which needs the
        text columns; ignored by the others.

    Returns
    -------
    tuple
        ``(orders, distances)``: per query, the neighbour indices nearest first
        and their distances.

    Raises
    ------
    ValidationError
        If the search matrix is missing, the embedding backend was given no
        query frame, or the torch encoder is absent from memory.

    Notes
    -----
    **Falling back to an exact scan changes speed, not results.** The same
    matrix is searched with the same metric; only the shortcut is missing.

    **Approximate index results are approximate.** The industry backend may miss
    a true nearest neighbour, which is the trade being made for speed over a
    large memory.

    **Distances from the embedding and torch backends live in the learned
    space.** They are not comparable with distances over raw features, and their
    absolute values mean nothing outside their own ranking.
    """
    backend = str(getattr(plan, "backend", "sklearn") or "sklearn")
    if backend == "sklearn":
        return batch_neighbor_orders(plan, q_num, q_cat, k)

    memory = plan.case_base
    search = memory.search_matrix_
    if search is None or search.shape[0] == 0:
        raise ValidationError("Case memory search_matrix_ is missing for backend retrieval.")

    if backend == "embedding":
        if query_frame is None:
            raise ValidationError(
                "embedding backend retrieval requires the query DataFrame."
            )
        q_search = embed_text_queries(
            query_frame,
            plan.text_columns,
            model_name=plan.text_model_name,
            numeric_matrix=q_num if q_num.shape[1] else None,
        )
    elif backend == "torch":
        from buildml.cbr.adapters.torch_metric import encode_with_torch

        if memory.torch_encoder_ is None:
            raise ValidationError("Torch encoder missing from case memory.")
        q_search = encode_with_torch(
            memory.torch_encoder_,
            q_num,
            device=str(plan.config.get("device", "cpu")),
        )
    else:
        q_search = q_num

    if memory.ann_index_ is not None and memory.ann_library_:
        dists, labels = query_ann_index(
            memory.ann_index_,
            memory.ann_library_,
            q_search,
            k=k,
            metric=plan.metric,
        )
        orders = [labels[i] for i in range(labels.shape[0])]
        drows = [dists[i] for i in range(dists.shape[0])]
        return orders, drows

    # Exact fallback on search matrix (embedding/torch without ANN libs).
    dists = pairwise_distances(
        q_search,
        search,
        metric=plan.metric,
        eps=plan.distance_eps,
    )
    orders: list[np.ndarray] = []
    drows: list[np.ndarray] = []
    for i in range(dists.shape[0]):
        order = top_k_indices(dists[i], k)
        orders.append(order)
        drows.append(dists[i][order])
    return orders, drows
