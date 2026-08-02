"""Unified neighbor retrieval dispatch for CBR backends."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.cbr.adapters.industry_ann import query_ann_index
from buildml.cbr.adapters.sklearn_retrieval import batch_neighbor_orders
from buildml.cbr.adapters.text_embed import embed_text_queries
from buildml.cbr.adapters.torch_metric import encode_with_torch
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
    """Retrieve k-neighbor orders and distances for each query row.

    Dispatches by ``plan.backend`` while preserving sklearn exact fallback.
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
