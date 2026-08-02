"""Exact kNN retrieval via numpy distance helpers (sklearn backend fallback)."""

from __future__ import annotations

import numpy as np

from buildml.cbr.cases import pairwise_distances, top_k_indices
from buildml.cbr.results import CbrPlan


def batch_neighbor_orders(
    plan: CbrPlan,
    q_num: np.ndarray,
    q_cat: np.ndarray,
    k: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-query neighbor index orders and distance vectors."""
    memory = plan.case_base
    dists = pairwise_distances(
        q_num,
        memory.numeric_matrix,
        metric=plan.metric,
        query_cat=q_cat,
        memory_cat=memory.categorical_matrix,
        numeric_ranges=memory.numeric_ranges_,
        eps=plan.distance_eps,
    )
    orders: list[np.ndarray] = []
    drows: list[np.ndarray] = []
    for i in range(dists.shape[0]):
        order = top_k_indices(dists[i], k)
        orders.append(order)
        drows.append(dists[i][order])
    return orders, drows
