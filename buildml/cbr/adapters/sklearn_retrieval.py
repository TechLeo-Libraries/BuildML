"""Exact nearest-neighbour search: examine every case, return the true nearest.

The default backend, and the one everything else is measured against. It
computes the distance to every case and takes the smallest, which means the
neighbours it returns are the neighbours — no approximation, no index parameters
to tune, no recall to lose.

The cost is linear in memory size per query. That is entirely fine up to tens of
thousands of cases and becomes the bottleneck beyond it, which is the point at
which the approximate backend starts to be worth its dependency.

This is also the only backend supporting every metric, since Manhattan and the
mixed Gower-style distance have no approximate-index implementation.

See Also
--------
buildml.cbr.adapters.industry_ann : The approximate alternative.
buildml.cbr.cases.pairwise_distances : The distance arithmetic.
"""

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
    """Find the true nearest cases for every query by checking all of them.

    Computes the full query-by-case distance matrix, then takes the ``k``
    smallest per row. Nothing is skipped, so the neighbours are exactly the
    nearest.

    Parameters
    ----------
    plan:
        The fitted reasoner, supplying memory, metric, and distance floor.
    q_num:
        Query numeric features, encoded with the train-fitted transforms.
    q_cat:
        Query categorical codes, used by the mixed metric.
    k:
        Neighbours per query, clamped to the case count.

    Returns
    -------
    tuple
        ``(orders, distances)`` — per query, the neighbour indices nearest first
        and their distances.

    Raises
    ------
    ValidationError
        If query and memory widths differ, memory is empty, or the mixed metric
        was requested without categorical codes.

    Notes
    -----
    **The full distance matrix is materialised**, so memory grows as queries
    times cases times features. Predicting a large partition against a large
    case base should be done in batches.

    **Ties break by case index**, so repeated runs return identical neighbours.
    """
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
