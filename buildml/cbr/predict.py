"""Reuse / adapt neighbor solutions into predictions with case traces."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np

from buildml.cbr.cases import CaseTrace, distance_weights, pairwise_distances
from buildml.cbr.retrieve import (
    _partition_frame,
    encode_query_features,
    neighbor_pack_for_row,
)
from buildml.cbr.results import CbrPlan, CbrPredictResult
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan

PartitionOrAll = PartitionName | Literal["all"]


def predict_cbr(
    dataset: Dataset,
    plan: CbrPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    k: int | None = None,
    return_traces: bool = True,
) -> CbrPredictResult:
    """Retrieve neighbors and reuse/adapt solutions (no case-base update)."""
    frame, indices = _partition_frame(dataset, split_plan, partition)
    kk = int(plan.k if k is None else k)
    if kk < 1:
        raise ValidationError("k must be >= 1.")

    q_num, q_cat = encode_query_features(frame, plan)
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

    predictions: list[Any] = []
    traces: list[CaseTrace] = []
    for i in range(len(frame)):
        neighbors, dvals, order = neighbor_pack_for_row(plan, dists[i], kk)
        weights = distance_weights(dvals, eps=plan.distance_eps)
        pred, notes = reuse_solutions(
            neighbors=[c.solution for c in neighbors],
            weights=weights,
            neighbor_features=memory.numeric_matrix[order],
            query_features=q_num[i],
            task=plan.task,
            reuse=plan.reuse,
            adapt=plan.adapt,
        )
        predictions.append(pred)
        if return_traces:
            traces.append(
                CaseTrace(
                    query_index=indices[i],
                    neighbor_case_ids=tuple(c.case_id for c in neighbors),
                    neighbor_row_indices=tuple(c.row_index for c in neighbors),
                    distances=tuple(float(d) for d in dvals),
                    weights=tuple(float(w) for w in weights),
                    neighbor_solutions=tuple(c.solution for c in neighbors),
                    prediction=pred,
                    reuse_mode=plan.reuse,
                    adapt_mode=plan.adapt,
                    notes=tuple(notes),
                )
            )

    return CbrPredictResult(
        partition=str(partition),
        task=plan.task,
        n_rows=len(frame),
        predictions=tuple(predictions),
        traces=tuple(traces),
        disclosures=plan.disclosures,
        warnings=(),
    )


def reuse_solutions(
    *,
    neighbors: list[Any],
    weights: np.ndarray,
    neighbor_features: np.ndarray,
    query_features: np.ndarray,
    task: str,
    reuse: str,
    adapt: str,
) -> tuple[Any, list[str]]:
    """Map neighbor solutions → prediction under the configured reuse mode."""
    if not neighbors:
        raise ValidationError("Cannot reuse an empty neighbor set.")
    notes: list[str] = []
    mode = str(reuse).lower().replace("-", "_")
    adapt_key = str(adapt).lower().replace("-", "_")

    if task == "classification":
        if mode == "majority":
            counts = Counter(str(s) for s in neighbors)
            winner = counts.most_common(1)[0][0]
            pred = _match_original(winner, neighbors)
            notes.append(f"majority vote over {len(neighbors)} neighbors → {pred!r}.")
        elif mode == "distance_weighted":
            scored: dict[str, float] = {}
            for sol, w in zip(neighbors, weights, strict=True):
                key = str(sol)
                scored[key] = scored.get(key, 0.0) + float(w)
            winner = max(scored.items(), key=lambda kv: kv[1])[0]
            pred = _match_original(winner, neighbors)
            notes.append(
                f"distance-weighted vote over {len(neighbors)} neighbors → {pred!r}."
            )
        else:
            raise ValidationError(
                f"reuse={reuse!r} unsupported for classification."
            )
        return pred, notes

    # Regression
    vals = np.asarray([float(s) for s in neighbors], dtype=float)
    if mode == "local_mean":
        pred_f = float(np.mean(vals))
        notes.append(f"local_mean of {len(neighbors)} neighbor solutions.")
    elif mode == "distance_weighted":
        w = np.asarray(weights, dtype=float)
        pred_f = float(np.sum(w * vals) / np.sum(w))
        notes.append("distance-weighted average of neighbor solutions.")
    elif mode == "local_ridge":
        pred_f = _local_ridge_predict(
            neighbor_features, vals, query_features
        )
        notes.append(
            "local_ridge: Ridge(alpha=1.0) fit on k neighbor features→solution."
        )
    else:
        raise ValidationError(f"reuse={reuse!r} unsupported for regression.")

    if adapt_key == "offset":
        # Lite adapt: shift prediction by mean residual of neighbors vs local mean
        # (identity when using local_mean; for weighted/ridge, nudge toward local mean).
        local = float(np.mean(vals))
        pred_f = 0.5 * pred_f + 0.5 * local
        notes.append(
            "adapt='offset': blended prediction with neighbor mean (lite)."
        )
    elif adapt_key != "none":
        raise ValidationError(f"Unknown adapt mode {adapt!r}.")

    return pred_f, notes


def _local_ridge_predict(
    neighbor_x: np.ndarray,
    neighbor_y: np.ndarray,
    query_x: np.ndarray,
) -> float:
    from sklearn.linear_model import Ridge

    x = np.asarray(neighbor_x, dtype=float)
    y = np.asarray(neighbor_y, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return float(np.mean(y))
    if x.shape[1] == 0:
        return float(np.mean(y))
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    return float(model.predict(np.atleast_2d(query_x))[0])


def _match_original(winner_str: str, neighbors: list[Any]) -> Any:
    for sol in neighbors:
        if str(sol) == winner_str:
            return sol
    return winner_str
