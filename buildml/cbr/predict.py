"""Turn the retrieved neighbours into an answer, and record how.

Retrieval finds the ``k`` most similar past cases; something still has to decide
what they collectively imply. That is *reuse*, the second step of the
case-based reasoning cycle, and the choice of mode is a real modelling decision
rather than a formatting one.

For classification, ``'majority'`` gives every neighbour an equal vote, which is
robust when distances are noisy and unhelpful when one neighbour is far closer
than the rest. ``'distance_weighted'`` weights by proximity, which respects that
a near-identical case is better evidence than a marginal one, at the cost of
letting a single close neighbour dominate.

For regression, ``'local_mean'`` averages, ``'distance_weighted'`` averages with
proximity weights, and ``'local_ridge'`` fits a small linear model to the
neighbours and evaluates it at the query. The last of those is the only mode
that can extrapolate: the others are bounded by the neighbour values they
combine, so a genuine trend within the neighbourhood is invisible to them.

Every prediction also produces a :class:`~buildml.cbr.cases.CaseTrace` naming
the cases behind it, which is the property that makes this method worth
choosing when a decision has to be justified.

See Also
--------
buildml.cbr.retrieve.retrieve_cases : The retrieval step alone.
buildml.cbr.types.CbrConfig : Where the reuse mode is set.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np

from buildml.cbr.cases import CaseTrace, distance_weights
from buildml.cbr.results import CbrPlan, CbrPredictResult
from buildml.cbr.retrieval_engine import retrieve_neighbor_batches
from buildml.cbr.retrieve import (
    _partition_frame,
    _plan_with_backend,
    encode_query_features,
    neighbor_pack_for_row,
)
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
    backend: str | None = None,
) -> CbrPredictResult:
    """Predict a partition by finding similar cases and reusing their outcomes.

    Retrieves neighbours for every row, combines their solutions under the
    plan's reuse mode, and returns the predictions with a trace for each one.
    Memory is not modified: retention is a separate, deliberate step.

    Parameters
    ----------
    dataset:
        The data to predict. Must supply the plan's feature columns.
    plan:
        The fitted reasoner.
    split_plan:
        Partition membership.
    partition:
        Which rows to predict: ``'train'``, ``'validation'``, ``'test'``, or
        ``'all'``.
    k:
        Override the plan's neighbour count for this call. Useful for seeing how
        sensitive predictions are to ``k`` without refitting.
    return_traces:
        Attach the per-row explanations. Leave on unless memory is tight; the
        traces are the reason to use this method.
    backend:
        Override the retrieval backend for this call.

    Returns
    -------
    CbrPredictResult
        Predictions in row order, decoded back to original labels for
        classification, with traces and the plan's disclosures.

    Raises
    ------
    ValidationError
        If ``k`` is below one, the partition is unknown, required columns are
        missing, or the reuse mode is not valid for the task.

    Notes
    -----
    **Predicting the train partition is not a measurement.** Every train row is
    its own nearest neighbour at distance zero, and under distance weighting
    that self-match dominates the vote. The number will look excellent and mean
    nothing.

    **A prediction is produced for every row regardless of how far away its
    neighbours are.** There is no abstention. Read the trace distances before
    acting on predictions for unusual inputs.

    **Cost is one full distance computation per row against the whole memory**
    under exact search, so time grows with rows times cases.

    Examples
    --------
    Predict, then read the evidence behind the first row::

        result = predict_cbr(dataset, plan, split_plan, partition="test")
        trace = result.traces[0]
        print(trace.prediction, trace.distances, trace.neighbor_solutions)

    See Also
    --------
    buildml.cbr.evaluate.evaluate_cbr : Scoring these predictions.
    reuse_solutions : The combination step on its own.
    """
    frame, indices = _partition_frame(dataset, split_plan, partition)
    kk = int(plan.k if k is None else k)
    if kk < 1:
        raise ValidationError("k must be >= 1.")

    q_num, q_cat = encode_query_features(frame, plan)
    memory = plan.case_base
    active_plan = _plan_with_backend(plan, backend)
    orders, drows = retrieve_neighbor_batches(
        active_plan,
        q_num,
        q_cat,
        k=kk,
        query_frame=frame if active_plan.backend == "embedding" else None,
    )

    predictions: list[Any] = []
    traces: list[CaseTrace] = []
    for i in range(len(frame)):
        order = orders[i]
        neighbors, dvals, order = neighbor_pack_for_row(active_plan, order, drows[i])
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
    """Combine the neighbours' outcomes into a single prediction.

    The reuse step in isolation, exposed so it can be reasoned about and tested
    independently of retrieval. Returns the prediction together with notes
    describing how it was reached, which become the trace's explanation.

    Parameters
    ----------
    neighbors:
        The neighbours' solutions, nearest first. Must be non-empty.
    weights:
        Per-neighbour influence, aligned with ``neighbors``.
    neighbor_features:
        Neighbour feature rows, used only by ``'local_ridge'``.
    query_features:
        The query's features, used only by ``'local_ridge'``.
    task:
        ``'classification'`` or ``'regression'``.
    reuse:
        ``'majority'`` or ``'distance_weighted'`` for classification;
        ``'local_mean'``, ``'distance_weighted'``, or ``'local_ridge'`` for
        regression.
    adapt:
        ``'none'``, or ``'offset'`` to blend the prediction halfway toward the
        unweighted neighbour mean. Regression only.

    Returns
    -------
    tuple
        ``(prediction, notes)``: the answer and a plain-language account of how
        it was produced.

    Raises
    ------
    ValidationError
        If the neighbour set is empty, or the reuse or adapt mode is not valid
        for the task.

    Notes
    -----
    **Only ``'local_ridge'`` can predict outside the neighbours' range.** The
    other regression modes return a weighted combination, so a value higher than
    every neighbour is unreachable: which is safe, and blind to a trend running
    through the neighbourhood.

    **``'local_ridge'`` falls back to the mean when it cannot fit**, which
    happens with fewer than two neighbours or no features. The prediction is
    then a ``'local_mean'`` prediction under a different name.

    **Classification votes compare solutions as strings**, so labels differing
    only by type are treated as the same class. The original value is returned,
    not the string.

    **``'offset'`` is a fixed half-and-half blend**, not a fitted correction. It
    is a hedge against an over-confident weighted or ridge prediction, and it is
    the identity under ``'local_mean'``.
    """
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
    """Fit a small ridge regression to the neighbours and evaluate it at the query.

    Locally weighted regression in miniature: instead of averaging the
    neighbours' values, fit a line through them and read it off at the query
    point. This captures a trend the neighbours share, which averaging cannot.

    Parameters
    ----------
    neighbor_x:
        Neighbour features, one row each.
    neighbor_y:
        Their solution values.
    query_x:
        The query's features.

    Returns
    -------
    float
        The fitted prediction, or the neighbour mean when a fit is impossible.

    Notes
    -----
    **Ridge rather than ordinary least squares because ``k`` is small.** With
    five neighbours and five features the system is exactly determined and an
    unregularised fit would interpolate them perfectly and extrapolate wildly.
    The ``alpha=1.0`` penalty keeps the local slope bounded.

    **Falls back to the mean rather than raising** when there are fewer than two
    neighbours or no features, since there is nothing to fit and a prediction is
    still owed.
    """
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
    """Recover the original label object behind a stringified vote winner.

    Votes are tallied on string forms so that labels of mixed types compare
    sensibly, but the caller should get back the value as it appears in their
    data: an integer class stays an integer, a categorical stays itself.

    Parameters
    ----------
    winner_str:
        The winning label's string form.
    neighbors:
        The neighbour solutions to search.

    Returns
    -------
    object
        The first neighbour solution matching the string, or the string itself
        if none does.
    """
    for sol in neighbors:
        if str(sol) == winner_str:
            return sol
    return winner_str
