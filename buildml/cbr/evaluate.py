"""Score a case-based reasoner on rows it does not already remember.

Evaluation matters unusually much here, and for an unusual reason. A case base
contains its training rows, so predicting them returns each row's own solution
at distance zero: a near-perfect score that measures storage rather than
reasoning. Only a partition held out of the case base says anything.

Alongside the metrics, evaluation reports the mean distance to retrieved
neighbours, and the two should be read together. A strong score with close
neighbours means the holdout is well covered by memory. The same score with
distant neighbours means the reasoner is generalising from cases that were not
very similar, and it will degrade sharply on inputs further out. That
distinction is invisible in accuracy alone.

See Also
--------
buildml.cbr.predict.predict_cbr : The predictions being scored.
buildml.cbr.results.CbrEvalResult : What comes back.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from buildml.cbr.features import classification_accuracy, regression_metrics
from buildml.cbr.predict import predict_cbr
from buildml.cbr.results import CbrEvalResult, CbrPlan
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_cbr(
    dataset: Dataset,
    plan: CbrPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    k: int | None = None,
) -> CbrEvalResult:
    """Score the reasoner on a holdout partition, with a coverage diagnostic.

    Predicts the partition, compares against the true targets, and reports the
    task's metrics together with the mean distance to retrieved neighbours.
    Memory is neither updated nor refitted.

    Parameters
    ----------
    dataset:
        The data holding both features and targets.
    plan:
        The fitted reasoner.
    split_plan:
        Partition membership. Required unless ``partition='all'``.
    partition:
        Which rows to score. Defaults to ``'validation'``; use ``'test'`` once,
        at the end.
    k:
        Override the plan's neighbour count, for a sensitivity check without
        refitting.

    Returns
    -------
    CbrEvalResult
        Accuracy for classification, or RMSE, MAE, and R² for regression, plus
        ``mean_neighbor_distance``.

    Raises
    ------
    ValidationError
        If no split plan was supplied for a named partition, the target column
        is missing, the partition has null targets, or prediction and target
        lengths disagree.

    Notes
    -----
    **Scoring ``'train'`` or ``'all'`` is not evaluation.** Those rows are in
    the case base and are their own nearest neighbours; the number will be
    excellent and will tell you nothing.

    **Null targets are refused rather than dropped.** Silently skipping rows
    would change what the denominator means without saying so.

    **Read ``mean_neighbor_distance`` next to the metrics.** It is the
    difference between a score that describes interpolation and one that
    describes extrapolation, and there is no absolute threshold: compare it
    against the distances seen within the training data.

    Examples
    --------
    Score, and check the score is well founded::

        result = evaluate_cbr(dataset, plan, split_plan, partition="validation")
        print(result.metrics, result.mean_neighbor_distance)

    See Also
    --------
    buildml.cbr.retrieve.retrieve_cases : Inspecting the neighbours directly.
    """
    pred = predict_cbr(
        dataset,
        plan,
        split_plan,
        partition=partition,
        k=k,
        return_traces=True,
    )
    y_true = _targets(dataset, split_plan, partition, plan.target_column)
    if len(y_true) != len(pred.predictions):
        raise ValidationError(
            "Prediction length does not match evaluation target length."
        )

    if plan.task == "classification":
        metrics = {
            "accuracy": classification_accuracy(
                y_true.tolist(), list(pred.predictions)
            )
        }
    else:
        metrics = regression_metrics(
            y_true.to_numpy(dtype=float),
            np.asarray(pred.predictions, dtype=float),
        )

    mean_dist = None
    if pred.traces:
        all_d = [d for t in pred.traces for d in t.distances]
        if all_d:
            mean_dist = float(np.mean(all_d))

    return CbrEvalResult(
        partition=str(partition),
        task=plan.task,
        n_rows=pred.n_rows,
        metrics=metrics,
        mean_neighbor_distance=mean_dist,
        disclosures=(
            "Holdout evaluation only: case memory was not updated from this "
            "partition.",
            *plan.disclosures[:2],
        ),
        warnings=(),
    )


def _targets(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
    target_column: str,
) -> pd.Series:
    """Pull the true targets for a partition, refusing anything unscoreable.

    Parameters
    ----------
    dataset:
        The source data.
    split_plan:
        Partition membership, or ``None`` when scoring everything.
    partition:
        Which rows to take targets from.
    target_column:
        The column the plan predicts.

    Returns
    -------
    pandas.Series
        The true values, aligned with the partition's rows.

    Raises
    ------
    ValidationError
        If a named partition was requested without a split plan, the target
        column is absent, or any target is null.

    Notes
    -----
    **Nulls are an error, not something to drop.** Quietly removing rows would
    shrink the denominator and inflate the score without leaving a trace.
    """
    from buildml.data.splits import frame_for_partition

    if partition == "all":
        frame = dataset._ensure_pandas()
    else:
        if split_plan is None:
            raise ValidationError(
                "evaluate_cbr requires a SplitPlan unless partition='all'."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
    if target_column not in frame.columns:
        raise ValidationError(
            f"Target column {target_column!r} missing from evaluation frame."
        )
    if frame[target_column].isna().any():
        raise ValidationError(
            "Evaluation partition has null targets; refuse silent drop."
        )
    return frame[target_column]
