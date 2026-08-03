"""Holdout evaluation for symbolic / neuro-symbolic plans."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan
from buildml.symbolic.features import (
    classification_accuracy,
    regression_metrics,
)
from buildml.symbolic.predict import predict_neuro_symbolic, predict_symbolic
from buildml.symbolic.results import (
    NeuroSymbolicPlan,
    SymbolicEvalResult,
    SymbolicPlan,
)

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_symbolic(
    dataset: Dataset,
    plan: SymbolicPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> SymbolicEvalResult:
    """Score the symbolic rule base on a holdout partition without refit.

    Runs predict with traces on the requested partition, then compares
    predictions to held-out labels for accuracy or regression metrics.

    Parameters
    ----------
    dataset:
        Session dataset with labeled holdout rows.
    plan:
        Train-fitted :class:`~buildml.symbolic.results.SymbolicPlan`.
    split_plan:
        Split plan defining the evaluation partition.
    partition:
        ``validation``, ``test``, or ``all``.

    Returns
    -------
    SymbolicEvalResult
        Accuracy or regression metrics plus rule coverage statistics.
    """
    pred = predict_symbolic(
        dataset, plan, split_plan, partition=partition, return_traces=True
    )
    y_true = _targets(dataset, split_plan, partition, plan.target_column)
    metrics, coverage, mean_fired = _score(
        y_true, pred.predictions, pred.traces, task=plan.task
    )
    return SymbolicEvalResult(
        partition=str(partition),
        path="symbolic",
        task=plan.task,
        n_rows=pred.n_rows,
        metrics=metrics,
        rule_coverage=coverage,
        mean_rules_fired=mean_fired,
        disclosures=(
            "Holdout evaluation only: rules were not re-induced on this partition.",
            *plan.disclosures[:2],
        ),
        warnings=(),
    )


def evaluate_neuro_symbolic(
    dataset: Dataset,
    plan: NeuroSymbolicPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> SymbolicEvalResult:
    """Score the neuro-symbolic hybrid on a holdout partition without refit.

    Combines neural and rule predictions from the fitted plan, then reports
    metrics plus rule coverage and repair rates on the holdout partition.

    Parameters
    ----------
    dataset:
        Session dataset with labeled holdout rows.
    plan:
        Train-fitted :class:`~buildml.symbolic.results.NeuroSymbolicPlan`.
    split_plan:
        Split plan defining the evaluation partition.
    partition:
        ``validation``, ``test``, or ``all``.

    Returns
    -------
    SymbolicEvalResult
        Metrics, rule coverage, repair rate, and optional neural/final agreement.
    """
    pred = predict_neuro_symbolic(
        dataset, plan, split_plan, partition=partition, return_traces=True
    )
    y_true = _targets(dataset, split_plan, partition, plan.target_column)
    metrics, coverage, mean_fired = _score(
        y_true, pred.predictions, pred.traces, task=plan.task
    )
    repair_rate = (
        float(pred.n_repaired) / float(pred.n_rows) if pred.n_rows else None
    )
    # Fidelity: agreement between neural and final when neural present.
    if pred.neural_predictions is not None:
        agree = sum(
            str(a) == str(b)
            for a, b in zip(
                pred.neural_predictions, pred.predictions, strict=True
            )
        )
        metrics = {
            **metrics,
            "neural_final_agreement": float(agree) / float(max(pred.n_rows, 1)),
        }
    return SymbolicEvalResult(
        partition=str(partition),
        path="neuro_symbolic",
        task=plan.task,
        n_rows=pred.n_rows,
        metrics=metrics,
        rule_coverage=coverage,
        mean_rules_fired=mean_fired,
        repair_rate=repair_rate,
        disclosures=(
            "Holdout evaluation only: base estimator and rules were not "
            "refit on this partition.",
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
    from buildml.data.splits import frame_for_partition

    if partition == "all":
        frame = dataset._ensure_pandas()
    else:
        if split_plan is None:
            raise ValidationError(
                "evaluate_symbolic requires a SplitPlan unless partition='all'."
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


def _score(
    y_true: pd.Series,
    predictions: tuple[Any, ...],
    traces: tuple[Any, ...],
    *,
    task: str,
) -> tuple[dict[str, float], float | None, float | None]:
    if len(y_true) != len(predictions):
        raise ValidationError(
            "Prediction length does not match evaluation target length."
        )
    if task == "classification":
        metrics = {
            "accuracy": classification_accuracy(
                y_true.tolist(), list(predictions)
            )
        }
    else:
        y_hat = np.asarray(
            [
                float(p) if p is not None else float("nan")
                for p in predictions
            ],
            dtype=float,
        )
        if np.isnan(y_hat).any():
            raise ValidationError(
                "Symbolic regression predictions contain nulls "
                "(missing default_consequent or unmatched rules)."
            )
        metrics = regression_metrics(y_true.to_numpy(dtype=float), y_hat)

    if traces:
        covered = sum(1 for t in traces if t.chosen_rule_id is not None)
        coverage = float(covered) / float(len(traces))
        mean_fired = float(
            np.mean([len(t.fired_rule_ids) for t in traces])
        )
    else:
        coverage = None
        mean_fired = None
    return metrics, coverage, mean_fired
