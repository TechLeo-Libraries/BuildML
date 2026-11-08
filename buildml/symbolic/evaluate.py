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
    """Score the symbolic rule base on the requested partition without refit.

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
        ``train``, ``validation``, ``test``, or ``all``. Train/all include fitting rows.

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
        y_true, pred.predictions, pred.traces, task=plan.task,
        legacy_labels=not plan.config.get("raw_class_labels", False),
    )
    warnings = list(pred.warnings)
    if partition in {"train", "all"}:
        warnings.append("Evaluation includes training rows; these metrics are not independent holdout evidence.")
    return SymbolicEvalResult(
        partition=str(partition),
        path="symbolic",
        task=plan.task,
        n_rows=pred.n_rows,
        metrics=metrics,
        rule_coverage=coverage,
        mean_rules_fired=mean_fired,
        disclosures=(
            "Evaluation scores the requested partition without re-inducing rules.",
            *plan.disclosures[:2],
        ),
        warnings=tuple(warnings),
    )


def evaluate_neuro_symbolic(
    dataset: Dataset,
    plan: NeuroSymbolicPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> SymbolicEvalResult:
    """Score the neuro-symbolic hybrid on the requested partition without refit.

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
        ``train``, ``validation``, ``test``, or ``all``. Train/all include fitting rows.

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
        y_true, pred.predictions, pred.traces, task=plan.task,
        legacy_labels=not plan.config.get("raw_class_labels", False),
    )
    repair_rate = (
        float(pred.n_repaired) / float(pred.n_rows) if pred.n_rows else None
    )
    # Fidelity: agreement between neural and final when neural present.
    if pred.neural_predictions is not None:
        agree = sum(
            classification_accuracy([a], [b], legacy_labels=not plan.config.get("raw_class_labels", False)) == 1.0
            for a, b in zip(
                pred.neural_predictions, pred.predictions, strict=True
            )
        )
        metrics = {
            **metrics,
            "neural_final_agreement": float(agree) / float(max(pred.n_rows, 1)),
        }
    warnings = list(pred.warnings)
    if partition in {"train", "all"}:
        warnings.append("Evaluation includes training rows; these metrics are not independent holdout evidence.")
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
            "Evaluation scores the requested partition without refitting "
            "the base estimator or rules.",
            *plan.disclosures[:2],
        ),
        warnings=tuple(warnings),
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
    legacy_labels: bool = False,
) -> tuple[dict[str, float], float | None, float | None]:
    if len(y_true) != len(predictions):
        raise ValidationError(
            "Prediction length does not match evaluation target length."
        )
    if task == "classification":
        metrics = {
            "accuracy": classification_accuracy(
                y_true.tolist(), list(predictions), legacy_labels=legacy_labels
            )
        }
    else:
        if np.iscomplexobj(y_true.to_numpy()) or np.iscomplexobj(np.asarray(predictions)):
            raise ValidationError(
                "Symbolic regression evaluation requires finite real targets and predictions."
            )
        try:
            y_actual = y_true.to_numpy(dtype=float)
            y_hat = np.asarray(
                [float(p) if p is not None else float("nan") for p in predictions], dtype=float
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "Symbolic regression evaluation requires finite real targets and predictions."
            ) from exc
        if not np.isfinite(y_actual).all() or not np.isfinite(y_hat).all():
            raise ValidationError(
                "Symbolic regression evaluation requires finite real targets and predictions "
                "(check missing defaults, unmatched rules and non-finite values)."
            )
        metrics = regression_metrics(y_actual, y_hat)

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
