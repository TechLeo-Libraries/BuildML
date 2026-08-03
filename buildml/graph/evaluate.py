"""Evaluate Graph ML node classifiers on a holdout partition."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.graph.data import partition_node_mask, target_array
from buildml.graph.predict import predict_graph
from buildml.graph.results import GraphEvalResult, GraphPlan

PartitionOrAll = Literal["train", "validation", "test", "all"]


def evaluate_graph(
    dataset: Dataset,
    plan: GraphPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> GraphEvalResult:
    """Score holdout nodes and compute classification metrics.

    Uses the fitted plan only: never refits on the evaluation partition.
    Falls back from validation to test when validation indices are absent.

    Parameters
    ----------
    dataset:
        Session dataset holding the node table.
    plan:
        Fitted :class:`GraphPlan` from :func:`fit_graph`.
    split_plan:
        Session split plan defining holdout indices.
    partition:
        Holdout partition to score (defaults to validation).

    Returns
    -------
    GraphEvalResult
        Accuracy, macro-F1, optional ROC-AUC, and honesty disclosures.

    Raises
    ------
    ValidationError
        When no plan exists, no holdout partition is available, or prediction
        length disagrees with labels.
    """
    if plan is None:
        raise ValidationError("No GraphPlan. Call fit_graph(...) first.")

    # Prefer validation; fall back to test when validation is absent.
    resolved = partition
    if partition == "validation" and split_plan is not None:
        if not split_plan.validation_indices:
            if split_plan.test_indices:
                resolved = "test"
            else:
                raise ValidationError(
                    "No validation or test partition available for evaluate_graph."
                )

    pred = predict_graph(
        dataset, plan, split_plan, partition=resolved  # type: ignore[arg-type]
    )
    # Align labels to the same ascending positional order as predict_graph's
    # boolean mask (split index tuples are not necessarily sorted).
    frame = dataset._ensure_pandas()
    n_nodes = int(len(frame))
    if resolved == "all" or split_plan is None:
        score_mask = np.ones(n_nodes, dtype=bool)
    else:
        score_mask = partition_node_mask(n_nodes, split_plan, resolved)
    y_true = target_array(frame, plan.target_column)[score_mask]
    y_pred = np.asarray(pred.predictions)
    if len(y_true) != len(y_pred):
        raise ValidationError(
            "Prediction length does not match partition rows "
            f"({len(y_pred)} vs {len(y_true)})."
        )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    disclosures = list(pred.disclosures)
    disclosures.append(
        "Holdout metrics use a frozen GraphPlan; structure filtering follows "
        f"mode={plan.mode}."
    )
    warnings: list[str] = list(pred.warnings)
    if resolved != partition:
        disclosures.append(
            f"Requested partition={partition!r}; evaluated on {resolved!r} "
            "(validation absent)."
        )

    if (
        pred.probabilities is not None
        and len(plan.classes_) == 2
        and pred.probabilities
    ):
        try:
            # Positive class = classes_[1]
            proba = np.asarray(pred.probabilities, dtype=np.float64)[:, 1]
            # Map labels to {0,1} aligned with plan.classes_
            pos = plan.classes_[1]
            y_bin = (y_true == pos).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_bin, proba))
        except Exception:
            warnings.append("roc_auc could not be computed for this partition.")

    return GraphEvalResult(
        partition=resolved,
        method=plan.method,
        mode=plan.mode,
        n_nodes=int(len(y_true)),
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
