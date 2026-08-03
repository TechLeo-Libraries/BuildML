"""Score-only symbolic / neuro-symbolic prediction with rule traces."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.symbolic.features import decode_predictions, matrix_from_frame
from buildml.symbolic.results import (
    NeuroSymbolicPlan,
    SymbolicPlan,
    SymbolicPredictResult,
)
from buildml.symbolic.rules import (
    RuleTrace,
    evaluate_predicate,
    fire_rules,
    rule_feature_matrix,
)

PartitionOrAll = PartitionName | Literal["all"]


def predict_symbolic(
    dataset: Dataset,
    plan: SymbolicPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    return_traces: bool = True,
) -> SymbolicPredictResult:
    """Apply the compiled rule base to a partition without refit.

    Evaluates decision-list semantics row by row and optionally attaches
    per-row firing traces for explain hooks and evaluate coverage stats.

    Parameters
    ----------
    dataset:
        Session dataset.
    plan:
        Train-fitted symbolic plan with knowledge base.
    split_plan:
        Split plan for the requested partition.
    partition:
        ``train``, ``validation``, ``test``, or ``all``.
    return_traces:
        When True, include per-row rule firing traces.

    Returns
    -------
    SymbolicPredictResult
        Predictions and optional explanation traces.
    """
    frame, indices = _partition_frame(dataset, split_plan, partition)
    preds, traces, _ = fire_rules(frame, plan.knowledge_base, row_indices=indices)
    if not return_traces:
        traces = []
    return SymbolicPredictResult(
        partition=str(partition),
        path="symbolic",
        task=plan.task,
        n_rows=len(frame),
        predictions=tuple(preds),
        traces=tuple(traces),
        disclosures=plan.disclosures,
        warnings=(),
    )


def predict_neuro_symbolic(
    dataset: Dataset,
    plan: NeuroSymbolicPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    return_traces: bool = True,
) -> SymbolicPredictResult:
    """Predict with a neuro-symbolic hybrid without refit.

    Applies constraint overlay, rules-as-features, or constraint-repair modes
    on top of the train-fitted base estimator.

    Parameters
    ----------
    dataset:
        Session dataset.
    plan:
        Train-fitted neuro-symbolic plan.
    split_plan:
        Split plan for the requested partition.
    partition:
        ``train``, ``validation``, ``test``, or ``all``.
    return_traces:
        When True, include per-row traces with neural vs final predictions.

    Returns
    -------
    SymbolicPredictResult
        Final predictions, neural baseline, traces, and repair counts.
    """
    frame, indices = _partition_frame(dataset, split_plan, partition)
    x = matrix_from_frame(frame, list(plan.columns))
    mode = plan.mode

    if mode == "rules_as_features":
        r_mat, _ = rule_feature_matrix(frame, plan.knowledge_base)
        x_aug = np.hstack([x, r_mat])
        raw = plan.estimator_.predict(x_aug)
        neural_preds = _decode(plan, raw)
        # Traces still report which rules fired as features.
        _, base_traces, _ = fire_rules(
            frame, plan.knowledge_base, row_indices=indices
        )
        traces = [
            RuleTrace(
                row_index=t.row_index,
                fired_rule_ids=t.fired_rule_ids,
                chosen_rule_id=t.chosen_rule_id,
                prediction=neural_preds[i],
                neural_prediction=neural_preds[i],
                repaired=False,
                notes=("rules_as_features: prediction from augmented model.",),
            )
            for i, t in enumerate(base_traces)
        ]
        final = neural_preds
        n_repaired = 0
    else:
        raw = plan.estimator_.predict(x)
        neural_preds = _decode(plan, raw)
        final, traces, n_repaired = _apply_symbolic_overlay(
            frame,
            indices,
            neural_preds,
            plan,
            repair=(mode == "constraint_repair"),
        )

    if not return_traces:
        traces = []
    return SymbolicPredictResult(
        partition=str(partition),
        path="neuro_symbolic",
        task=plan.task,
        n_rows=len(frame),
        predictions=tuple(final),
        traces=tuple(traces),
        neural_predictions=tuple(neural_preds),
        n_repaired=n_repaired,
        disclosures=plan.disclosures,
        warnings=(),
    )


def _apply_symbolic_overlay(
    frame: pd.DataFrame,
    indices: list[Any],
    neural_preds: list[Any],
    plan: NeuroSymbolicPlan,
    *,
    repair: bool,
) -> tuple[list[Any], list[RuleTrace], int]:
    """Apply hard/soft rules on top of neural predictions."""
    kb = plan.knowledge_base
    soft = float(plan.soft_strength)
    final: list[Any] = []
    traces: list[RuleTrace] = []
    n_repaired = 0

    # Precompute fire masks.
    fire_masks: list[np.ndarray] = []
    for rule in kb.rules:
        if not rule.antecedents:
            fire_masks.append(np.ones(len(frame), dtype=bool))
        else:
            mask = np.ones(len(frame), dtype=bool)
            for pred in rule.antecedents:
                mask &= evaluate_predicate(frame[pred.column], pred)
            fire_masks.append(mask)

    for i in range(len(frame)):
        neural = neural_preds[i]
        pred = neural
        fired: list[str] = []
        chosen: str | None = None
        repaired = False
        notes: list[str] = []

        # Rules are already priority-sorted in KB.
        for j, rule in enumerate(kb.rules):
            if not fire_masks[j][i]:
                continue
            fired.append(rule.rule_id)
            is_constraint = rule.kind == "constraint" or repair
            if rule.hardness == "hard":
                if repair or is_constraint or rule.kind in {
                    "classification",
                    "regression",
                    "constraint",
                }:
                    # Hard overlay / repair: override neural prediction.
                    if str(pred) != str(rule.consequent):
                        if repair or rule.kind == "constraint":
                            repaired = True
                            notes.append(
                                f"Hard rule {rule.rule_id} repaired "
                                f"{neural!r} → {rule.consequent!r}."
                            )
                        else:
                            notes.append(
                                f"Hard rule {rule.rule_id} overlaid "
                                f"{neural!r} → {rule.consequent!r}."
                            )
                    pred = rule.consequent
                    chosen = rule.rule_id
                    break
            elif rule.hardness == "soft":
                if plan.task == "regression":
                    try:
                        pred = (1.0 - soft * rule.strength) * float(neural) + (
                            soft * rule.strength
                        ) * float(rule.consequent)
                        chosen = rule.rule_id
                        notes.append(
                            f"Soft rule {rule.rule_id} blended with "
                            f"strength={soft * rule.strength:.3f}."
                        )
                        break
                    except (TypeError, ValueError):
                        notes.append(
                            f"Soft rule {rule.rule_id} skipped (non-numeric)."
                        )
                else:
                    # Soft classification: override when strength*soft >= 0.5
                    if soft * rule.strength >= 0.5:
                        if str(pred) != str(rule.consequent):
                            notes.append(
                                f"Soft rule {rule.rule_id} preferred "
                                f"(strength={soft * rule.strength:.3f})."
                            )
                        pred = rule.consequent
                        chosen = rule.rule_id
                        break
                    notes.append(
                        f"Soft rule {rule.rule_id} fired but below threshold."
                    )

        if repaired:
            n_repaired += 1
        final.append(pred)
        traces.append(
            RuleTrace(
                row_index=indices[i],
                fired_rule_ids=tuple(fired),
                chosen_rule_id=chosen,
                prediction=pred,
                neural_prediction=neural,
                repaired=repaired,
                notes=tuple(notes),
            )
        )
    return final, traces, n_repaired


def _decode(plan: NeuroSymbolicPlan, raw: np.ndarray) -> list[Any]:
    if plan.task == "classification" and plan.label_encoder_ is not None:
        return decode_predictions(np.asarray(raw), plan.label_encoder_)
    if plan.task == "regression":
        return [float(v) for v in np.asarray(raw).ravel()]
    return list(np.asarray(raw).ravel())


def _partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
) -> tuple[pd.DataFrame, list[Any]]:
    if partition == "all":
        frame = dataset._ensure_pandas()
        return frame, list(frame.index)
    if split_plan is None:
        raise ValidationError(
            "predict_symbolic / predict_neuro_symbolic require a SplitPlan "
            "unless partition='all'."
        )
    frame = frame_for_partition(dataset, split_plan, partition)
    return frame, list(frame.index)
