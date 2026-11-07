"""Episodic holdout evaluation for meta-learning (never for meta-train)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from buildml.core.errors import ValidationError
from buildml.core.validation import _positive_integer
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.metalearning.features import (
    compute_prototypes,
    encode_labels,
    frame_for_task,
    matrix_from_frame,
    nearest_prototype_predict,
    sample_support_query,
    task_ids_in_frame,
)
from buildml.metalearning.results import MetaLearningEvalResult, MetaLearningPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_metalearning(
    dataset: Dataset,
    plan: MetaLearningPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    k_shot: int | None = None,
    n_query: int | None = None,
    n_way: int | None = None,
    prefer_novel_tasks: bool = True,
    random_state: int | None = 0,
) -> MetaLearningEvalResult:
    """Score episodic few-shot performance on a holdout partition.

    Prefer tasks whose ids were **not** in meta-train (true novel-task
    few-shot). When only overlapping task ids exist, evaluate with a clear
    disclosure that this is not out-of-task generalization.

    Holdout rows are never used for meta-training.

    Parameters
    ----------
    dataset:
        BuildML dataset with features, target, and task columns.
    plan:
        Fitted :class:`~buildml.metalearning.results.MetaLearningPlan`.
    split_plan:
        Split plan; required unless ``partition='all'``.
    partition:
        Holdout partition to score (``validation``, ``test``, ``train``, or ``all``).
    k_shot:
        Positive integral override for episodic support size.
    n_query:
        Positive integral override for episodic query size.
    n_way:
        Integral classes per episode, between two and the fitted class count.
        Boolean, fractional and string episode counts are refused.
    prefer_novel_tasks:
        When ``True``, prefer task ids absent from meta-train, including internal
        held-out tasks on train. When ``False``, score all observed task ids.
    random_state:
        Seed for episodic support/query sampling.

    Returns
    -------
    MetaLearningEvalResult
        Aggregate and per-task episodic metrics with overlap disclosures.

    Raises
    ------
    ValidationError
        When plan, partition, or column preconditions are invalid.
    """
    if plan is None:
        raise ValidationError("No MetaLearningPlan. Call fit_metalearning first.")
    k = _positive_integer(plan.k_shot if k_shot is None else k_shot, name="k_shot")
    nq = _positive_integer(plan.n_query if n_query is None else n_query, name="n_query")
    nw = _positive_integer(plan.n_way if n_way is None else n_way, name="n_way")
    if nw < 2 or nw > len(plan.classes_):
        raise ValidationError(f"n_way must be between 2 and {len(plan.classes_)} fitted classes.")

    if partition == "all":
        frame = dataset._ensure_pandas()
        part_name = "all"
    else:
        if split_plan is None:
            raise ValidationError(
                f"partition='{partition}' requires a SplitPlan. "
                "Call session.split(...)."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
        part_name = str(partition)

    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for evaluation: {missing}")
    for col in (plan.target_column, plan.task_column):
        if col not in frame.columns:
            raise ValidationError(f"Column {col!r} missing from evaluation frame.")

    rng = np.random.default_rng(random_state)

    eval_task_ids = task_ids_in_frame(frame, plan.task_column)
    train_set = set(plan.train_task_ids)
    novel = [t for t in eval_task_ids if t not in train_set]
    overlapping = [t for t in eval_task_ids if t in train_set]

    disclosures = [
        "Meta-learning evaluation forms disjoint support/query episodes "
        "within the requested partition.",
        f"method={plan.method}, partition={part_name}, k_shot={k}, "
        f"n_query={nq}, n_way={nw}.",
    ]
    if part_name in {"validation", "test"}:
        disclosures.append("Holdout rows were never used for meta-training.")
    warnings: list[str] = []

    if prefer_novel_tasks and novel:
        selected = novel
        disclosures.append(
            f"Evaluating {len(selected)} novel task id(s) absent from "
            f"meta-train: {selected}."
        )
        if overlapping:
            disclosures.append(
                f"Overlapping task ids present but skipped "
                f"(prefer_novel_tasks=True): {overlapping}."
            )
    elif not prefer_novel_tasks and eval_task_ids:
        selected = eval_task_ids
        disclosures.append(
            f"Evaluating all observed task ids (prefer_novel_tasks=False): {selected}."
        )
    elif overlapping:
        selected = overlapping
        disclosures.append(
            f"Evaluating overlapping task ids: {selected}."
        )
    else:
        warnings.append("Evaluation partition has no task ids.")
        return MetaLearningEvalResult(
            partition=part_name,
            method=plan.method,
            n_tasks_evaluated=0,
            n_query_rows=0,
            metrics={},
            per_task_metrics={},
            novel_task_ids=tuple(novel),
            overlapping_task_ids=tuple(overlapping),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    # Prefer internal held-out train tasks when evaluating train partition.
    if prefer_novel_tasks and part_name == "train" and plan.held_out_task_ids:
        held = [t for t in plan.held_out_task_ids if t in set(eval_task_ids)]
        if held:
            selected = held
            disclosures.append(
                f"Using fit-time held_out_task_ids for train-partition "
                f"episodic eval: {held}."
            )

    if len({str(task) for task in selected}) != len(selected):
        raise ValidationError(
            "Task report key collision: distinct task ids share a display key. "
            "Recode task identifiers to unique text before evaluation."
        )
    selected_overlap = [task for task in selected if task in train_set]
    if selected_overlap:
        prefix = "Including" if novel else "No novel task ids; scoring"
        warnings.append(
            f"{prefix} overlapping tasks with within-partition support/query splits. "
            "This is not true out-of-task few-shot generalization. Prefer a "
            "task-disjoint split or rely on held_out_task_ids from fit."
        )
        if part_name in {"train", "all"}:
            warnings.append("Selected overlapping tasks include meta-training rows.")

    per_task: dict[str, dict[str, float]] = {}
    accs: list[float] = []
    f1s: list[float] = []
    total_query = 0
    skipped = 0

    for task_id in selected:
        task_frame = frame_for_task(frame, plan.task_column, task_id)
        sampled = sample_support_query(
            task_frame,
            target_column=plan.target_column,
            columns=list(plan.columns),
            label_encoder=plan.label_encoder_,
            k_shot=k,
            n_query=nq,
            n_way=nw,
            rng=rng,
        )
        if sampled is None:
            skipped += 1
            continue
        support, query, _ = sampled
        y_true_codes, _, _ = encode_labels(
            query[plan.target_column], label_encoder=plan.label_encoder_
        )
        pred_codes = _predict_episode(
            plan, support, query, list(plan.columns)
        )
        acc = float(accuracy_score(y_true_codes, pred_codes))
        f1 = float(
            f1_score(
                y_true_codes,
                pred_codes,
                average="macro",
                zero_division=0,
            )
        )
        key = str(task_id)
        per_task[key] = {
            "accuracy": acc,
            "f1_macro": f1,
            "n_support": float(len(support)),
            "n_query": float(len(query)),
        }
        accs.append(acc)
        f1s.append(f1)
        total_query += int(len(query))

    if skipped:
        warnings.append(
            f"Skipped {skipped}/{len(selected)} task(s) with insufficient "
            "per-class rows for support/query."
        )

    metrics: dict[str, float] = {}
    if accs:
        metrics = {
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "n_tasks_scored": float(len(accs)),
        }
    else:
        warnings.append("No tasks produced evaluable episodes; metrics empty.")

    return MetaLearningEvalResult(
        partition=part_name,
        method=plan.method,
        n_tasks_evaluated=len(accs),
        n_query_rows=total_query,
        metrics=metrics,
        per_task_metrics=per_task,
        novel_task_ids=tuple(novel),
        overlapping_task_ids=tuple(overlapping),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _predict_episode(
    plan: MetaLearningPlan,
    support: Any,
    query: Any,
    columns: list[str],
) -> np.ndarray:
    x_s = matrix_from_frame(support, columns)
    y_s, _, _ = encode_labels(
        support[plan.target_column], label_encoder=plan.label_encoder_
    )
    x_q = matrix_from_frame(query, columns)
    if plan.method == "prototypical":
        protos = compute_prototypes(x_s, y_s)
        return nearest_prototype_predict(x_q, protos)
    if plan.method == "prototypical_torch":
        if plan.meta_learner_ is None:
            raise ValidationError("Torch prototypical plan missing meta_learner_.")
        return plan.meta_learner_.predict_from_support(x_s, y_s, x_q)
    if plan.method in {"maml", "reptile"}:
        if plan.meta_learner_ is None:
            raise ValidationError(f"{plan.method} plan missing meta_learner_.")
        return plan.meta_learner_.adapt_predict(x_s, y_s, x_q)
    if plan.method == "warm_start":
        if plan.init_estimator_ is None:
            raise ValidationError("Warm-start plan missing init_estimator_.")
        adapted = deepcopy(plan.init_estimator_)
        adapted.fit(x_s, y_s)
        return np.asarray(adapted.predict(x_q), dtype=int)
    raise ValidationError(f"Unknown plan.method={plan.method!r}.")
