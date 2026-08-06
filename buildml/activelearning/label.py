"""Incorporate user-provided labels into the Session target (no oracle)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from buildml.activelearning.fit import fit_active_learner
from buildml.activelearning.results import (
    ActiveLearningLabelResult,
    ActiveLearningPlan,
)
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.features import is_unlabeled_mask


def label_rows(
    dataset: Dataset,
    plan: ActiveLearningPlan,
    split_plan: SplitPlan | None,
    *,
    indices: Sequence[Any],
    labels: Sequence[Any],
    refit: bool | None = None,
    reduce_plan: Any | None = None,
) -> tuple[Dataset, ActiveLearningPlan, ActiveLearningLabelResult, Any | None]:
    """Write user labels onto train-pool rows and optionally refit.

    Labels must come from the user: BuildML core never invents an oracle.
    Only train-pool rows that are currently unlabeled (or overwrite with
    disclosure) may be labeled.

    Parameters
    ----------
    dataset:
        BuildML dataset whose target column will be updated in-place logically.
    plan:
        Fitted :class:`~buildml.activelearning.results.ActiveLearningPlan`.
    split_plan:
        Split plan restricting labeling to the train partition.
    indices:
        Dataset-level indices previously suggested (must be in the train pool).
    labels:
        Human-provided labels aligned 1:1 with ``indices``.
    refit:
        When ``None``, uses ``plan.config['auto_refit']`` (default ``True``).
    reduce_plan:
        Optional preprocess reduce plan forwarded to refit.

    Returns
    -------
    new_dataset:
        Dataset copy with updated target values on labeled indices.
    updated_plan:
        Plan with refreshed pool bookkeeping and optional refit.
    label_result:
        Serializable summary of the labeling round.
    fit_result_or_none:
        Fit result when ``refit=True``; otherwise ``None``.

    Raises
    ------
    ValidationError
        When indices are outside train, budget is exceeded, or labels are null.
    """
    if plan is None:
        raise ValidationError("No ActiveLearningPlan. Call fit_active_learner first.")
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    idx_list = list(indices)
    lab_list = list(labels)
    if len(idx_list) != len(lab_list):
        raise ValidationError(
            f"indices/labels length mismatch ({len(idx_list)} vs {len(lab_list)})."
        )
    if not idx_list:
        raise ValidationError("label_rows requires at least one index/label pair.")

    train_set = set(split_plan.train_indices)
    test_set = set(split_plan.test_indices or ())
    val_set = set(split_plan.validation_indices or ())
    for idx in idx_list:
        if idx in test_set or idx in val_set:
            raise ValidationError(
                "Refusing to label validation/test rows for active learning "
                f"(index={idx!r}). Query pool is train-only."
            )
        if idx not in train_set:
            raise ValidationError(
                f"Index {idx!r} is not in the train partition; cannot label."
            )

    budget = plan.label_budget
    n_new = len(idx_list)
    if budget is not None and plan.n_queries_used + n_new > int(budget):
        raise ValidationError(
            f"Label budget exceeded: used={plan.n_queries_used}, "
            f"adding={n_new}, budget={budget}. Reduce the batch or raise label_budget."
        )

    target = plan.target_column
    full = dataset._ensure_pandas().copy()
    if target not in full.columns:
        raise ValidationError(f"Target column {target!r} missing from dataset.")

    marker = (plan.config or {}).get("unlabeled_marker")
    # Only allow labeling currently-unlabeled train rows (or re-label with disclosure).
    warnings: list[str] = []
    for idx, lab in zip(idx_list, lab_list):
        if idx not in full.index:
            raise ValidationError(f"Index {idx!r} not present in the Session frame.")
        current = full.loc[idx, target]
        series = pd.Series([current])
        if not bool(is_unlabeled_mask(series, marker)[0]):
            warnings.append(
                f"Index {idx!r} already had a label; overwriting with user-provided value."
            )
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            raise ValidationError(
                f"Refusing null label for index {idx!r}. Active learning labels "
                "must be concrete user-provided values."
            )
        full.loc[idx, target] = lab

    roles = dict(dataset.roles)
    new_dataset = Dataset.from_transformed(
        dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=roles,
    )

    round_record = {
        "indices": list(idx_list),
        "n_labeled": n_new,
        "strategy": plan.strategy,
        "n_queries_used_before": plan.n_queries_used,
    }
    updated_history = tuple(plan.query_history) + (round_record,)
    n_queries_used = int(plan.n_queries_used) + n_new
    budget_remaining = None if budget is None else max(0, int(budget) - n_queries_used)

    # Stash history onto a shallow plan copy before optional refit.
    plan.query_history = updated_history
    plan.n_queries_used = n_queries_used

    do_refit = bool((plan.config or {}).get("auto_refit", True)) if refit is None else bool(refit)
    fit_result = None
    disclosures = [
        "Labels were supplied by the user (human-in-the-loop). "
        "BuildML core does not invent oracle labels.",
        f"Incorporated {n_new} label(s) on the train partition.",
        f"Query budget: used={n_queries_used}, budget={budget}, "
        f"remaining={budget_remaining}.",
    ]

    if do_refit:
        new_plan, fit_result = fit_active_learner(
            new_dataset,
            split_plan,
            backend=plan.backend,  # type: ignore[arg-type]
            strategy=plan.strategy,  # type: ignore[arg-type]
            base_estimator=plan.base_estimator,  # type: ignore[arg-type]
            columns=list(plan.columns),
            random_state=(plan.config or {}).get("random_state", 0),
            batch_size=int((plan.config or {}).get("batch_size", 5)),
            label_budget=budget,
            unlabeled_marker=marker,
            prefer_reduce_components=bool(
                (plan.config or {}).get("prefer_reduce_components", True)
            ),
            committee_size=int((plan.config or {}).get("committee_size", 5)),
            auto_refit=bool((plan.config or {}).get("auto_refit", True)),
            epochs=int((plan.config or {}).get("epochs", 60)),
            learning_rate=float((plan.config or {}).get("learning_rate", 1e-3)),
            mc_samples=int((plan.config or {}).get("mc_samples", 20)),
            device=str((plan.config or {}).get("device", "cpu")),
            reduce_plan=reduce_plan,
            prior_plan=plan,
        )
        updated_plan = new_plan
        disclosures.append("Refit the active learner on the expanded labeled train set.")
    else:
        # Refresh pool bookkeeping without refitting the estimator.
        from buildml.data.splits import frame_for_partition

        train = frame_for_partition(new_dataset, split_plan, "train")
        unlabeled = is_unlabeled_mask(train[target], marker)
        train_indices = list(split_plan.train_indices)
        plan.labeled_train_indices = tuple(
            train_indices[i] for i, flag in enumerate(unlabeled) if not flag
        )
        plan.unlabeled_pool_indices = tuple(
            train_indices[i] for i, flag in enumerate(unlabeled) if flag
        )
        plan.n_labeled_train = int((~unlabeled).sum())
        plan.n_unlabeled_pool = int(unlabeled.sum())
        updated_plan = plan
        disclosures.append(
            "Labels written without refit (refit=False). Call fit_active_learner to update the model."
        )

    # Count labeled now on train.
    n_labeled_now = updated_plan.n_labeled_train
    result = ActiveLearningLabelResult(
        n_labeled_now=n_labeled_now,
        n_newly_labeled=n_new,
        indices=tuple(idx_list),
        n_queries_used=n_queries_used,
        label_budget=budget,
        budget_remaining=budget_remaining,
        refit=do_refit,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return new_dataset, updated_plan, result, fit_result
