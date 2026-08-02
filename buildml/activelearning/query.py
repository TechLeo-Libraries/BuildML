"""Query strategies for pool-based active learning (train pool only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.activelearning.fit import pool_masks_from_plan
from buildml.activelearning.results import ActiveLearningPlan, ActiveLearningQueryResult
from buildml.activelearning.types import ActiveLearningStrategy


def suggest_query(
    dataset: Dataset,
    plan: ActiveLearningPlan,
    split_plan: SplitPlan | None,
    *,
    batch_size: int | None = None,
    strategy: ActiveLearningStrategy | None = None,
) -> ActiveLearningQueryResult:
    """Suggest unlabeled *train* indices for the user to label.

    Never queries validation/test. Does not invent labels. Honors the remaining
    label budget when ``plan.label_budget`` is set.
    """
    if plan is None:
        raise ValidationError("No ActiveLearningPlan. Call fit_active_learner first.")
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    # Hard leakage guard: pool must be a subset of train indices.
    train_set = set(split_plan.train_indices)
    test_set = set(split_plan.test_indices or ())
    val_set = set(split_plan.validation_indices or ())

    _, x_pool, _, pool_indices = pool_masks_from_plan(dataset, plan, split_plan)
    for idx in pool_indices:
        if idx not in train_set:
            raise ValidationError(
                f"Active-learning pool index {idx!r} is not in the train partition."
            )
        if idx in test_set or idx in val_set:
            raise ValidationError(
                "Active learning refuses to query validation/test rows. "
                f"Index {idx!r} is outside the train pool."
            )

    resolved_strategy = strategy or plan.strategy
    requested = int(batch_size if batch_size is not None else (plan.config or {}).get("batch_size", 5))
    if requested < 1:
        raise ValidationError("batch_size must be >= 1.")

    budget_remaining = _budget_remaining(plan)
    warnings: list[str] = []
    disclosures = [
        f"Query strategy={resolved_strategy!r}; pool is train unlabeled rows only.",
        "Suggested indices require human labels via label_rows — no oracle in core.",
        "Validation/test partitions are never used as the query pool.",
    ]
    if budget_remaining is not None:
        disclosures.append(
            f"Label budget remaining: {budget_remaining} "
            f"(used={plan.n_queries_used}, budget={plan.label_budget})."
        )
        if budget_remaining <= 0:
            warnings.append("Label budget exhausted; returning no query indices.")
            return ActiveLearningQueryResult(
                strategy=resolved_strategy,
                batch_size_requested=requested,
                indices=(),
                scores=(),
                n_unlabeled_pool=len(pool_indices),
                n_queries_used=plan.n_queries_used,
                label_budget=plan.label_budget,
                budget_remaining=0,
                disclosures=tuple(disclosures),
                warnings=tuple(warnings),
            )
        requested = min(requested, budget_remaining)

    if len(pool_indices) == 0:
        warnings.append("Unlabeled train pool is empty.")
        return ActiveLearningQueryResult(
            strategy=resolved_strategy,
            batch_size_requested=requested,
            indices=(),
            scores=(),
            n_unlabeled_pool=0,
            n_queries_used=plan.n_queries_used,
            label_budget=plan.label_budget,
            budget_remaining=budget_remaining,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    scores = _score_pool(plan, x_pool, resolved_strategy)
    order = np.argsort(-scores)  # higher score = more informative / uncertain
    take = min(requested, len(pool_indices))
    chosen_local = order[:take]
    indices = tuple(pool_indices[i] for i in chosen_local)
    chosen_scores = tuple(float(scores[i]) for i in chosen_local)

    return ActiveLearningQueryResult(
        strategy=resolved_strategy,
        batch_size_requested=requested,
        indices=indices,
        scores=chosen_scores,
        n_unlabeled_pool=len(pool_indices),
        n_queries_used=plan.n_queries_used,
        label_budget=plan.label_budget,
        budget_remaining=budget_remaining,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


# Alias matching the job brief.
query_indices = suggest_query


def _budget_remaining(plan: ActiveLearningPlan) -> int | None:
    if plan.label_budget is None:
        return None
    return max(0, int(plan.label_budget) - int(plan.n_queries_used))


def _score_pool(
    plan: ActiveLearningPlan,
    x_pool: np.ndarray,
    strategy: str,
) -> np.ndarray:
    if strategy in {"least_confidence", "margin", "entropy", "expected_model_change_lite"}:
        if not hasattr(plan.estimator_, "predict_proba"):
            raise ValidationError(
                f"Strategy {strategy!r} requires predict_proba on the base estimator."
            )
        proba = np.asarray(plan.estimator_.predict_proba(x_pool), dtype=float)
        proba = np.clip(proba, 1e-12, 1.0)
        if strategy == "least_confidence":
            return 1.0 - proba.max(axis=1)
        if strategy == "margin":
            # Smaller margin ⇒ higher uncertainty. Score = -(top1 - top2).
            part = np.partition(proba, -2, axis=1)
            top2 = part[:, -2:]
            margin = top2.max(axis=1) - top2.min(axis=1)
            return -margin
        if strategy == "entropy":
            return -np.sum(proba * np.log(proba), axis=1)
        # expected_model_change_lite: ||x|| * (1 - p_max) as a gradient-magnitude proxy
        # for multiclass logistic / similar linear decision surfaces.
        conf = proba.max(axis=1)
        norms = np.linalg.norm(x_pool, axis=1)
        return norms * (1.0 - conf)

    if strategy == "committee":
        committee = plan.committee_
        if committee is None:
            raise ValidationError(
                "Committee strategy requires a fitted committee. "
                "Call fit_active_learner(strategy='committee')."
            )
        # Vote entropy across bagged members.
        member_preds = []
        estimators = getattr(committee, "estimators_", None)
        if not estimators:
            raise ValidationError("Committee has no fitted estimators_.")
        for est in estimators:
            member_preds.append(np.asarray(est.predict(x_pool)))
        votes = np.vstack(member_preds)  # (n_members, n_pool)
        n_members = votes.shape[0]
        # Per-row vote entropy over observed class codes.
        scores = np.zeros(votes.shape[1], dtype=float)
        for j in range(votes.shape[1]):
            _, counts = np.unique(votes[:, j], return_counts=True)
            p = counts.astype(float) / float(n_members)
            p = np.clip(p, 1e-12, 1.0)
            scores[j] = float(-np.sum(p * np.log(p)))
        return scores

    raise ValidationError(
        f"Unsupported active-learning strategy {strategy!r}. "
        "Supported: least_confidence, margin, entropy, committee, "
        "expected_model_change_lite."
    )
