"""Query strategies for pool-based active learning (train pool only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.activelearning.adapters.scikit_activeml import score_industry_pool
from buildml.activelearning.adapters.sklearn import score_sklearn_pool
from buildml.activelearning.adapters.torch_uncertainty import score_torch_pool
from buildml.activelearning.catalog import resolve_backend_strategy
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
    backend: str | None = None,
) -> ActiveLearningQueryResult:
    """Suggest unlabeled train indices for the user to label.

    Never queries validation/test. Does not invent labels. Honors the remaining
    label budget when ``plan.label_budget`` is set.

    Parameters
    ----------
    dataset:
        BuildML dataset containing the train partition and target column.
    plan:
        Fitted :class:`~buildml.activelearning.results.ActiveLearningPlan`.
    split_plan:
        Split plan restricting the query pool to train indices.
    batch_size:
        Optional override for how many indices to suggest this round.
    strategy:
        Optional query strategy override; defaults to ``plan.strategy``.
    backend:
        Optional backend override; defaults to ``plan.backend``.

    Returns
    -------
    ActiveLearningQueryResult
        Suggested indices, scores, pool size, and budget remaining.

    Raises
    ------
    ValidationError
        When no plan exists, pool indices leave train, or ``batch_size`` is invalid.
    MissingExtraError
        When the resolved backend requires an optional extra.
    """
    if plan is None:
        raise ValidationError("No ActiveLearningPlan. Call fit_active_learner first.")
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train_set = set(split_plan.train_indices)
    test_set = set(split_plan.test_indices or ())
    val_set = set(split_plan.validation_indices or ())

    _, x_pool, _, pool_indices, x_labeled, y_labeled = pool_masks_from_plan(
        dataset, plan, split_plan
    )
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
    resolved_backend = backend or plan.backend
    _, resolved_strategy = resolve_backend_strategy(
        backend=resolved_backend, strategy=resolved_strategy
    )

    requested = int(batch_size if batch_size is not None else (plan.config or {}).get("batch_size", 5))
    if requested < 1:
        raise ValidationError("batch_size must be >= 1.")

    budget_remaining = _budget_remaining(plan)
    warnings: list[str] = []
    disclosures = [
        f"Query backend={resolved_backend}, strategy={resolved_strategy!r}; "
        "pool is train unlabeled rows only.",
        "Suggested indices require human labels via label_rows: no oracle in core.",
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

    scores = _score_pool(
        plan,
        x_pool,
        resolved_backend,
        resolved_strategy,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        disclosures=disclosures,
    )
    order = np.argsort(-scores)
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


query_indices = suggest_query


def _budget_remaining(plan: ActiveLearningPlan) -> int | None:
    if plan.label_budget is None:
        return None
    return max(0, int(plan.label_budget) - int(plan.n_queries_used))


def _score_pool(
    plan: ActiveLearningPlan,
    x_pool: np.ndarray,
    backend: str,
    strategy: str,
    *,
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    disclosures: list[str] | None = None,
) -> np.ndarray:
    if backend == "torch":
        mc_samples = int((plan.config or {}).get("mc_samples", 20))
        return score_torch_pool(
            strategy=strategy,  # type: ignore[arg-type]
            x_pool=x_pool,
            estimator=plan.estimator_,
            mc_samples=mc_samples,
        )
    if backend == "industry":
        scores, industry_notes = score_industry_pool(
            strategy=strategy,
            x_labeled=x_labeled,
            y_labeled=y_labeled,
            x_pool=x_pool,
            estimator=plan.estimator_,
            committee=plan.committee_,
        )
        if disclosures is not None:
            disclosures.extend(industry_notes)
        return scores
    return score_sklearn_pool(
        strategy=strategy,
        x_pool=x_pool,
        estimator=plan.estimator_,
        committee=plan.committee_,
    )
