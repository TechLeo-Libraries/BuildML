"""Train-only active learner fit (labeled train rows only; pool stays unlabeled)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.activelearning.features import (
    encode_labeled_targets,
    is_unlabeled_mask,
    matrix_from_frame,
    resolve_activelearning_columns,
)
from buildml.activelearning.results import ActiveLearningFitResult, ActiveLearningPlan
from buildml.activelearning.types import (
    ActiveLearningConfig,
    ActiveLearningEstimator,
    ActiveLearningStrategy,
)

_BASE_ESTIMATORS = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=500, random_state=rs
    ),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(
        random_state=rs
    ),
}


def fit_active_learner(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    strategy: ActiveLearningStrategy = "margin",
    base_estimator: ActiveLearningEstimator = "logistic_regression",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    batch_size: int = 5,
    label_budget: int | None = 50,
    unlabeled_marker: Any = None,
    prefer_reduce_components: bool = True,
    committee_size: int = 5,
    auto_refit: bool = True,
    reduce_plan: Any | None = None,
    prior_plan: ActiveLearningPlan | None = None,
) -> tuple[ActiveLearningPlan, ActiveLearningFitResult]:
    """Fit a supervised classifier on currently labeled train rows only.

    Pool convention
    ---------------
    Unlabeled pool = train rows whose target is missing (NaN by default), matching
    the semi-supervised missingness contract. Validation/test are never used as
    the query pool and never invent labels for selection.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    target = dataset.require_target()
    train = frame_for_partition(dataset, split_plan, "train")
    cols, used_reduce, disclosures = resolve_activelearning_columns(
        dataset,
        train,
        columns if columns is not None else (
            list(prior_plan.columns) if prior_plan is not None else None
        ),
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    x_all = matrix_from_frame(train, cols)
    y_codes, encoder, classes, labeled_mask, n_labeled, n_unlabeled = (
        encode_labeled_targets(
            train[target],
            unlabeled_marker=unlabeled_marker,
            label_encoder=(
                None if prior_plan is None else prior_plan.label_encoder_
            ),
        )
    )
    n_train = int(x_all.shape[0])
    train_indices = list(split_plan.train_indices)
    if len(train_indices) != n_train:
        raise ValidationError(
            "Train frame length does not match SplitPlan.train_indices; "
            "cannot align the active-learning pool."
        )

    labeled_idx = tuple(
        train_indices[i] for i, flag in enumerate(labeled_mask) if flag
    )
    pool_idx = tuple(
        train_indices[i] for i, flag in enumerate(labeled_mask) if not flag
    )

    warnings: list[str] = []
    disclosures.extend(
        [
            "Active learning fits a supervised classifier on labeled train rows only.",
            "The unlabeled pool is train-partition target missingness (NaN by default).",
            "Validation/test partitions are never used as the query pool.",
            "Labels come from the user (human-in-the-loop). BuildML core does not "
            "simulate an oracle; tests/examples may.",
            f"Train mix: n_labeled={n_labeled}, n_unlabeled_pool={n_unlabeled} "
            f"of n_train={n_train}.",
        ]
    )
    if n_unlabeled == 0:
        warnings.append(
            "Unlabeled pool is empty; suggest_query will return no indices until "
            "more train targets are blanked."
        )

    estimator = _build_estimator(base_estimator, random_state)
    try:
        estimator.fit(x_all[labeled_mask], y_codes)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Active learner fit failed for base_estimator={base_estimator!r}: {exc}"
        ) from exc

    committee = None
    if strategy == "committee" or (
        prior_plan is not None and prior_plan.strategy == "committee"
    ):
        # Always keep a committee available when strategy prefers QBC.
        committee = _build_committee(
            base_estimator=base_estimator,
            random_state=random_state,
            committee_size=committee_size,
        )
        try:
            committee.fit(x_all[labeled_mask], y_codes)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Active learning committee fit failed: {exc}"
            ) from exc

    n_queries_used = 0 if prior_plan is None else int(prior_plan.n_queries_used)
    query_history = () if prior_plan is None else tuple(prior_plan.query_history)
    budget = label_budget if prior_plan is None else prior_plan.label_budget
    # Prefer explicit call-site budget when provided on a fresh fit.
    if prior_plan is None:
        budget = label_budget
    resolved_strategy = strategy if prior_plan is None else prior_plan.strategy
    if prior_plan is not None and strategy != prior_plan.strategy:
        # Allow strategy override on explicit refit.
        resolved_strategy = strategy

    config = ActiveLearningConfig(
        strategy=resolved_strategy,  # type: ignore[arg-type]
        base_estimator=base_estimator,
        columns=tuple(cols),
        random_state=random_state,
        batch_size=batch_size if prior_plan is None else int(
            (prior_plan.config or {}).get("batch_size", batch_size)
        ),
        label_budget=budget,
        unlabeled_marker=unlabeled_marker,
        prefer_reduce_components=prefer_reduce_components,
        committee_size=committee_size,
        auto_refit=auto_refit,
    )
    plan = ActiveLearningPlan(
        strategy=resolved_strategy,
        base_estimator=base_estimator,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_labeled_train=n_labeled,
        n_unlabeled_pool=n_unlabeled,
        classes_=classes,
        labeled_train_indices=labeled_idx,
        unlabeled_pool_indices=pool_idx,
        query_history=query_history,
        n_queries_used=n_queries_used,
        label_budget=budget,
        estimator_=estimator,
        label_encoder_=encoder,
        committee_=committee,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = ActiveLearningFitResult(
        strategy=resolved_strategy,
        base_estimator=base_estimator,
        n_train_rows=n_train,
        n_labeled_train=n_labeled,
        n_unlabeled_pool=n_unlabeled,
        n_queries_used=n_queries_used,
        label_budget=budget,
        columns=tuple(cols),
        target_column=target,
        classes=classes,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _build_estimator(base_estimator: str, random_state: int | None) -> Any:
    key = str(base_estimator).lower().replace("-", "_")
    if key not in _BASE_ESTIMATORS:
        raise ValidationError(
            f"Unknown base_estimator={base_estimator!r}. "
            f"Supported: {sorted(_BASE_ESTIMATORS)}"
        )
    return _BASE_ESTIMATORS[key](random_state)


def _build_committee(
    *,
    base_estimator: str,
    random_state: int | None,
    committee_size: int,
) -> BaggingClassifier:
    if int(committee_size) < 2:
        raise ValidationError("committee_size must be >= 2 for query-by-committee.")
    base = _build_estimator(base_estimator, random_state)
    return BaggingClassifier(
        estimator=base,
        n_estimators=int(committee_size),
        bootstrap=True,
        random_state=random_state,
    )


def pool_masks_from_plan(
    dataset: Dataset,
    plan: ActiveLearningPlan,
    split_plan: SplitPlan,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[Any]]:
    """Return train frame, feature matrix for pool rows, pool local mask, pool indices."""
    train = frame_for_partition(dataset, split_plan, "train")
    target = plan.target_column
    marker = (plan.config or {}).get("unlabeled_marker")
    unlabeled = is_unlabeled_mask(train[target], marker)
    train_indices = list(split_plan.train_indices)
    pool_indices = [train_indices[i] for i, flag in enumerate(unlabeled) if flag]
    if not pool_indices:
        x_pool = np.empty((0, len(plan.columns)), dtype=float)
        return train, x_pool, unlabeled, pool_indices
    x_all = matrix_from_frame(train, list(plan.columns))
    x_pool = x_all[unlabeled]
    return train, x_pool, unlabeled, pool_indices
