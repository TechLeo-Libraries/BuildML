"""XGBoost cost-sensitive threshold policy (optimize-industry)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.supervised import FitResult, _feature_target_frames
from buildml.optimize.extras import require_xgboost
from buildml.optimize.policies import fit_threshold_policy
from buildml.optimize.results import DecisionPlan


def fit_xgb_threshold_policy(
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    *,
    partition: str,
    allow_test_tuning: bool,
    fp_cost: float | None,
    fn_cost: float | None,
    tp_benefit: float,
    tn_benefit: float,
) -> tuple[DecisionPlan, dict[str, Any], Any]:
    """Train a cost-sensitive XGBoost classifier and tune its threshold.

    Fits :class:`~xgboost.XGBClassifier` on train with ``scale_pos_weight``
    derived from ``fn_cost / fp_cost``, then selects an operating threshold
    via :func:`~buildml.optimize.policies.fit_threshold_policy` on the tuning
    partition. Binary classification only.

    Parameters
    ----------
    dataset:
        Tabular data with features and target for scoring partitions.
    split_plan:
        Train/validation/test split used to isolate the tuning partition.
    fit_result:
        Classification :class:`~buildml.model.supervised.FitResult` defining
        feature columns and positive-class convention.
    partition:
        Split name where the operating threshold is selected.
    allow_test_tuning:
        When ``True``, permits tuning on the test partition (dangerous opt-in).
    fp_cost, fn_cost:
        False-positive and false-negative costs; both required for this backend.
    tp_benefit, tn_benefit:
        Optional benefits subtracted from total expected cost during threshold
        selection.

    Returns
    -------
    tuple[DecisionPlan, dict[str, Any], DiagnosticReport]
        Frozen policy with auxiliary XGB estimator, fit metrics, and threshold
        diagnostic report.

    Raises
    ------
    ValidationError
        When the fit is not binary classification, costs are missing, or
        training/threshold steps fail validation.
    """
    if fit_result.task != "classification":
        raise ValidationError("method='threshold' requires a classification fit.")
    if fp_cost is None or fn_cost is None:
        raise ValidationError(
            "backend='xgb' threshold policy requires fp_cost and fn_cost."
        )

    xgb = require_xgboost()
    x_train, y_train, _, _, _ = _feature_target_frames(
        dataset, split_plan, "train"
    )
    x_train = x_train[list(fit_result.feature_columns)]
    y_series = np.asarray(y_train).astype(str)
    classes = [str(c) for c in fit_result.estimator.classes_]
    if len(classes) != 2:
        raise ValidationError("backend='xgb' threshold policy is binary only.")
    positive = str(classes[1])
    y_bin = (y_series == positive).astype(int)

    scale = float(fn_cost) / max(float(fp_cost), 1e-12)
    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
    )
    clf.fit(x_train.to_numpy(), y_bin)

    aux_fit = FitResult(
        estimator=clf,
        task="classification",
        feature_columns=tuple(fit_result.feature_columns),
        target_column=fit_result.target_column,
        n_train_rows=int(len(x_train)),
        weight_column=fit_result.weight_column,
    )
    plan, metrics, diagnostic = fit_threshold_policy(
        dataset,
        split_plan,
        aux_fit,
        partition=partition,
        allow_test_tuning=allow_test_tuning,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
    )
    plan.positive_class = positive
    pos_idx = 1  # XGB trained on {0,1} with positive=1
    plan.config = {**plan.config, "backend": "xgb", "scale_pos_weight": scale}
    plan.operating_points = {
        **plan.operating_points,
        "backend": "xgb",
        "scale_pos_weight": scale,
        "positive_class_index": pos_idx,
    }
    plan.aux_estimator_ = clf
    extra_disclosures = (
        "backend='xgb': XGBClassifier trained on train with scale_pos_weight "
        f"from fn_cost/fp_cost={scale:.4f}; threshold tuned on partition "
        f"{partition}.",
        "Auxiliary estimator stored in DecisionPlan for apply: not Session.fit.",
    )
    plan.disclosures = tuple(list(plan.disclosures) + list(extra_disclosures))
    metrics["backend"] = 1.0
    metrics["scale_pos_weight"] = scale
    return plan, metrics, diagnostic
