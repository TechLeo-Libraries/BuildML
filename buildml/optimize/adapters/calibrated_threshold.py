"""Sklearn calibrated threshold policy."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.supervised import FitResult, _feature_target_frames
from buildml.optimize.policies import fit_threshold_policy
from buildml.optimize.results import DecisionPlan


def fit_calibrated_threshold_policy(
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
    method: str = "sigmoid",
) -> tuple[DecisionPlan, dict[str, Any], Any]:
    """Calibrate Session.fit estimator on train, then tune threshold."""
    if fit_result.task != "classification":
        raise ValidationError("method='threshold' requires a classification fit.")
    if not hasattr(fit_result.estimator, "predict_proba"):
        raise ValidationError(
            "backend='calibrated' requires a base estimator with predict_proba."
        )

    x_train, y_train, _, _, _ = _feature_target_frames(
        dataset, split_plan, "train"
    )
    x_train = x_train[list(fit_result.feature_columns)]
    try:
        from sklearn.frozen import FrozenEstimator

        base = FrozenEstimator(fit_result.estimator)
        calibrated = CalibratedClassifierCV(base, method=method, cv=3)
        calibrated.fit(x_train, y_train)
    except ImportError:
        calibrated = CalibratedClassifierCV(
            fit_result.estimator,
            method=method,
            cv="prefit",
        )
        calibrated.fit(x_train, y_train)

    aux_fit = FitResult(
        estimator=calibrated,
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
    plan.config = {**plan.config, "backend": "calibrated", "calibration_method": method}
    plan.operating_points = {
        **plan.operating_points,
        "backend": "calibrated",
        "calibration_method": method,
    }
    plan.aux_estimator_ = calibrated
    extra_disclosures = (
        f"backend='calibrated': CalibratedClassifierCV(method={method!r}) on train; "
        f"threshold tuned on partition={partition}.",
        "Auxiliary calibrated estimator stored in DecisionPlan for apply.",
    )
    plan.disclosures = tuple(list(plan.disclosures) + list(extra_disclosures))
    metrics["backend"] = 2.0
    return plan, metrics, diagnostic
