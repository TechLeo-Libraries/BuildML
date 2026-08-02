"""Holdout evaluation for causal nuisance quality + effect estimate."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from buildml.causal.estimate import (
    _predict_outcome,
    _predict_propensity,
    estimate_ate_from_models,
    estimate_causal,
)
from buildml.causal.features import (
    design_matrix,
    outcome_array,
    partition_frame,
    validate_columns_present,
)
from buildml.causal.results import CausalEvalResult, CausalPlan
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan


def evaluate_causal(
    dataset: Dataset,
    plan: CausalPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "validation",
    bootstrap_samples: int | None = None,
) -> CausalEvalResult:
    """Evaluate nuisance predictive quality and ATE on a holdout partition.

    Notes
    -----
    Holdout rows are **never** used to refit nuisances. Metrics describe
    predictive calibration of outcome / propensity models and the estimator
    applied out-of-sample; they do **not** prove identification. Causal
    claims still rest entirely on the declared :class:`CausalAssumptions`.
    """
    if plan is None:
        raise ValidationError("No CausalPlan. Call fit_causal first.")

    resolved = partition
    if (
        partition == "validation"
        and split_plan is not None
        and not split_plan.validation_indices
    ):
        resolved = "test"

    frame = partition_frame(dataset, split_plan, resolved)
    validate_columns_present(frame, plan.assumptions)

    control, treated = plan.treatment_levels
    t = np.where(frame[plan.treatment_column].to_numpy() == treated, 1, 0).astype(int)
    y = outcome_array(frame[plan.outcome_column], kind=plan.outcome_kind)
    x = design_matrix(frame, plan.confounder_columns)

    metrics: dict[str, float] = {}
    warnings: list[str] = []
    disclosures = [
        f"Causal evaluate on partition={resolved!r} with train-fitted nuisances.",
        "Holdout metrics are predictive checks — not proof of unconfoundedness.",
        "Identification remains the caller-declared CausalAssumptions "
        "(EDA is associational and does not identify effects).",
    ]

    if plan.mu0_ is not None and plan.mu1_ is not None:
        mu0_hat = _predict_outcome(plan.mu0_, x)
        mu1_hat = _predict_outcome(plan.mu1_, x)
        y_hat = np.where(t == 1, mu1_hat, mu0_hat)
        if plan.outcome_kind == "continuous":
            metrics["outcome_rmse"] = float(np.sqrt(mean_squared_error(y, y_hat)))
            metrics["outcome_r2"] = float(r2_score(y, y_hat))
        else:
            y_hat_cls = (y_hat >= 0.5).astype(int)
            metrics["outcome_accuracy"] = float(accuracy_score(y.astype(int), y_hat_cls))
            try:
                metrics["outcome_brier"] = float(brier_score_loss(y.astype(int), y_hat))
            except ValueError:
                warnings.append("Could not compute outcome Brier score on holdout.")

    if plan.propensity_ is not None:
        e_hat = np.clip(
            _predict_propensity(plan.propensity_, x),
            plan.clip_propensity[0],
            plan.clip_propensity[1],
        )
        metrics["propensity_mean"] = float(np.mean(e_hat))
        metrics["propensity_min"] = float(np.min(e_hat))
        metrics["propensity_max"] = float(np.max(e_hat))
        try:
            metrics["propensity_auc"] = float(roc_auc_score(t, e_hat))
        except ValueError:
            warnings.append("Propensity AUC undefined (single class in partition).")
        try:
            metrics["propensity_brier"] = float(brier_score_loss(t, e_hat))
        except ValueError:
            warnings.append("Propensity Brier undefined on this partition.")
        if metrics["propensity_min"] <= plan.clip_propensity[0] + 1e-12:
            warnings.append("Propensity hit clip floor on holdout — overlap concern.")
        if metrics["propensity_max"] >= plan.clip_propensity[1] - 1e-12:
            warnings.append("Propensity hit clip ceiling on holdout — overlap concern.")

    ate, extras = estimate_ate_from_models(
        x,
        t,
        y,
        method=plan.method,
        mu0=plan.mu0_,
        mu1=plan.mu1_,
        propensity=plan.propensity_,
        clip_propensity=plan.clip_propensity,
    )
    for key, value in extras.items():
        metrics[key] = float(value)

    est = estimate_causal(
        dataset,
        plan,
        split_plan,
        partition=resolved,
        bootstrap_samples=bootstrap_samples,
    )
    # Prefer estimate_causal's bootstrap fields; keep point ate consistent.
    ate = est.ate

    return CausalEvalResult(
        partition=resolved,
        method=plan.method,
        estimand=plan.assumptions.estimand,
        n_rows=int(len(frame)),
        ate=float(ate),
        ate_std=est.ate_std,
        ate_ci_low=est.ate_ci_low,
        ate_ci_high=est.ate_ci_high,
        metrics=metrics,
        disclosures=tuple(disclosures + list(est.disclosures)),
        warnings=tuple(warnings + list(est.warnings)),
    )
