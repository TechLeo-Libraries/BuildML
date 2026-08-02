"""Fit causal nuisance models and estimate ATE (train-only fit)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from buildml.causal.estimate import estimate_ate_from_models
from buildml.causal.features import (
    design_matrix,
    encode_binary_treatment,
    infer_outcome_kind,
    outcome_array,
    propensity_clip_bounds,
    train_partition_frame,
    validate_columns_present,
)
from buildml.causal.results import CausalFitResult, CausalPlan
from buildml.causal.types import (
    CausalAssumptions,
    CausalConfig,
    CausalMethod,
)
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition


def fit_causal(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    assumptions: CausalAssumptions,
    *,
    method: CausalMethod = "aipw",
    bootstrap_samples: int = 200,
    random_state: int | None = 0,
    clip_propensity: tuple[float, float] = (0.01, 0.99),
    outcome_model: str = "ridge",
    propensity_model: str = "logistic_regression",
) -> tuple[CausalPlan, CausalFitResult]:
    """Fit nuisance models on Session train and estimate backdoor ATE.

    Honesty
    -------
    Requires a validated :class:`CausalAssumptions` declaration
    (treatment, outcome, confounders, estimand, unconfoundedness and
    positivity acknowledgements). EDA / association diagnostics never
    satisfy those acknowledgements. Nuisance models fit on **train only**;
    validation/test are never used for fitting. Bootstrap uncertainty
    resamples the train partition. This is not causal discovery and not a
    DoWhy/EconML product surface.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    assumptions.validate()

    method_key = str(method).lower().replace("-", "_")
    if method_key not in {"t_learner", "ipw", "aipw"}:
        raise ValidationError(
            f"Unknown causal method={method!r}. "
            "Supported: t_learner, ipw, aipw."
        )
    clip = propensity_clip_bounds(clip_propensity)
    if bootstrap_samples < 0:
        raise ValidationError("bootstrap_samples must be >= 0.")

    train = train_partition_frame(dataset, split_plan)
    validate_columns_present(train, assumptions)

    t_codes, levels, t_disc = encode_binary_treatment(train[assumptions.treatment])
    outcome_kind = infer_outcome_kind(train[assumptions.outcome])
    y = outcome_array(train[assumptions.outcome], kind=outcome_kind)
    x = design_matrix(train, assumptions.confounders)

    n = int(len(train))
    n_treated = int(t_codes.sum())
    n_control = n - n_treated
    if n_treated < 5 or n_control < 5:
        raise ValidationError(
            f"Need at least 5 treated and 5 control train rows; "
            f"found treated={n_treated}, control={n_control}."
        )

    disclosures: list[str] = list(t_disc)
    warnings: list[str] = []
    disclosures.extend(
        [
            "CausalAssumptions declared by the caller; EDA associations are "
            "not used as identification evidence.",
            f"Estimand={assumptions.estimand} under {assumptions.identification} "
            f"adjustment with confounders={list(assumptions.confounders)}.",
            "Caller acknowledged unconfoundedness and positivity.",
            "Nuisance models fitted on Session train only; "
            "validation/test never used for fit.",
            f"method={method_key}; outcome_kind={outcome_kind}.",
        ]
    )
    if not assumptions.confounders:
        warnings.append(
            "Empty confounders with allow_empty_confounders=True: "
            "ATE reduces to a simple mean difference under a no-confounding "
            "assumption — extremely strong."
        )
        disclosures.append(warnings[-1])

    mu0, mu1, propensity = _fit_nuisance_models(
        x,
        t_codes,
        y,
        method=method_key,
        outcome_kind=outcome_kind,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        random_state=random_state,
    )

    ate, _ = estimate_ate_from_models(
        x,
        t_codes,
        y,
        method=method_key,
        mu0=mu0,
        mu1=mu1,
        propensity=propensity,
        clip_propensity=clip,
    )

    ate_std: float | None = None
    ate_ci_low: float | None = None
    ate_ci_high: float | None = None
    if bootstrap_samples > 0:
        boots = _bootstrap_ate(
            x,
            t_codes,
            y,
            method=method_key,
            outcome_kind=outcome_kind,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            clip_propensity=clip,
            n_boot=bootstrap_samples,
            random_state=random_state,
        )
        ate_std = float(np.std(boots, ddof=1)) if len(boots) > 1 else None
        ate_ci_low = float(np.quantile(boots, 0.025))
        ate_ci_high = float(np.quantile(boots, 0.975))
        disclosures.append(
            f"Train bootstrap ATE CI from {bootstrap_samples} resamples "
            f"(percentile 2.5/97.5): [{ate_ci_low:.6g}, {ate_ci_high:.6g}]."
        )
    else:
        disclosures.append(
            "bootstrap_samples=0: point ATE only; no bootstrap uncertainty."
        )

    config = CausalConfig(
        method=method_key,  # type: ignore[arg-type]
        bootstrap_samples=int(bootstrap_samples),
        random_state=random_state,
        clip_propensity=clip,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
    )
    plan = CausalPlan(
        method=method_key,
        assumptions=assumptions,
        treatment_column=assumptions.treatment,
        outcome_column=assumptions.outcome,
        confounder_columns=tuple(assumptions.confounders),
        outcome_kind=outcome_kind,
        treatment_levels=levels,
        n_train_rows=n,
        n_treated=n_treated,
        n_control=n_control,
        ate=float(ate),
        ate_std=ate_std,
        ate_ci_low=ate_ci_low,
        ate_ci_high=ate_ci_high,
        bootstrap_samples=int(bootstrap_samples),
        clip_propensity=clip,
        outcome_model_name=outcome_model,
        propensity_model_name=propensity_model,
        mu0_=mu0,
        mu1_=mu1,
        propensity_=propensity,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = CausalFitResult(
        method=method_key,
        estimand=assumptions.estimand,
        identification=assumptions.identification,
        treatment_column=assumptions.treatment,
        outcome_column=assumptions.outcome,
        confounder_columns=tuple(assumptions.confounders),
        n_train_rows=n,
        n_treated=n_treated,
        n_control=n_control,
        ate=float(ate),
        ate_std=ate_std,
        ate_ci_low=ate_ci_low,
        ate_ci_high=ate_ci_high,
        bootstrap_samples=int(bootstrap_samples),
        outcome_kind=outcome_kind,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _fit_nuisance_models(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    outcome_kind: str,
    outcome_model: str,
    propensity_model: str,
    random_state: int | None,
) -> tuple[Any | None, Any | None, Any | None]:
    need_outcome = method in {"t_learner", "aipw"}
    need_propensity = method in {"ipw", "aipw"}

    mu0 = mu1 = propensity = None
    if need_outcome:
        mu0 = _build_outcome_estimator(outcome_kind, outcome_model, random_state)
        mu1 = _build_outcome_estimator(outcome_kind, outcome_model, random_state)
        control = t == 0
        treated = t == 1
        if int(control.sum()) < 2 or int(treated.sum()) < 2:
            raise ValidationError(
                "T-learner / AIPW need ≥2 rows in each treatment arm on train."
            )
        mu0.fit(x[control], y[control])
        mu1.fit(x[treated], y[treated])
    if need_propensity:
        propensity = _build_propensity_estimator(propensity_model, random_state)
        # Ensure both classes present (already checked counts).
        propensity.fit(x, t)
    return mu0, mu1, propensity


def _build_outcome_estimator(
    outcome_kind: str,
    name: str,
    random_state: int | None,
) -> Any:
    key = str(name).lower().replace("-", "_")
    if outcome_kind == "binary":
        if key not in {"logistic_regression", "logreg"}:
            # Default binary outcome model.
            key = "logistic_regression"
        return make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )
    if key not in {"ridge", "linear_regression"}:
        key = "ridge"
    if key == "linear_regression":
        from sklearn.linear_model import LinearRegression

        return make_pipeline(StandardScaler(with_mean=True), LinearRegression())
    return make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0))


def _build_propensity_estimator(name: str, random_state: int | None) -> Any:
    key = str(name).lower().replace("-", "_")
    if key not in {"logistic_regression", "logreg"}:
        key = "logistic_regression"
    return make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )


def _bootstrap_ate(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    outcome_kind: str,
    outcome_model: str,
    propensity_model: str,
    clip_propensity: tuple[float, float],
    n_boot: int,
    random_state: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    estimates: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb, tb, yb = x[idx], t[idx], y[idx]
        if int(tb.sum()) < 2 or int((1 - tb).sum()) < 2:
            continue
        try:
            mu0, mu1, propensity = _fit_nuisance_models(
                xb,
                tb,
                yb,
                method=method,
                outcome_kind=outcome_kind,
                outcome_model=outcome_model,
                propensity_model=propensity_model,
                random_state=random_state,
            )
            ate_b, _ = estimate_ate_from_models(
                xb,
                tb,
                yb,
                method=method,
                mu0=mu0,
                mu1=mu1,
                propensity=propensity,
                clip_propensity=clip_propensity,
            )
        except Exception:  # noqa: BLE001 — skip degenerate bootstrap draws
            continue
        estimates.append(float(ate_b))
    if len(estimates) < max(10, n_boot // 10):
        raise ValidationError(
            "Bootstrap failed: too few valid resamples "
            f"({len(estimates)}/{n_boot}). Grow the train set or reduce "
            "bootstrap_samples."
        )
    return np.asarray(estimates, dtype=float)
