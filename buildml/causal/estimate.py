"""ATE estimation from fitted nuisance models (T-learner / IPW / AIPW)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.causal.features import (
    design_matrix,
    outcome_array,
    partition_frame,
    propensity_clip_bounds,
    validate_columns_present,
)
from buildml.causal.results import CausalEstimateResult, CausalPlan
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan


def estimate_ate_from_models(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    mu0: Any | None,
    mu1: Any | None,
    propensity: Any | None,
    clip_propensity: tuple[float, float],
) -> tuple[float, dict[str, float]]:
    """Compute ATE and diagnostic summaries from fitted nuisances."""
    method_key = str(method).lower().replace("-", "_")
    low, high = propensity_clip_bounds(clip_propensity)
    extras: dict[str, float] = {}

    mu0_hat = mu1_hat = None
    e_hat = None
    if mu0 is not None:
        mu0_hat = _predict_outcome(mu0, x)
    if mu1 is not None:
        mu1_hat = _predict_outcome(mu1, x)
    if propensity is not None:
        e_hat = _predict_propensity(propensity, x)
        e_hat = np.clip(e_hat, low, high)
        extras["propensity_mean"] = float(np.mean(e_hat))
        extras["propensity_min"] = float(np.min(e_hat))
        extras["propensity_max"] = float(np.max(e_hat))

    if method_key == "t_learner":
        if mu0_hat is None or mu1_hat is None:
            raise ValidationError("T-learner requires fitted mu0 and mu1.")
        cate = mu1_hat - mu0_hat
        ate = float(np.mean(cate))
        extras["cate_std"] = float(np.std(cate, ddof=1)) if len(cate) > 1 else 0.0
        return ate, extras

    if method_key == "ipw":
        if e_hat is None:
            raise ValidationError("IPW requires a fitted propensity model.")
        scores = t * y / e_hat - (1 - t) * y / (1.0 - e_hat)
        ate = float(np.mean(scores))
        extras["ipw_score_std"] = (
            float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        )
        return ate, extras

    if method_key == "aipw":
        if mu0_hat is None or mu1_hat is None or e_hat is None:
            raise ValidationError("AIPW requires mu0, mu1, and propensity.")
        scores = (
            (mu1_hat - mu0_hat)
            + t * (y - mu1_hat) / e_hat
            - (1 - t) * (y - mu0_hat) / (1.0 - e_hat)
        )
        ate = float(np.mean(scores))
        extras["aipw_score_std"] = (
            float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        )
        return ate, extras

    raise ValidationError(f"Unknown causal method={method!r}.")


def estimate_causal(
    dataset: Dataset,
    plan: CausalPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "train",
    bootstrap_samples: int | None = None,
    random_state: int | None = None,
) -> CausalEstimateResult:
    """Estimate ATE on a partition with the fitted nuisance models.

    Notes
    -----
    Identification assumptions apply to the data-generating process; computing
    the estimator on holdout rows is a sample-splitting / evaluation choice,
    not a substitute for declaring assumptions. Bootstrap (when requested)
    resamples the **evaluation partition rows** while keeping fitted train
    nuisances fixed — a cheaper influence-function-style uncertainty band,
    distinct from the full retrain bootstrap used in ``fit_causal``.
    """
    if plan is None:
        raise ValidationError("No CausalPlan. Call fit_causal first.")
    frame = partition_frame(dataset, split_plan, partition)
    validate_columns_present(frame, plan.assumptions)

    backend = str(getattr(plan, "backend", "native") or "native")

    # Re-encode treatment with the plan's level mapping.
    control, treated = plan.treatment_levels
    if frame[plan.treatment_column].isna().any():
        raise ValidationError("Causal treatment column contains nulls.")
    t = np.where(frame[plan.treatment_column].to_numpy() == treated, 1, 0).astype(int)
    # Guard unexpected levels.
    levels = set(pd_unique_levels(frame[plan.treatment_column]))
    expected = {control, treated}
    if not levels <= expected:
        raise ValidationError(
            f"Partition {partition!r} has treatment levels {sorted(map(str, levels))} "
            f"outside the plan levels {sorted(map(str, expected))}."
        )
    y = outcome_array(frame[plan.outcome_column], kind=plan.outcome_kind)
    x = design_matrix(frame, plan.confounder_columns)

    n = int(len(frame))
    n_treated = int(t.sum())
    n_control = n - n_treated
    if n < 2 or n_treated < 1 or n_control < 1:
        raise ValidationError(
            f"Partition {partition!r} needs both treatment arms "
            f"(n={n}, treated={n_treated}, control={n_control})."
        )

    warnings: list[str] = []
    if backend == "econml":
        from buildml.causal.adapters.econml import score_econml_partition

        ate, extras = score_econml_partition(plan, x, t, y)
    elif backend == "dowhy":
        ate = float(plan.ate)
        extras = {}
        warnings.append(
            "DoWhy backend reports train-identified ATE; partition re-estimation "
            "is not performed — use native/econml for holdout ATE scoring."
        )
    else:
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

    n_boot = plan.bootstrap_samples if bootstrap_samples is None else int(bootstrap_samples)
    seed = plan.config.get("random_state") if random_state is None else random_state
    ate_std = ate_ci_low = ate_ci_high = None
    disclosures = [
        f"ATE estimated on partition={partition!r} "
        f"(backend={backend}, method={plan.method}).",
        "This does not replace CausalAssumptions; identification remains "
        "caller-declared backdoor adjustment.",
        "EDA / association screens are not used here.",
    ]
    if n_boot and n_boot > 0 and backend == "native":
        boots = _bootstrap_scores(
            x,
            t,
            y,
            plan=plan,
            n_boot=n_boot,
            random_state=seed if isinstance(seed, int) or seed is None else int(seed),
        )
        ate_std = float(np.std(boots, ddof=1)) if len(boots) > 1 else None
        ate_ci_low = float(np.quantile(boots, 0.025))
        ate_ci_high = float(np.quantile(boots, 0.975))
        disclosures.append(
            f"Partition bootstrap (fixed nuisances, {n_boot} resamples) "
            f"CI=[{ate_ci_low:.6g}, {ate_ci_high:.6g}]."
        )
    if extras.get("propensity_min") is not None and extras["propensity_min"] <= plan.clip_propensity[0] + 1e-12:
        warnings.append(
            "Some propensity scores hit the clip floor — check positivity / overlap."
        )
    if extras.get("propensity_max") is not None and extras["propensity_max"] >= plan.clip_propensity[1] - 1e-12:
        warnings.append(
            "Some propensity scores hit the clip ceiling — check positivity / overlap."
        )

    return CausalEstimateResult(
        partition=partition,
        method=plan.method,
        estimand=plan.assumptions.estimand,
        n_rows=n,
        n_treated=n_treated,
        n_control=n_control,
        ate=float(ate),
        ate_std=ate_std,
        ate_ci_low=ate_ci_low,
        ate_ci_high=ate_ci_high,
        bootstrap_samples=int(n_boot or 0),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _predict_outcome(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)
        # Positive class column.
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def _predict_propensity(model: Any, x: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValidationError("Propensity model must implement predict_proba.")
    proba = np.asarray(model.predict_proba(x), dtype=float)
    classes = list(getattr(model, "classes_", []))
    # Pipeline: classes_ on final estimator.
    if not classes and hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        classes = list(getattr(final, "classes_", []))
    if len(classes) == 2:
        # Column for class 1.
        try:
            idx = list(classes).index(1)
        except ValueError:
            idx = 1
        return proba[:, idx]
    return proba[:, -1]


def _bootstrap_scores(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    plan: CausalPlan,
    n_boot: int,
    random_state: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    out: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        ate_b, _ = estimate_ate_from_models(
            x[idx],
            t[idx],
            y[idx],
            method=plan.method,
            mu0=plan.mu0_,
            mu1=plan.mu1_,
            propensity=plan.propensity_,
            clip_propensity=plan.clip_propensity,
        )
        out.append(float(ate_b))
    return np.asarray(out, dtype=float)


def pd_unique_levels(series) -> list[Any]:
    import pandas as pd

    return list(pd.unique(series))
