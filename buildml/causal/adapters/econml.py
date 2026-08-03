"""EconML adapter: DML, CausalForest, policy learning on declared backdoor sets."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.causal.extras import require_econml
from buildml.causal.features import (
    design_matrix,
    encode_binary_treatment,
    infer_outcome_kind,
    outcome_array,
    train_partition_frame,
    validate_columns_present,
)
from buildml.causal.results import CausalFitResult, CausalPlan
from buildml.causal.types import CausalAssumptions, CausalConfig
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition


def _build_first_stage_models(outcome_kind: str, random_state: int | None) -> tuple[Any, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    rs = random_state
    if outcome_kind == "binary":
        model_y = GradientBoostingClassifier(random_state=rs, n_estimators=100)
    else:
        model_y = GradientBoostingRegressor(random_state=rs, n_estimators=100)
    model_t = GradientBoostingClassifier(random_state=rs, n_estimators=100)
    return model_y, model_t


def fit_econml(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    assumptions: CausalAssumptions,
    *,
    method: str = "dml",
    bootstrap_samples: int = 200,
    random_state: int | None = 0,
) -> tuple[CausalPlan, CausalFitResult]:
    """Fit EconML DML / CausalForest / PolicyTree on Session train only.

    Fits a declared-backdoor EconML estimator on the train partition, stores
    the fitted object on the plan artifact, and optionally bootstrap-refits for
    train ATE uncertainty (except ``policy_tree``).

    Parameters
    ----------
    dataset:
        Session dataset containing treatment, outcome, and confounders.
    split_plan:
        Split plan with train indices.
    assumptions:
        Caller-declared backdoor identification contract.
    method:
        ``dml``, ``causal_forest``, or ``policy_tree``.
    bootstrap_samples:
        Number of train bootstrap refits for ATE CI (skipped for policy tree).
    random_state:
        RNG seed for EconML and gradient-boosting first stages.

    Returns
    -------
    tuple[CausalPlan, CausalFitResult]
        Persistable plan with EconML artifact and train ATE summary.

    Raises
    ------
    ValidationError
        When assumptions fail validation, train arms are too small,
        ``method`` is unsupported, or bootstrap resampling fails. Also raised
        when EconML extras are missing (via :func:`require_econml`).
    """
    require_econml(feature="EconML causal backend")
    from econml.dml import CausalForestDML, LinearDML
    from econml.policy import PolicyTree

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    assumptions.validate()

    method_key = str(method).lower().replace("-", "_")
    if method_key not in {"dml", "causal_forest", "policy_tree"}:
        raise ValidationError(
            f"Unknown EconML method={method!r}. Supported: dml, causal_forest, policy_tree."
        )
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

    model_y, model_t = _build_first_stage_models(outcome_kind, random_state)
    estimator: Any
    if method_key == "dml":
        estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            random_state=random_state,
        )
    elif method_key == "causal_forest":
        estimator = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            random_state=random_state,
            n_estimators=100,
        )
    else:
        estimator = PolicyTree(
            random_state=random_state,
        )

    if method_key == "policy_tree":
        estimator.fit(y, t_codes, X=x)
        # PolicyTree reports value; approximate ATE via implied treatment effects.
        policy = estimator.predict(x)
        treated_effect = float(np.mean(y[t_codes == 1])) if n_treated else 0.0
        control_effect = float(np.mean(y[t_codes == 0])) if n_control else 0.0
        ate = treated_effect - control_effect
        cate_std = None
    else:
        estimator.fit(y, t_codes, X=x)
        ate = float(np.asarray(estimator.ate(x)).reshape(-1)[0])
        cate_std = None
        if hasattr(estimator, "effect"):
            effects = np.asarray(estimator.effect(x), dtype=float).reshape(-1)
            if len(effects) > 1:
                cate_std = float(np.std(effects, ddof=1))

    ate_std: float | None = None
    ate_ci_low: float | None = None
    ate_ci_high: float | None = None
    if bootstrap_samples > 0 and method_key != "policy_tree":
        boots = _bootstrap_econml_ate(
            y,
            t_codes,
            x,
            method=method_key,
            outcome_kind=outcome_kind,
            n_boot=bootstrap_samples,
            random_state=random_state,
        )
        ate_std = float(np.std(boots, ddof=1)) if len(boots) > 1 else None
        ate_ci_low = float(np.quantile(boots, 0.025))
        ate_ci_high = float(np.quantile(boots, 0.975))

    disclosures: list[str] = list(t_disc)
    disclosures.extend(
        [
            "CausalAssumptions declared by the caller; EDA associations are "
            "not used as identification evidence.",
            f"EconML backend method={method_key} on declared backdoor confounders.",
            f"Estimand={assumptions.estimand} under {assumptions.identification} "
            f"adjustment with confounders={list(assumptions.confounders)}.",
            "Caller acknowledged unconfoundedness and positivity.",
            "EconML fit on Session train only; validation/test never used for fit.",
            "CATE heterogeneity available via causal_forest; policy_tree learns "
            "assignment rules: not a deployment product.",
        ]
    )
    warnings: list[str] = []
    if not assumptions.confounders:
        warnings.append(
            "Empty confounders with allow_empty_confounders=True: "
            "EconML still runs but unconfoundedness is extremely strong."
        )
        disclosures.append(warnings[-1])
    if bootstrap_samples > 0 and method_key != "policy_tree":
        disclosures.append(
            f"Train bootstrap ATE CI from {bootstrap_samples} EconML refits "
            f"(percentile 2.5/97.5): [{ate_ci_low:.6g}, {ate_ci_high:.6g}]."
        )
    elif method_key == "policy_tree":
        disclosures.append(
            "policy_tree reports a learned assignment rule; ATE shown is a "
            "simple treated-minus-control mean difference for disclosure."
        )

    config = CausalConfig(
        method=method_key,  # type: ignore[arg-type]
        bootstrap_samples=int(bootstrap_samples),
        random_state=random_state,
    )
    artifact = {
        "estimator": estimator,
        "method": method_key,
        "x_train": x,
        "y_train": y,
        "t_train": t_codes,
    }
    plan = CausalPlan(
        method=method_key,
        backend="econml",
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
        clip_propensity=(0.01, 0.99),
        outcome_model_name="econml",
        propensity_model_name="econml",
        backend_artifact_=artifact,
        cate_std=cate_std,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config={**config.to_dict(), "backend": "econml", "econml_method": method_key},
    )
    result = CausalFitResult(
        method=method_key,
        backend="econml",
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
        cate_std=cate_std,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _bootstrap_econml_ate(
    y: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    *,
    method: str,
    outcome_kind: str,
    n_boot: int,
    random_state: int | None,
) -> np.ndarray:
    require_econml()
    from econml.dml import CausalForestDML, LinearDML

    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    estimates: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb, tb, yb = x[idx], t[idx], y[idx]
        if int(tb.sum()) < 2 or int((1 - tb).sum()) < 2:
            continue
        try:
            model_y, model_t = _build_first_stage_models(outcome_kind, random_state)
            if method == "causal_forest":
                est = CausalForestDML(
                    model_y=model_y,
                    model_t=model_t,
                    discrete_treatment=True,
                    random_state=random_state,
                    n_estimators=50,
                )
            else:
                est = LinearDML(
                    model_y=model_y,
                    model_t=model_t,
                    discrete_treatment=True,
                    random_state=random_state,
                )
            est.fit(yb, tb, X=xb)
            estimates.append(float(np.asarray(est.ate(xb)).reshape(-1)[0]))
        except Exception:  # noqa: BLE001
            continue
    if len(estimates) < max(10, n_boot // 10):
        raise ValidationError(
            "EconML bootstrap failed: too few valid resamples "
            f"({len(estimates)}/{n_boot})."
        )
    return np.asarray(estimates, dtype=float)


def score_econml_partition(
    plan: CausalPlan,
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Score ATE on a partition with a fitted EconML estimator.

    Applies the train-fitted EconML object from the plan artifact to holdout
    confounders without refitting, reporting partition-level ATE and optional
    CATE dispersion when the estimator exposes ``effect``.

    Parameters
    ----------
    plan:
        :class:`~buildml.causal.results.CausalPlan` fitted with
        ``backend='econml'``.
    x:
        Confounder design matrix for the evaluation partition.
    t:
        Binary treatment indicators aligned with ``x``.
    y:
        Outcome vector aligned with ``x`` and ``t``.

    Returns
    -------
    tuple[float, dict[str, float]]
        Partition ATE and optional diagnostics (e.g. ``cate_std``).

    Raises
    ------
    ValidationError
        When the plan lacks an EconML artifact.
    """
    artifact = getattr(plan, "backend_artifact_", None)
    if not isinstance(artifact, dict):
        raise ValidationError("EconML scoring requires backend='econml' plan.")
    estimator = artifact["estimator"]
    method = artifact.get("method", plan.method)
    extras: dict[str, float] = {}
    if method == "policy_tree":
        treated = float(np.mean(y[t == 1])) if int(t.sum()) else 0.0
        control = float(np.mean(y[t == 0])) if int((1 - t).sum()) else 0.0
        ate = treated - control
    else:
        ate = float(np.asarray(estimator.ate(x)).reshape(-1)[0])
        if hasattr(estimator, "effect"):
            effects = np.asarray(estimator.effect(x), dtype=float).reshape(-1)
            if len(effects) > 1:
                extras["cate_std"] = float(np.std(effects, ddof=1))
    return ate, extras
