"""DoWhy adapter: causal graph, identification, industry refutation suite."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from buildml.causal.extras import require_dowhy
from buildml.causal.features import (
    encode_binary_treatment,
    infer_outcome_kind,
    partition_frame,
    train_partition_frame,
    validate_columns_present,
)
from buildml.causal.results import CausalFitResult, CausalPlan, CausalRefuteResult
from buildml.causal.types import CausalAssumptions, CausalConfig
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition

logger = logging.getLogger(__name__)

DOWHY_METHOD_MAP = {
    "backdoor_linear": "backdoor.linear_regression",
    "backdoor_propensity_score": "backdoor.propensity_score_matching",
    "backdoor_propensity_weighting": "backdoor.propensity_score_weighting",
}

DOWHY_REFUTE_MAP = {
    "placebo_treatment": "placebo_treatment_refuter",
    "random_confounder": "random_common_cause",
    "random_common_cause": "random_common_cause",
    "add_unobserved_common_cause": "add_unobserved_common_cause",
    "data_subset": "data_subset_refuter",
    "placebo_outcome": "placebo_outcome_refuter",
}


def _build_graph(
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...],
) -> str:
    lines = ["digraph {"]
    for conf in confounders:
        lines.append(f'  {conf} -> {treatment};')
        lines.append(f'  {conf} -> {outcome};')
    lines.append(f'  {treatment} -> {outcome};')
    lines.append("}")
    return "\n".join(lines)


def fit_dowhy(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    assumptions: CausalAssumptions,
    *,
    method: str = "backdoor_linear",
    random_state: int | None = 0,
) -> tuple[CausalPlan, CausalFitResult]:
    """Fit DoWhy backdoor ATE on Session train only.

    Builds a causal graph from declared confounders (not discovered), runs
    DoWhy identification and backdoor estimation, and stores the model artifact
    on the returned :class:`~buildml.causal.results.CausalPlan` for refutation.

    Parameters
    ----------
    dataset:
        Session dataset containing treatment, outcome, and confounders.
    split_plan:
        Split plan with train indices.
    assumptions:
        Caller-declared backdoor identification contract.
    method:
        DoWhy backdoor method key (e.g. ``backdoor_linear``).
    random_state:
        RNG seed forwarded to DoWhy estimation when supported.

    Returns
    -------
    tuple[CausalPlan, CausalFitResult]
        Persistable plan with DoWhy artifact and train ATE summary.

    Raises
    ------
    ValidationError
        When assumptions fail validation, train arms are too small, or
        ``method`` is unsupported. Also raised when DoWhy extras are missing
        (via :func:`require_dowhy`).
    """
    require_dowhy(feature="DoWhy causal backend")
    from dowhy import CausalModel

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    assumptions.validate()

    method_key = str(method).lower().replace("-", "_")
    if method_key not in DOWHY_METHOD_MAP:
        raise ValidationError(
            f"Unknown DoWhy method={method!r}. Supported: "
            f"{sorted(DOWHY_METHOD_MAP)}."
        )

    train = train_partition_frame(dataset, split_plan)
    validate_columns_present(train, assumptions)
    t_codes, levels, t_disc = encode_binary_treatment(train[assumptions.treatment])
    outcome_kind = infer_outcome_kind(train[assumptions.outcome])

    cols = [assumptions.treatment, assumptions.outcome, *assumptions.confounders]
    df = train[cols].copy()
    df[assumptions.treatment] = t_codes.astype(int)
    n = int(len(df))
    n_treated = int(t_codes.sum())
    n_control = n - n_treated
    if n_treated < 5 or n_control < 5:
        raise ValidationError(
            f"Need at least 5 treated and 5 control train rows; "
            f"found treated={n_treated}, control={n_control}."
        )

    graph = _build_graph(
        assumptions.treatment,
        assumptions.outcome,
        assumptions.confounders,
    )
    model = CausalModel(
        data=df,
        treatment=assumptions.treatment,
        outcome=assumptions.outcome,
        common_causes=list(assumptions.confounders) or None,
        graph=graph,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name=DOWHY_METHOD_MAP[method_key],
        method_params={"random_state": random_state} if random_state is not None else None,
    )
    ate = float(estimate.value)
    ci = getattr(estimate, "get_confidence_intervals", lambda: None)()
    ate_ci_low = ate_ci_high = None
    if ci is not None:
        try:
            ci_arr = np.asarray(ci, dtype=float).reshape(-1)
            if len(ci_arr) >= 2:
                ate_ci_low = float(ci_arr[0])
                ate_ci_high = float(ci_arr[1])
        except (TypeError, ValueError):
            # CI payload shape/dtype unexpected; leave bounds unset rather than fail the estimate.
            logger.debug(
                "dowhy: could not parse ATE confidence interval payload",
                exc_info=True,
            )

    disclosures: list[str] = list(t_disc)
    disclosures.extend(
        [
            "CausalAssumptions declared by the caller; EDA associations are "
            "not used as identification evidence.",
            f"DoWhy backend method={method_key}; graph built from declared "
            f"confounders (not discovered).",
            f"Estimand={assumptions.estimand} under {assumptions.identification} "
            f"adjustment with confounders={list(assumptions.confounders)}.",
            "Caller acknowledged unconfoundedness and positivity.",
            "DoWhy identify_effect + estimate_effect on Session train only.",
            "Use refute_causal with DoWhy refuters for sensitivity checks.",
        ]
    )
    warnings: list[str] = []
    if not assumptions.confounders:
        warnings.append(
            "Empty confounders with allow_empty_confounders=True: "
            "graph has treatment→outcome only — extremely strong assumption."
        )
        disclosures.append(warnings[-1])

    artifact = {
        "model": model,
        "identified_estimand": identified,
        "estimate": estimate,
        "graph": graph,
        "train_df": df,
    }
    config = CausalConfig(
        method=method_key,  # type: ignore[arg-type]
        bootstrap_samples=0,
        random_state=random_state,
    )
    plan = CausalPlan(
        method=method_key,
        backend="dowhy",
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
        ate_std=None,
        ate_ci_low=ate_ci_low,
        ate_ci_high=ate_ci_high,
        bootstrap_samples=0,
        clip_propensity=(0.01, 0.99),
        outcome_model_name="dowhy",
        propensity_model_name="dowhy",
        backend_artifact_=artifact,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config={**config.to_dict(), "backend": "dowhy", "dowhy_method": method_key},
    )
    result = CausalFitResult(
        method=method_key,
        backend="dowhy",
        estimand=assumptions.estimand,
        identification=assumptions.identification,
        treatment_column=assumptions.treatment,
        outcome_column=assumptions.outcome,
        confounder_columns=tuple(assumptions.confounders),
        n_train_rows=n,
        n_treated=n_treated,
        n_control=n_control,
        ate=float(ate),
        ate_std=None,
        ate_ci_low=ate_ci_low,
        ate_ci_high=ate_ci_high,
        bootstrap_samples=0,
        outcome_kind=outcome_kind,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def estimate_dowhy_partition(
    plan: CausalPlan,
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    partition: str,
    random_state: int | None = 0,
) -> tuple[float, dict[str, float]]:
    """Re-estimate DoWhy ATE on a holdout partition (honest partition effect).

    Rebuilds the declared causal graph and runs DoWhy identification plus
    backdoor estimation on the requested partition rows only. Train-fitted
    nuisances are **not** reused — this is a fresh estimate on holdout data.

    Parameters
    ----------
    plan:
        Train-fitted :class:`~buildml.causal.results.CausalPlan` with
        ``backend='dowhy'``.
    dataset:
        Session dataset containing the evaluation partition.
    split_plan:
        Split plan defining partition indices.
    partition:
        Holdout partition name such as ``validation`` or ``test``.
    random_state:
        RNG seed forwarded to DoWhy when supported.

    Returns
    -------
    float
        Partition-level average treatment effect from DoWhy.
    dict[str, float]
        Extra scalar metrics (empty when none are produced).

    Raises
    ------
    ValidationError
        When the plan lacks DoWhy metadata or the partition is too small.
    """
    require_dowhy(feature="DoWhy holdout evaluation")
    from dowhy import CausalModel

    method_key = str(plan.method).lower().replace("-", "_")
    if method_key not in DOWHY_METHOD_MAP:
        raise ValidationError(
            f"Unknown DoWhy method={plan.method!r} on plan; cannot evaluate holdout."
        )

    assumptions = plan.assumptions
    frame = partition_frame(dataset, split_plan, partition)
    validate_columns_present(frame, assumptions)
    t_codes, _, _ = encode_binary_treatment(frame[assumptions.treatment])
    cols = [assumptions.treatment, assumptions.outcome, *assumptions.confounders]
    df = frame[cols].copy()
    df[assumptions.treatment] = t_codes.astype(int)
    n = int(len(df))
    n_treated = int(t_codes.sum())
    n_control = n - n_treated
    if n_treated < 5 or n_control < 5:
        raise ValidationError(
            f"DoWhy holdout evaluate needs ≥5 treated and ≥5 control rows on "
            f"partition={partition!r}; found treated={n_treated}, control={n_control}."
        )

    graph = _build_graph(
        assumptions.treatment,
        assumptions.outcome,
        assumptions.confounders,
    )
    model = CausalModel(
        data=df,
        treatment=assumptions.treatment,
        outcome=assumptions.outcome,
        common_causes=list(assumptions.confounders) or None,
        graph=graph,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name=DOWHY_METHOD_MAP[method_key],
        method_params={"random_state": random_state} if random_state is not None else None,
    )
    return float(estimate.value), {}


def refute_dowhy(
    plan: CausalPlan,
    *,
    kind: str,
    random_state: int | None = 0,
) -> CausalRefuteResult:
    """Run a DoWhy refutation on the stored train estimate.

    Reuses the DoWhy model, identified estimand, and point estimate captured
    during :func:`fit_dowhy` to execute an industry refutation method and
    report the perturbed ATE relative to the original train estimate.

    Parameters
    ----------
    plan:
        :class:`~buildml.causal.results.CausalPlan` fitted with
        ``backend='dowhy'``.
    kind:
        Refutation kind mapped to a DoWhy refuter name.
    random_state:
        RNG seed forwarded to the DoWhy refuter when supported.

    Returns
    -------
    CausalRefuteResult
        Original vs refuted ATE, optional p-value, and disclosures.

    Raises
    ------
    ValidationError
        When the plan lacks a DoWhy artifact, ``kind`` is unknown, or DoWhy
        extras are missing.
    """
    require_dowhy(feature="DoWhy refutation suite")
    artifact = getattr(plan, "backend_artifact_", None)
    if not isinstance(artifact, dict):
        raise ValidationError(
            "DoWhy refutation requires a plan fitted with backend='dowhy'."
        )
    kind_key = str(kind).lower().replace("-", "_")
    refute_method = DOWHY_REFUTE_MAP.get(kind_key)
    if refute_method is None:
        from buildml.causal.catalog import list_refute_kinds

        raise ValidationError(
            f"Unknown DoWhy refute kind={kind!r}. Supported: "
            f"{list_refute_kinds(backend='dowhy')}."
        )

    model = artifact["model"]
    identified = artifact["identified_estimand"]
    estimate = artifact["estimate"]
    original = float(plan.ate)

    refute_kwargs: dict[str, Any] = {}
    if random_state is not None:
        refute_kwargs["random_state"] = random_state
    refutation = model.refute_estimate(
        identified,
        estimate,
        method_name=refute_method,
        **refute_kwargs,
    )
    refute_ate = float(getattr(refutation, "new_effect", getattr(refutation, "estimated_effect", 0.0)))
    p_value = getattr(refutation, "refutation_result", None)
    refute_p_value: float | None = None
    if p_value is not None and hasattr(p_value, "p_value"):
        try:
            refute_p_value = float(p_value.p_value)
        except (TypeError, ValueError):
            refute_p_value = None
    elif hasattr(refutation, "p_value"):
        try:
            refute_p_value = float(refutation.p_value)
        except (TypeError, ValueError):
            refute_p_value = None

    disclosures = [
        f"DoWhy refutation kind={kind_key} (method={refute_method}).",
        f"Original train ATE={original:.6g}; refute estimate={refute_ate:.6g}.",
        "DoWhy refutation is a sensitivity disclosure — not proof of identification.",
        "EDA / association paths never substitute for CausalAssumptions.",
    ]
    warnings: list[str] = []
    if refute_p_value is not None and refute_p_value < 0.05:
        warnings.append(
            f"DoWhy refutation p-value={refute_p_value:.4g} — treat the original "
            "estimate with caution."
        )

    return CausalRefuteResult(
        kind=kind_key,
        method=plan.method,
        backend=plan.backend,
        original_ate=original,
        refute_ate=refute_ate,
        ate_shift=float(refute_ate) - original,
        n_rows=int(plan.n_train_rows),
        refute_p_value=refute_p_value,
        refute_details={"dowhy_method": refute_method},
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
