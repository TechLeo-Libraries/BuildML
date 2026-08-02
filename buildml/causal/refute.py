"""Simple causal sensitivity / placebo disclosures (not a full DoWhy suite)."""

from __future__ import annotations

import numpy as np

from buildml.causal.estimate import estimate_ate_from_models
from buildml.causal.features import (
    design_matrix,
    encode_binary_treatment,
    outcome_array,
    train_partition_frame,
    validate_columns_present,
)
from buildml.causal.fit import _fit_nuisance_models
from buildml.causal.results import CausalPlan, CausalRefuteResult
from buildml.causal.types import CausalRefuteKind
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition


def refute_causal(
    dataset: Dataset,
    plan: CausalPlan,
    split_plan: SplitPlan | None,
    *,
    kind: CausalRefuteKind = "placebo_treatment",
    random_state: int | None = 0,
) -> CausalRefuteResult:
    """Run a simple refutation / sensitivity check on the train partition.

    Supported kinds
    ---------------
    placebo_treatment:
        Shuffle treatment labels on train, refit nuisances, re-estimate ATE.
        Under a well-specified identified effect, the placebo ATE should move
        toward zero — this is a disclosure, not a formal proof.
    random_confounder:
        Append a pure-noise covariate, refit, and report the ATE shift.

    Honesty: this is **not** a full DoWhy refutation suite (placebo outcome,
    data subsets, unobserved confounder simulation, etc.).
    """
    if plan is None:
        raise ValidationError("No CausalPlan. Call fit_causal first.")
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    kind_key = str(kind).lower().replace("-", "_")
    if kind_key not in {"placebo_treatment", "random_confounder"}:
        raise ValidationError(
            f"Unknown refute kind={kind!r}. "
            "Supported: placebo_treatment, random_confounder."
        )

    train = train_partition_frame(dataset, split_plan)
    validate_columns_present(train, plan.assumptions)
    t, _, _ = encode_binary_treatment(train[plan.treatment_column])
    y = outcome_array(train[plan.outcome_column], kind=plan.outcome_kind)
    x = design_matrix(train, plan.confounder_columns)
    rng = np.random.default_rng(random_state)

    original = float(plan.ate)
    disclosures = [
        f"Refutation kind={kind_key} on Session train "
        f"(original ATE={original:.6g}).",
        "Simple sensitivity disclosure — not a complete DoWhy refutation suite.",
        "A passing placebo does not prove identification; a failing one "
        "is a warning to revisit assumptions / overlap / specification.",
    ]
    warnings: list[str] = []

    if kind_key == "placebo_treatment":
        t_placebo = rng.permutation(t)
        if int(t_placebo.sum()) < 2 or int((1 - t_placebo).sum()) < 2:
            raise ValidationError("Placebo shuffle produced a degenerate treatment arm.")
        mu0, mu1, propensity = _fit_nuisance_models(
            x,
            t_placebo,
            y,
            method=plan.method,
            outcome_kind=plan.outcome_kind,
            outcome_model=plan.outcome_model_name,
            propensity_model=plan.propensity_model_name,
            random_state=random_state,
        )
        refute_ate, _ = estimate_ate_from_models(
            x,
            t_placebo,
            y,
            method=plan.method,
            mu0=mu0,
            mu1=mu1,
            propensity=propensity,
            clip_propensity=plan.clip_propensity,
        )
        disclosures.append(
            "Placebo treatment: shuffled train treatment labels and refit "
            "nuisances; expect refute_ate near 0 if the original signal was "
            "not an artifact of the estimator alone."
        )
    else:
        noise = rng.normal(size=(x.shape[0], 1))
        x_aug = np.concatenate([x, noise], axis=1)
        mu0, mu1, propensity = _fit_nuisance_models(
            x_aug,
            t,
            y,
            method=plan.method,
            outcome_kind=plan.outcome_kind,
            outcome_model=plan.outcome_model_name,
            propensity_model=plan.propensity_model_name,
            random_state=random_state,
        )
        refute_ate, _ = estimate_ate_from_models(
            x_aug,
            t,
            y,
            method=plan.method,
            mu0=mu0,
            mu1=mu1,
            propensity=propensity,
            clip_propensity=plan.clip_propensity,
        )
        disclosures.append(
            "Random confounder: appended N(0,1) noise column and refit; "
            "large ATE shifts may indicate instability / weak overlap."
        )

    shift = float(refute_ate) - original
    if kind_key == "placebo_treatment" and abs(float(refute_ate)) > abs(original) * 0.5 + 0.05:
        warnings.append(
            "Placebo ATE is not near zero relative to the original estimate — "
            "treat the effect estimate with caution."
        )

    return CausalRefuteResult(
        kind=kind_key,
        method=plan.method,
        original_ate=original,
        refute_ate=float(refute_ate),
        ate_shift=shift,
        n_rows=int(len(train)),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
