"""Causal sensitivity / refutation: native disclosures + DoWhy suite when installed."""

from __future__ import annotations

import numpy as np

from buildml.causal.catalog import list_refute_kinds
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
    """Run a refutation / sensitivity check on the train partition.

    Perturbs treatment labels or confounders and refits (native/EconML) or
    invokes DoWhy refuters to surface instability relative to the original ATE.
    Results are sensitivity disclosures: not proof of identification.

    Native backend
    --------------
    placebo_treatment, random_confounder: sklearn refit disclosures.

    DoWhy backend (``buildml[causal-industry]``)
    --------------------------------------------
    placebo_treatment, random_common_cause, add_unobserved_common_cause,
    data_subset, placebo_outcome: full DoWhy refutation suite.

    Honesty: refutation is a sensitivity disclosure: not proof of
    identification. EDA never substitutes for CausalAssumptions.

    Parameters
    ----------
    dataset:
        Session dataset containing the train partition.
    plan:
        :class:`~buildml.causal.results.CausalPlan` from :func:`fit_causal`.
    split_plan:
        Split plan with train indices.
    kind:
        Refutation kind supported by the plan's backend.
    random_state:
        RNG seed for placebo shuffles and random confounder noise.

    Returns
    -------
    CausalRefuteResult
        Original vs refuted ATE, shift magnitude, and teaching disclosures.

    Raises
    ------
    ValidationError
        When ``plan`` is missing, ``kind`` is unsupported, or a native
        placebo shuffle produces a degenerate treatment arm.
    """
    if plan is None:
        raise ValidationError("No CausalPlan. Call fit_causal first.")

    backend = str(getattr(plan, "backend", "native") or "native")
    kind_key = str(kind).lower().replace("-", "_")
    allowed = list_refute_kinds(backend=backend)  # type: ignore[arg-type]
    if kind_key not in allowed:
        raise ValidationError(
            f"Unknown refute kind={kind!r} for backend={backend!r}. "
            f"Supported: {allowed}."
        )

    if backend == "dowhy":
        from buildml.causal.adapters.dowhy import refute_dowhy

        return refute_dowhy(plan, kind=kind_key, random_state=random_state)

    if backend == "econml":
        return _refute_econml(
            dataset,
            plan,
            split_plan,
            kind_key=kind_key,
            random_state=random_state,
        )

    return _refute_native(
        dataset,
        plan,
        split_plan,
        kind_key=kind_key,
        random_state=random_state,
    )


def _refute_native(
    dataset: Dataset,
    plan: CausalPlan,
    split_plan: SplitPlan | None,
    *,
    kind_key: str,
    random_state: int | None,
) -> CausalRefuteResult:
    """Native placebo / random-confounder refutation."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

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
        "Native sensitivity disclosure: use backend='dowhy' for the full "
        "DoWhy refutation suite when buildml[causal-industry] is installed.",
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
            "Placebo ATE is not near zero relative to the original estimate: "
            "treat the effect estimate with caution."
        )

    return CausalRefuteResult(
        kind=kind_key,
        method=plan.method,
        backend="native",
        original_ate=original,
        refute_ate=float(refute_ate),
        ate_shift=shift,
        n_rows=int(len(train)),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _refute_econml(
    dataset: Dataset,
    plan: CausalPlan,
    split_plan: SplitPlan | None,
    *,
    kind_key: str,
    random_state: int | None,
) -> CausalRefuteResult:
    """EconML placebo / random-confounder refutation via refit."""
    from buildml.causal.adapters.econml import _build_first_stage_models
    from buildml.causal.extras import require_econml

    require_econml(feature="EconML refutation")
    from econml.dml import CausalForestDML, LinearDML
    from econml.policy import PolicyTree

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    train = train_partition_frame(dataset, split_plan)
    validate_columns_present(train, plan.assumptions)
    t, _, _ = encode_binary_treatment(train[plan.treatment_column])
    y = outcome_array(train[plan.outcome_column], kind=plan.outcome_kind)
    x = design_matrix(train, plan.confounder_columns)
    rng = np.random.default_rng(random_state)
    original = float(plan.ate)
    disclosures = [
        f"EconML refutation kind={kind_key} on Session train "
        f"(original ATE={original:.6g}).",
        "EconML refit sensitivity disclosure: not proof of identification.",
    ]
    warnings: list[str] = []

    method = plan.method
    if kind_key == "placebo_treatment":
        t_ref = rng.permutation(t)
        x_ref = x
        disclosures.append("Placebo: shuffled treatment labels and refit EconML.")
    else:
        t_ref = t
        noise = rng.normal(size=(x.shape[0], 1))
        x_ref = np.concatenate([x, noise], axis=1)
        disclosures.append("Random confounder: appended noise column and refit EconML.")

    model_y, model_t = _build_first_stage_models(plan.outcome_kind, random_state)
    if method == "causal_forest":
        estimator = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            random_state=random_state,
            n_estimators=100,
        )
    elif method == "policy_tree":
        estimator = PolicyTree(random_state=random_state)
    else:
        estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            random_state=random_state,
        )

    if method == "policy_tree":
        estimator.fit(y, t_ref, X=x_ref)
        treated = float(np.mean(y[t_ref == 1])) if int(t_ref.sum()) else 0.0
        control = float(np.mean(y[t_ref == 0])) if int((1 - t_ref).sum()) else 0.0
        refute_ate = treated - control
    else:
        estimator.fit(y, t_ref, X=x_ref)
        refute_ate = float(np.asarray(estimator.ate(x_ref)).reshape(-1)[0])

    return CausalRefuteResult(
        kind=kind_key,
        method=plan.method,
        backend="econml",
        original_ate=original,
        refute_ate=float(refute_ate),
        ate_shift=float(refute_ate) - original,
        n_rows=int(len(train)),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
