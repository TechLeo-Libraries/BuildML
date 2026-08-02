"""History / catalog / walkthrough helpers for causal operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_causal`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "estimand": payload.get("estimand"),
        "identification": payload.get("identification"),
        "treatment_column": payload.get("treatment_column"),
        "outcome_column": payload.get("outcome_column"),
        "n_train_rows": payload.get("n_train_rows"),
        "ate": payload.get("ate"),
        "ate_ci_low": payload.get("ate_ci_low"),
        "ate_ci_high": payload.get("ate_ci_high"),
        "bootstrap_samples": payload.get("bootstrap_samples"),
    }


def estimate_result_summary(estimate_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``estimate_causal`` history."""
    if estimate_result is None:
        return {}
    payload = (
        estimate_result.to_dict()
        if hasattr(estimate_result, "to_dict")
        else dict(estimate_result)
    )
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "estimand": payload.get("estimand"),
        "n_rows": payload.get("n_rows"),
        "ate": payload.get("ate"),
        "ate_ci_low": payload.get("ate_ci_low"),
        "ate_ci_high": payload.get("ate_ci_high"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_causal`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "estimand": payload.get("estimand"),
        "n_rows": payload.get("n_rows"),
        "ate": payload.get("ate"),
        "metrics": payload.get("metrics"),
    }


def refute_result_summary(refute_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``refute_causal`` history."""
    if refute_result is None:
        return {}
    payload = (
        refute_result.to_dict()
        if hasattr(refute_result, "to_dict")
        else dict(refute_result)
    )
    return {
        "kind": payload.get("kind"),
        "method": payload.get("method"),
        "original_ate": payload.get("original_ate"),
        "refute_ate": payload.get("refute_ate"),
        "ate_shift": payload.get("ate_shift"),
    }


def assumptions_summary(assumptions: Any) -> dict[str, Any]:
    """Compact summary for ``declare_causal_assumptions`` history."""
    if assumptions is None:
        return {}
    payload = (
        assumptions.to_dict() if hasattr(assumptions, "to_dict") else dict(assumptions)
    )
    return {
        "treatment": payload.get("treatment"),
        "outcome": payload.get("outcome"),
        "confounders": payload.get("confounders"),
        "estimand": payload.get("estimand"),
        "identification": payload.get("identification"),
        "acknowledge_unconfoundedness": payload.get("acknowledge_unconfoundedness"),
        "acknowledge_positivity": payload.get("acknowledge_positivity"),
    }


def causal_status(
    plan: Any = None,
    *,
    assumptions: Any = None,
    fit_result: Any = None,
    estimate_result: Any = None,
    eval_result: Any = None,
    refute_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for causal ML."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "declare_causal_assumptions",
            "fit_causal",
            "estimate_causal",
            "evaluate_causal",
            "refute_causal",
            "save_causal_bundle",
            "load_causal_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    has_assumptions = assumptions is not None or (
        plan is not None and getattr(plan, "assumptions", None) is not None
    )
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"CausalPlan method={getattr(plan, 'method', None)}, "
                f"ATE={getattr(plan, 'ate', None)}, "
                f"treatment={getattr(plan, 'treatment_column', None)}, "
                f"outcome={getattr(plan, 'outcome_column', None)}.",
                "Nuisance models fitted on Session train only. "
                "Validation/test are evaluation / estimate scoring only.",
                "Session checkpoints do not embed CausalPlan; use "
                "save_causal_bundle / load_causal_bundle.",
                "Honesty: backdoor ATE under caller-declared CausalAssumptions "
                "— not causal discovery; not DoWhy/EconML; EDA stays associational.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif has_assumptions:
        disclosures.append(
            "CausalAssumptions are declared, but no CausalPlan is fitted yet. "
            "Call fit_causal after acknowledgements are complete."
        )
    elif saw:
        disclosures.append(
            "Causal operations appear in history, but no live CausalPlan "
            "or CausalAssumptions are attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last causal eval: "
            f"partition={eval_payload.get('partition')}, "
            f"ate={eval_payload.get('ate')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    refute_payload = None
    if refute_result is not None:
        refute_payload = (
            refute_result.to_dict()
            if hasattr(refute_result, "to_dict")
            else dict(refute_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw or has_assumptions,
        "has_causal_plan": enabled,
        "has_assumptions": has_assumptions,
        "method": None if plan is None else getattr(plan, "method", None),
        "ate": None if plan is None else getattr(plan, "ate", None),
        "has_fit_result": fit_result is not None,
        "has_estimate_result": estimate_result is not None,
        "has_eval_result": eval_result is not None,
        "has_refute_result": refute_result is not None,
        "eval": eval_payload,
        "refute": refute_payload,
        "disclosures": disclosures,
        "boundary": (
            "Causal ML estimates backdoor ATE only under explicit "
            "CausalAssumptions (treatment, outcome, confounders, estimand, "
            "unconfoundedness + positivity acknowledgements). "
            "EDA / association / importance paths never identify causal effects."
        ),
    }


def causal_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return causal_status(
        getattr(session, "_causal_plan", None),
        assumptions=getattr(session, "_causal_assumptions", None),
        fit_result=getattr(session, "_causal_fit_result", None),
        estimate_result=getattr(session, "_causal_estimate_result", None),
        eval_result=getattr(session, "_causal_eval_result", None),
        refute_result=getattr(session, "_causal_refute_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
