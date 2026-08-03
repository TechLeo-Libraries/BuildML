"""History / catalog / walkthrough helpers for causal operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a causal fit result.

    Records backend, method, ATE point estimate, and bootstrap CI metadata for
    Session audit logs without embedding nuisance models.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.causal.results.CausalFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Backend, method, estimand, ATE, and CI summaries.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
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
    """Build a compact history summary from a causal estimate result.

    Extracts partition, method, and ATE metadata for Session audit logs
    without serialising full nuisance models or bootstrap draw arrays.

    Parameters
    ----------
    estimate_result:
        :class:`~buildml.causal.results.CausalEstimateResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, ATE, and CI summaries for history logs.
    """
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
    """Build a compact history summary from a causal evaluation result.

    Captures holdout partition metrics and point ATE for walkthrough panels
    while omitting large nested metric dict copies when the input is ``None``.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.causal.results.CausalEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, ATE, and policy/value metrics for history logs.
    """
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
    """Build a compact history summary from a causal refutation result.

    Records refutation kind, original vs perturbed ATE, and shift magnitude
    for sensitivity disclosures attached to Session history.

    Parameters
    ----------
    refute_result:
        :class:`~buildml.causal.results.CausalRefuteResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Refutation kind, original ATE, refuted ATE, and shift summary.
    """
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
    """Build a compact summary from declared causal assumptions.

    Surfaces treatment/outcome/confounder columns and acknowledgement flags
    for teaching overlays without re-running :meth:`CausalAssumptions.validate`.

    Parameters
    ----------
    assumptions:
        :class:`~buildml.causal.types.CausalAssumptions` or ``None``.

    Returns
    -------
    dict[str, Any]
        Treatment, outcome, confounders, estimand, and acknowledgement flags.
    """
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
    """Build factual walkthrough disclosure for causal Session state.

    Combines assumptions, plan metadata, result summaries, and
    :func:`~buildml.causal.catalog.causal_capability_matrix` for teaching overlays.

    Parameters
    ----------
    plan:
        Active :class:`~buildml.causal.results.CausalPlan`, if any.
    assumptions:
        Declared :class:`~buildml.causal.types.CausalAssumptions`, if any.
    fit_result, estimate_result, eval_result, refute_result:
        Last operation reports attached to the Session.
    history:
        Session operation records.

    Returns
    -------
    dict[str, Any]
        Enabled flags, ATE metadata, capability matrix, disclosures, and boundaries.
    """
    from buildml.causal.catalog import causal_capability_matrix

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
                f"CausalPlan backend={getattr(plan, 'backend', 'native')}, "
                f"method={getattr(plan, 'method', None)}, "
                f"ATE={getattr(plan, 'ate', None)}, "
                f"treatment={getattr(plan, 'treatment_column', None)}, "
                f"outcome={getattr(plan, 'outcome_column', None)}.",
                "Nuisance models fitted on Session train only. "
                "Validation/test are evaluation / estimate scoring only.",
                "Session checkpoints do not embed CausalPlan; use "
                "save_causal_bundle / load_causal_bundle.",
                "Honesty: backdoor ATE under caller-declared CausalAssumptions "
                "— not causal discovery; EDA stays associational.",
                "Industry backends (DoWhy/EconML) require buildml[causal-industry].",
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
        "backend": None if plan is None else getattr(plan, "backend", "native"),
        "method": None if plan is None else getattr(plan, "method", None),
        "ate": None if plan is None else getattr(plan, "ate", None),
        "capability_matrix": causal_capability_matrix(),
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
    """Report causal status for a Session walkthrough panel.

    Reads causal plan, assumptions, and result slots without mutating Session.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`causal_status` for the Session's causal state.
    """
    return causal_status(
        getattr(session, "_causal_plan", None),
        assumptions=getattr(session, "_causal_assumptions", None),
        fit_result=getattr(session, "_causal_fit_result", None),
        estimate_result=getattr(session, "_causal_estimate_result", None),
        eval_result=getattr(session, "_causal_eval_result", None),
        refute_result=getattr(session, "_causal_refute_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
