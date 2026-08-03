"""History / catalog / walkthrough helpers for decision operations."""

from __future__ import annotations

from typing import Any

from buildml.optimize.catalog import decision_capability_matrix


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a decision fit result.

    Records method, backend, tuning partition, threshold, and allocation
    metrics for Session audit logs without embedding full plan objects.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.optimize.results.DecisionFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, partition, threshold, expected cost, and selection summaries.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "backend": payload.get("backend"),
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "threshold": payload.get("threshold"),
        "recommendation_basis": payload.get("recommendation_basis"),
        "expected_cost": payload.get("expected_cost"),
        "n_selected": payload.get("n_selected"),
        "selected_value": payload.get("selected_value"),
        "selected_cost": payload.get("selected_cost"),
        "allow_test_tuning": payload.get("allow_test_tuning"),
    }


def apply_result_summary(apply_result: Any) -> dict[str, Any]:
    """Build a compact history summary from an apply-decisions result.

    Captures method, partition, selection counts, and aggregate allocation
    totals for Session walkthrough logs.

    Parameters
    ----------
    apply_result:
        :class:`~buildml.optimize.results.ApplyDecisionsResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, partition, row/selection counts, and value/cost totals.
    """
    if apply_result is None:
        return {}
    payload = (
        apply_result.to_dict() if hasattr(apply_result, "to_dict") else dict(apply_result)
    )
    return {
        "method": payload.get("method"),
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "n_selected": payload.get("n_selected"),
        "threshold": payload.get("threshold"),
        "selected_value": payload.get("selected_value"),
        "selected_cost": payload.get("selected_cost"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a decision evaluation result.

    Records partition, method, headline metrics, and realized cost for Session
    audit trails after holdout evaluation.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.optimize.results.DecisionEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, method, metrics dict, and realized-cost summary.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "realized_cost": payload.get("realized_cost"),
        "n_selected": payload.get("n_selected"),
    }


def decision_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    apply_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return factual walkthrough disclosure for decision helpers.

    Summarizes whether a live :class:`~buildml.optimize.results.DecisionPlan`
    is attached, recent history operations, capability matrix, and boundary
    notes distinguishing decision helpers from general OR platforms.

    Parameters
    ----------
    plan:
        Current :class:`~buildml.optimize.results.DecisionPlan`, if any.
    fit_result:
        Latest fit summary object, when present on the session.
    eval_result:
        Latest evaluation summary object, when present.
    apply_result:
        Latest apply summary object, when present.
    history:
        Session history records used to detect prior decision operations.

    Returns
    -------
    dict[str, Any]
        Enabled/present flags, capability matrix, method metadata, eval
        payload, disclosures, and platform boundary text.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_decision_policy",
            "apply_decisions",
            "evaluate_decisions",
            "save_decision_bundle",
            "load_decision_bundle",
            "tune_threshold",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"DecisionPlan method={getattr(plan, 'method', None)}, "
                f"fitted_on={getattr(plan, 'partition_fitted', None)}, "
                f"threshold={getattr(plan, 'threshold', None)}.",
                "Session checkpoints do not embed DecisionPlan; use "
                "save_decision_bundle / load_decision_bundle.",
                "Cross-link: Session.tune_threshold remains the diagnostic "
                "threshold explorer; fit_decision_policy(method='threshold') "
                "persists the chosen operating point.",
                "Not a general OR / digital-twin platform.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Decision operations appear in history, but no live DecisionPlan "
            "is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last decision eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_decision_plan": enabled,
        "capability_matrix": decision_capability_matrix(),
        "method": None if plan is None else getattr(plan, "method", None),
        "partition_fitted": (
            None if plan is None else getattr(plan, "partition_fitted", None)
        ),
        "threshold": None if plan is None else getattr(plan, "threshold", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_apply_result": apply_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Optimisation / decision helpers are a Session domain path: "
            "cost-sensitive thresholds, multiclass cost matrices, top-K / "
            "knapsack / LP allocation over ML scores. Native scipy/numpy "
            "fallback; PuLP/OR-Tools MIP knapsack, CVXPY LP, and XGB "
            "cost-sensitive thresholds via buildml[optimize-industry]. "
            "Not a general operations-research platform. tune_threshold "
            "remains available as the classical diagnostic sweep."
        ),
    }


def decision_status_for_session(session: Any) -> dict[str, Any]:
    """Return decision walkthrough status from a Session object's private state.

    Reads attached decision plan, fit/eval/apply results, and history from
    ``session`` and delegates to :func:`decision_status`.

    Parameters
    ----------
    session:
        BuildML :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`decision_status` for the session's current
        decision state.
    """
    return decision_status(
        getattr(session, "_decision_plan", None),
        fit_result=getattr(session, "_decision_fit_result", None),
        eval_result=getattr(session, "_decision_eval_result", None),
        apply_result=getattr(session, "_decision_apply_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
