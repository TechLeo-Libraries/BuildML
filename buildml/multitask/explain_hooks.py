"""History / catalog / walkthrough helpers for multi-task operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_multitask`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "task": payload.get("task"),
        "n_train_rows": payload.get("n_train_rows"),
        "target_columns": payload.get("target_columns"),
        "n_tasks": payload.get("n_tasks"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_multitask`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "per_task_metrics": payload.get("per_task_metrics"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``predict_multitask`` history."""
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_tasks": payload.get("n_tasks"),
        "attached": payload.get("attached"),
        "prediction_prefix": payload.get("prediction_prefix"),
    }


def multitask_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    predict_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for multi-task learning."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_multitask",
            "predict_multitask",
            "evaluate_multitask",
            "save_multitask_bundle",
            "load_multitask_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"MultiTaskPlan method={getattr(plan, 'method', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"n_tasks={len(getattr(plan, 'target_columns', ()) or ())}, "
                f"targets={list(getattr(plan, 'target_columns', ()) or ())}.",
                "Fit uses sklearn MultiOutput / Chain on train only; "
                "validation/test are evaluation-only.",
                "Same-type tasks only (all classification or all regression); "
                "mixed targets are refused.",
                "Classical Session.fit remains single-target.",
                "Session checkpoints do not embed MultiTaskPlan; use "
                "save_multitask_bundle / load_multitask_bundle.",
                "Honesty: shared-feature multi-output — not a deep MTL "
                "research platform.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Multi-task operations appear in history, but no live "
            "MultiTaskPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last multi-task eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_rows={eval_payload.get('n_rows')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_multitask_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "target_columns": (
            None
            if plan is None
            else list(getattr(plan, "target_columns", ()) or ())
        ),
        "n_tasks": (
            None
            if plan is None
            else len(getattr(plan, "target_columns", ()) or ())
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_predict_result": predict_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Multi-task uses sklearn MultiOutput/Chain on shared features with "
            "multiple same-type targets. Holdout is evaluation-only. Not deep "
            "MTL; not causal; not federated."
        ),
    }


def multitask_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return multitask_status(
        getattr(session, "_multitask_plan", None),
        fit_result=getattr(session, "_multitask_fit_result", None),
        eval_result=getattr(session, "_multitask_eval_result", None),
        predict_result=getattr(session, "_multitask_predict_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
