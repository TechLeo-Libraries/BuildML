"""History / catalog / walkthrough helpers for multi-task operations."""
from __future__ import annotations
from typing import Any
from buildml.multitask.catalog import multitask_capability_matrix

def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a multi-task fit result.
    Strips heavy estimator objects so Session history records only the fields
    needed for walkthrough overlays and audit replay.
    Parameters
    ----------
    fit_result:
        :class:`~buildml.multitask.results.MultiTaskFitResult` or compatible
        mapping; ``None`` yields an empty dict.
    Returns
    -------
    dict[str, Any]
        Backend, method, target columns, and train row counts.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "backend": payload.get("backend"),
        "task": payload.get("task"),
        "n_train_rows": payload.get("n_train_rows"),
        "target_columns": payload.get("target_columns"),
        "n_tasks": payload.get("n_tasks"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }

def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a multi-task evaluation result.
    Captures partition-level metrics and per-task scores for explain overlays
    without serializing full prediction blobs.
    Parameters
    ----------
    eval_result:
        :class:`~buildml.multitask.results.MultiTaskEvalResult` or compatible
        mapping; ``None`` yields an empty dict.
    Returns
    -------
    dict[str, Any]
        Partition, row counts, aggregate metrics, and per-task metrics summary.
    """
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
    """Build a compact history payload from a multi-task predict result.
    Records partition, task count, and attach metadata without embedding raw
    prediction vectors in history.
    Parameters
    ----------
    predict_result:
        :class:`~buildml.multitask.results.MultiTaskPredictResult` or compatible
        mapping; ``None`` yields an empty dict.
    Returns
    -------
    dict[str, Any]
        Partition, row counts, attach flag, and prediction prefix.
    """
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
    """Build factual walkthrough disclosure for multi-task learning state.
    Combines live plan fields, latest fit/eval/predict payloads, and history
    evidence into a teaching-oriented status dict with capability matrix
    attachment.
    Parameters
    ----------
    plan:
        Optional :class:`~buildml.multitask.results.MultiTaskPlan`.
    fit_result:
        Optional latest fit result for ``has_fit_result``.
    eval_result:
        Optional latest eval result; metrics are summarized in disclosures.
    predict_result:
        Optional latest predict result for attach/partition disclosures.
    history:
        Session operation history used to detect multi-task activity when no
        plan is attached.
    Returns
    -------
    dict[str, Any]
        Enabled flags, target column summary, disclosures, boundary text, and
        nested capability matrix from
        :func:`buildml.multitask.catalog.multitask_capability_matrix`.
    """
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
                f"MultiTaskPlan backend={getattr(plan, 'backend', None)}, "
                f"method={getattr(plan, 'method', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"n_tasks={len(getattr(plan, 'target_columns', ()) or ())}, "
                f"targets={list(getattr(plan, 'target_columns', ()) or ())}.",
                "Fit uses train only; validation/test are evaluation-only.",
                (
                    "Sklearn/industry backends require same-type targets; torch "
                    "shared_trunk_multihead supports mixed cls+reg."
                ),
                "Classical Session.fit remains single-target.",
                "Session checkpoints do not embed MultiTaskPlan; use "
                "save_multitask_bundle / load_multitask_bundle.",
                "Honesty: shared-feature multi-target — not a deep MTL "
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
        "backend": None if plan is None else getattr(plan, "backend", None),
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
        "capability_matrix": multitask_capability_matrix(),
        "disclosures": disclosures,
        "boundary": (
            "Multi-task uses sklearn MultiOutput/Chain, industry GBDT multi-target, "
            "or torch shared-trunk multi-head on shared features. Holdout is "
            "evaluation-only. Not deep MTL; not causal; not federated."
        ),
    }

def multitask_status_for_session(session: Any) -> dict[str, Any]:
    """Build multi-task walkthrough status from a Session instance.
    Reads private Session attributes set by multi-task operations and delegates
    to :func:`multitask_status`.
    Parameters
    ----------
    session:
        BuildML Session with optional ``_multitask_*`` state attributes.
    Returns
    -------
    dict[str, Any]
        Same payload as :func:`multitask_status` for the session's plan and
        results.
    """
    return multitask_status(
        getattr(session, "_multitask_plan", None),
        fit_result=getattr(session, "_multitask_fit_result", None),
        eval_result=getattr(session, "_multitask_eval_result", None),
        predict_result=getattr(session, "_multitask_predict_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
