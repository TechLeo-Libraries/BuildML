"""History / catalog / walkthrough helpers for meta-learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a meta-learning fit result.

    Strips heavy estimator objects so Session history records only the fields
    needed for walkthrough overlays and audit replay.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.metalearning.results.MetaLearningFitResult` or
        compatible mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Backend, method, episodic protocol knobs, and meta-train accuracy.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "method": payload.get("method"),
        "n_train_rows": payload.get("n_train_rows"),
        "task_column": payload.get("task_column"),
        "target_column": payload.get("target_column"),
        "n_meta_train_tasks": payload.get("n_meta_train_tasks"),
        "n_held_out_tasks": payload.get("n_held_out_tasks"),
        "n_way": payload.get("n_way"),
        "k_shot": payload.get("k_shot"),
        "meta_train_accuracy": payload.get("meta_train_accuracy"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a meta-learning evaluation result.

    Captures partition-level episodic metrics and task overlap disclosures for
    explain overlays without serializing per-task detail blobs.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.metalearning.results.MetaLearningEvalResult` or
        compatible mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Partition, task counts, aggregate metrics, and overlap flags.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_tasks_evaluated": payload.get("n_tasks_evaluated"),
        "n_query_rows": payload.get("n_query_rows"),
        "metrics": payload.get("metrics"),
        "novel_task_ids": payload.get("novel_task_ids"),
        "overlapping_task_ids": payload.get("overlapping_task_ids"),
    }


def adapt_result_summary(adapt_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a fast-adapt result.

    Records support size and whether an adapted estimator was produced without
    embedding private model weights in history.

    Parameters
    ----------
    adapt_result:
        :class:`~buildml.metalearning.results.MetaAdaptResult` or compatible
        mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Method, task id, support size, and adaptation summary fields.
    """
    if adapt_result is None:
        return {}
    payload = (
        adapt_result.to_dict()
        if hasattr(adapt_result, "to_dict")
        else dict(adapt_result)
    )
    return {
        "method": payload.get("method"),
        "task_id": payload.get("task_id"),
        "n_support": payload.get("n_support"),
        "n_classes_adapted": payload.get("n_classes_adapted"),
        "has_adapted_estimator": payload.get("has_adapted_estimator"),
    }


def metalearning_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    adapt_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for meta-learning state.

    Combines live plan fields, latest fit/eval/adapt payloads, and history
    evidence into a teaching-oriented status dict with capability matrix
    attachment.

    Parameters
    ----------
    plan:
        Optional :class:`~buildml.metalearning.results.MetaLearningPlan`.
    fit_result:
        Optional latest fit result for ``has_fit_result``.
    eval_result:
        Optional latest eval result; metrics are summarized in disclosures.
    adapt_result:
        Optional latest adapt result.
    history:
        Session operation history used to detect meta-learning activity when no
        plan is attached.

    Returns
    -------
    dict[str, Any]
        Enabled flags, episodic protocol summary, disclosures, boundary text,
        and nested capability matrix from
        :func:`buildml.explain.capability_status.attach_capability_matrix`.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_metalearning",
            "adapt_to_task",
            "evaluate_metalearning",
            "save_metalearning_bundle",
            "load_metalearning_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"MetaLearningPlan backend={getattr(plan, 'backend', None)}, "
                f"method={getattr(plan, 'method', None)}, "
                f"task_column={getattr(plan, 'task_column', None)}, "
                f"n_meta_train_tasks={len(getattr(plan, 'train_task_ids', ()) or ())}, "
                f"n_way={getattr(plan, 'n_way', None)}, "
                f"k_shot={getattr(plan, 'k_shot', None)}.",
                f"held_out_task_ids={list(getattr(plan, 'held_out_task_ids', ()) or ())}.",
                "Meta-train uses train partition only; validation/test are "
                "evaluation-only.",
                "Session checkpoints do not embed MetaLearningPlan; use "
                "save_metalearning_bundle / load_metalearning_bundle.",
                "Honesty: tabular few-shot / episodic protocol: not "
                "foundation-model meta-learning or MAML-at-scale.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Meta-learning operations appear in history, but no live "
            "MetaLearningPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict()
            if hasattr(eval_result, "to_dict")
            else dict(eval_result)
        )
        disclosures.append(
            "Last meta-learning eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_tasks_evaluated={eval_payload.get('n_tasks_evaluated')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    adapt_payload = None
    if adapt_result is not None:
        adapt_payload = (
            adapt_result.to_dict()
            if hasattr(adapt_result, "to_dict")
            else dict(adapt_result)
        )

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
        "enabled": enabled,
        "present": enabled or saw,
        "has_metalearning_plan": enabled,
        "backend": None if plan is None else getattr(plan, "backend", None),
        "method": None if plan is None else getattr(plan, "method", None),
        "task_column": None if plan is None else getattr(plan, "task_column", None),
        "n_meta_train_tasks": (
            None
            if plan is None
            else len(getattr(plan, "train_task_ids", ()) or ())
        ),
        "n_way": None if plan is None else getattr(plan, "n_way", None),
        "k_shot": None if plan is None else getattr(plan, "k_shot", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_adapt_result": adapt_result is not None,
        "eval": eval_payload,
        "adapt": adapt_payload,
        "disclosures": disclosures,
        "boundary": (
            "Meta-learning provides practical tabular few-shot / episodic "
            "protocols (sklearn prototypical/warm_start; optional torch "
            "prototypical_torch and industry MAML/Reptile). Holdout is "
            "evaluation-only. Not foundation-model meta-learning; not "
            "MAML-at-scale; not causal; not federated."
        ),
    },
        "metalearning_capability_matrix",
    )


def metalearning_status_for_session(session: Any) -> dict[str, Any]:
    """Build meta-learning walkthrough status from a Session instance.

    Reads private Session attributes set by meta-learning operations and
    delegates to :func:`metalearning_status`.

    Parameters
    ----------
    session:
        BuildML Session with optional ``_metalearning_*`` state attributes.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`metalearning_status` for the session's plan and
        results.
    """
    return metalearning_status(
        getattr(session, "_metalearning_plan", None),
        fit_result=getattr(session, "_metalearning_fit_result", None),
        eval_result=getattr(session, "_metalearning_eval_result", None),
        adapt_result=getattr(session, "_metalearning_adapt_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
