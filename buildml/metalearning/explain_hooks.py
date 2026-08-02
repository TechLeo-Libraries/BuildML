"""History / catalog / walkthrough helpers for meta-learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_metalearning`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
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
    """Compact result_summary for ``evaluate_metalearning`` history."""
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
    """Compact result_summary for ``adapt_to_task`` history."""
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
    """Factual walkthrough disclosure for meta-learning."""
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
                f"MetaLearningPlan method={getattr(plan, 'method', None)}, "
                f"task_column={getattr(plan, 'task_column', None)}, "
                f"n_meta_train_tasks={len(getattr(plan, 'train_task_ids', ()) or ())}, "
                f"n_way={getattr(plan, 'n_way', None)}, "
                f"k_shot={getattr(plan, 'k_shot', None)}.",
                "Meta-train uses train partition only; validation/test are "
                "evaluation-only.",
                "Session checkpoints do not embed MetaLearningPlan; use "
                "save_metalearning_bundle / load_metalearning_bundle.",
                "Honesty: tabular few-shot / episodic protocol — not "
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

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_metalearning_plan": enabled,
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
            "protocols (prototypical nearest-centroid and warm-start adapt). "
            "Holdout is evaluation-only. Not foundation-model meta-learning; "
            "not MAML-at-scale; not causal; not federated."
        ),
    }


def metalearning_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return metalearning_status(
        getattr(session, "_metalearning_plan", None),
        fit_result=getattr(session, "_metalearning_fit_result", None),
        eval_result=getattr(session, "_metalearning_eval_result", None),
        adapt_result=getattr(session, "_metalearning_adapt_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
