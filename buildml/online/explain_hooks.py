"""History / catalog / walkthrough helpers for online-learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_online`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "estimator_name": payload.get("estimator_name"),
        "backend": payload.get("backend"),
        "task": payload.get("task"),
        "n_init_rows": payload.get("n_init_rows"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_remaining_train": payload.get("n_remaining_train"),
        "target_column": payload.get("target_column"),
        "used_refit_fallback": payload.get("used_refit_fallback"),
    }


def update_result_summary(update_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``partial_fit_online`` history."""
    if update_result is None:
        return {}
    payload = (
        update_result.to_dict() if hasattr(update_result, "to_dict") else dict(update_result)
    )
    return {
        "estimator_name": payload.get("estimator_name"),
        "n_chunk_rows": payload.get("n_chunk_rows"),
        "n_seen_rows": payload.get("n_seen_rows"),
        "n_updates": payload.get("n_updates"),
        "n_remaining_train": payload.get("n_remaining_train"),
        "update_mode": payload.get("update_mode"),
        "used_refit_fallback": payload.get("used_refit_fallback"),
        "drift_notes": payload.get("drift_notes"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_online`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_seen_rows": payload.get("n_seen_rows"),
        "n_updates": payload.get("n_updates"),
        "metrics": payload.get("metrics"),
        "drift_detected": payload.get("drift_detected"),
        "drift_notes": payload.get("drift_notes"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``predict_online`` history."""
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "partition": payload.get("partition"),
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
    }


def online_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    update_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for online / continual learning."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_online",
            "partial_fit_online",
            "evaluate_online",
            "predict_online",
            "save_online_bundle",
            "load_online_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"OnlinePlan backend={getattr(plan, 'backend', None)}, "
                f"estimator={getattr(plan, 'estimator_name', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"n_seen_rows={getattr(plan, 'n_seen_rows', None)}, "
                f"n_updates={getattr(plan, 'n_updates', None)}, "
                f"cursor={getattr(plan, 'cursor', None)}.",
                "Updates use partial_fit on train chunks (sklearn, River, or torch "
                "continual) or role-aligned external frames. Validation/test are "
                "never used for updates.",
                "Silent full refits are refused unless allow_refit_fallback was "
                "explicitly enabled (always disclosed).",
                "Session checkpoints do not embed OnlinePlan; use "
                "save_online_bundle / load_online_bundle.",
                "Honesty: batch/stream-chunk Session updates — not a distributed "
                "streaming platform or lifelong-learning research suite.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Online-learning operations appear in history, but no live "
            "OnlinePlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last online eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_rows={eval_payload.get('n_rows')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    update_payload = None
    if update_result is not None:
        update_payload = (
            update_result.to_dict()
            if hasattr(update_result, "to_dict")
            else dict(update_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_online_plan": enabled,
        "backend": None if plan is None else getattr(plan, "backend", None),
        "estimator_name": None if plan is None else getattr(plan, "estimator_name", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "n_seen_rows": None if plan is None else getattr(plan, "n_seen_rows", None),
        "n_updates": None if plan is None else getattr(plan, "n_updates", None),
        "cursor": None if plan is None else getattr(plan, "cursor", None),
        "used_refit_fallback": (
            None if plan is None else getattr(plan, "used_refit_fallback", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_update_result": update_result is not None,
        "has_eval_result": eval_result is not None,
        "last_update": update_payload,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Online / continual learning performs incremental partial_fit updates "
            "on Session train chunks. Holdout is evaluation-only. Not a streaming "
            "platform; not causal; not federated."
        ),
    }


def online_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return online_status(
        getattr(session, "_online_plan", None),
        fit_result=getattr(session, "_online_fit_result", None),
        update_result=getattr(session, "_online_update_result", None),
        eval_result=getattr(session, "_online_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
