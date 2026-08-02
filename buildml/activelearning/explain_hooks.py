"""History / catalog / walkthrough helpers for active-learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_active_learner`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "strategy": payload.get("strategy"),
        "base_estimator": payload.get("base_estimator"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_labeled_train": payload.get("n_labeled_train"),
        "n_unlabeled_pool": payload.get("n_unlabeled_pool"),
        "n_queries_used": payload.get("n_queries_used"),
        "label_budget": payload.get("label_budget"),
        "target_column": payload.get("target_column"),
    }


def query_result_summary(query_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``suggest_query`` history."""
    if query_result is None:
        return {}
    payload = (
        query_result.to_dict() if hasattr(query_result, "to_dict") else dict(query_result)
    )
    return {
        "strategy": payload.get("strategy"),
        "n_suggested": payload.get("n_suggested"),
        "n_unlabeled_pool": payload.get("n_unlabeled_pool"),
        "budget_remaining": payload.get("budget_remaining"),
        "indices": payload.get("indices"),
    }


def label_result_summary(label_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``label_rows`` history."""
    if label_result is None:
        return {}
    payload = (
        label_result.to_dict() if hasattr(label_result, "to_dict") else dict(label_result)
    )
    return {
        "n_newly_labeled": payload.get("n_newly_labeled"),
        "n_labeled_now": payload.get("n_labeled_now"),
        "n_queries_used": payload.get("n_queries_used"),
        "budget_remaining": payload.get("budget_remaining"),
        "refit": payload.get("refit"),
        "indices": payload.get("indices"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_active_learning`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "strategy": payload.get("strategy"),
        "n_labeled_eval": payload.get("n_labeled_eval"),
        "n_unlabeled_eval": payload.get("n_unlabeled_eval"),
        "n_queries_used": payload.get("n_queries_used"),
        "metrics": payload.get("metrics"),
    }


def activelearning_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    query_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for active learning."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_active_learner",
            "suggest_query",
            "label_rows",
            "evaluate_active_learning",
            "save_active_learning_bundle",
            "load_active_learning_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"ActiveLearningPlan strategy={getattr(plan, 'strategy', None)}, "
                f"n_labeled_train={getattr(plan, 'n_labeled_train', None)}, "
                f"n_unlabeled_pool={getattr(plan, 'n_unlabeled_pool', None)}, "
                f"n_queries_used={getattr(plan, 'n_queries_used', None)}, "
                f"label_budget={getattr(plan, 'label_budget', None)}.",
                "Unlabeled pool uses NaN missingness by default (train partition only).",
                "Labels come from the user — core never invents an oracle.",
                "Session checkpoints do not embed ActiveLearningPlan; use "
                "save_active_learning_bundle / load_active_learning_bundle.",
                "Distinct from semi-supervised propagation and self-supervised pretext.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Active-learning operations appear in history, but no live "
            "ActiveLearningPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last active-learning eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_labeled_eval={eval_payload.get('n_labeled_eval')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    query_payload = None
    if query_result is not None:
        query_payload = (
            query_result.to_dict()
            if hasattr(query_result, "to_dict")
            else dict(query_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_activelearning_plan": enabled,
        "strategy": None if plan is None else getattr(plan, "strategy", None),
        "n_labeled_train": None if plan is None else getattr(plan, "n_labeled_train", None),
        "n_unlabeled_pool": (
            None if plan is None else getattr(plan, "n_unlabeled_pool", None)
        ),
        "n_queries_used": None if plan is None else getattr(plan, "n_queries_used", None),
        "label_budget": None if plan is None else getattr(plan, "label_budget", None),
        "has_fit_result": fit_result is not None,
        "has_query_result": query_result is not None,
        "has_eval_result": eval_result is not None,
        "last_query": query_payload,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Active learning is human-in-the-loop labeling on the train pool. "
            "Holdout labels are evaluation-only; the library never queries test "
            "and never invents oracle labels."
        ),
    }


def activelearning_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return activelearning_status(
        getattr(session, "_activelearning_plan", None),
        fit_result=getattr(session, "_activelearning_fit_result", None),
        query_result=getattr(session, "_activelearning_query_result", None),
        eval_result=getattr(session, "_activelearning_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
