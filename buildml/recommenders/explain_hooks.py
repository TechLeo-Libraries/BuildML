"""History / catalog / walkthrough helpers for recommender operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_recommender`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "n_train_interactions": payload.get("n_train_interactions"),
        "n_users": payload.get("n_users"),
        "n_items": payload.get("n_items"),
        "feedback": payload.get("feedback"),
        "n_neighbors": payload.get("n_neighbors"),
        "n_factors": payload.get("n_factors"),
    }


def recommend_result_summary(recommend_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``recommend`` history."""
    if recommend_result is None:
        return {}
    payload = (
        recommend_result.to_dict()
        if hasattr(recommend_result, "to_dict")
        else dict(recommend_result)
    )
    return {
        "k": payload.get("k"),
        "n_users": payload.get("n_users"),
        "method": payload.get("method"),
        "n_cold_start_users": payload.get("n_cold_start_users"),
        "n_recommendations": payload.get("n_recommendations"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_recommender`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "k": payload.get("k"),
        "n_users_scored": payload.get("n_users_scored"),
        "metrics": payload.get("metrics"),
    }


def recommender_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    recommend_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for recommenders."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_recommender",
            "recommend",
            "evaluate_recommender",
            "save_recommender_bundle",
            "load_recommender_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"RecommenderPlan method={getattr(plan, 'method', None)}, "
                f"users={getattr(plan, 'n_users', None)}, "
                f"items={getattr(plan, 'n_items', None)}, "
                f"feedback={getattr(plan, 'feedback', None)}.",
                "Session checkpoints do not embed RecommenderPlan; use "
                "save_recommender_bundle / load_recommender_bundle.",
                "Recommenders are not RAG and not EDA Recommendation Findings.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Recommender operations appear in history, but no live "
            "RecommenderPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last recommender eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_recommender_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "n_users": None if plan is None else getattr(plan, "n_users", None),
        "n_items": None if plan is None else getattr(plan, "n_items", None),
        "feedback": None if plan is None else getattr(plan, "feedback", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_recommend_result": recommend_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Recommenders are a Session domain path: user/item interactions → "
            "CF or content top-K with ranking metrics. Not RAG, not EDA "
            "Recommendation Findings, not a Netflix-scale platform."
        ),
    }


def recommender_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return recommender_status(
        getattr(session, "_recommender_plan", None),
        fit_result=getattr(session, "_recommender_fit_result", None),
        eval_result=getattr(session, "_recommender_eval_result", None),
        recommend_result=getattr(session, "_recommender_recommend_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
