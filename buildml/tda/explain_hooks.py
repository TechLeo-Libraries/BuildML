"""History / catalog / walkthrough helpers for TDA operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_tda`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "vectorization": payload.get("vectorization"),
        "n_train_rows": payload.get("n_train_rows"),
        "feature_dim": payload.get("feature_dim"),
        "knn": payload.get("knn"),
        "head": payload.get("head"),
        "task": payload.get("task"),
        "train_score": payload.get("train_score"),
    }


def transform_result_summary(transform_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``transform_tda`` history."""
    if transform_result is None:
        return {}
    payload = (
        transform_result.to_dict()
        if hasattr(transform_result, "to_dict")
        else dict(transform_result)
    )
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "feature_dim": payload.get("feature_dim"),
        "vectorization": payload.get("vectorization"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``predict_tda`` history."""
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "task": payload.get("task"),
        "n_predictions": payload.get("n_predictions"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_tda`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "vectorization": payload.get("vectorization"),
    }


def tda_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    transform_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for TDA."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_tda",
            "transform_tda",
            "predict_tda",
            "evaluate_tda",
            "save_tda_bundle",
            "load_tda_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"TdaPlan vectorization={getattr(plan, 'vectorization', None)}, "
                f"knn={getattr(plan, 'knn', None)}, "
                f"feature_dim={getattr(plan, 'feature_dim', None)}, "
                f"head={getattr(plan, 'head', None)}.",
                "Session checkpoints do not embed TdaPlan; use "
                "save_tda_bundle / load_tda_bundle.",
                "Requires buildml[tda] (ripser + persim). import buildml stays light.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "TDA operations appear in history, but no live TdaPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last TDA eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_tda_plan": enabled,
        "vectorization": None if plan is None else getattr(plan, "vectorization", None),
        "knn": None if plan is None else getattr(plan, "knn", None),
        "feature_dim": None if plan is None else getattr(plan, "feature_dim", None),
        "head": None if plan is None else getattr(plan, "head", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_transform_result": transform_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "TDA is a Session domain path: persistent homology + vectorization "
            "→ optional sklearn head. Not a Mapper research suite."
        ),
    }


def tda_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return tda_status(
        getattr(session, "_tda_plan", None),
        fit_result=getattr(session, "_tda_fit_result", None),
        eval_result=getattr(session, "_tda_eval_result", None),
        transform_result=getattr(session, "_tda_transform_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
