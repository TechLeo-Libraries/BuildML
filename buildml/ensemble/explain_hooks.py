"""History / catalog / walkthrough helpers for ensemble operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ensemble fit history."""
    if fit_result is None:
        return {}
    if hasattr(fit_result, "to_dict"):
        payload = fit_result.to_dict()
    else:
        payload = dict(fit_result)
    return {
        "strategy": payload.get("strategy"),
        "task": payload.get("task"),
        "estimator_names": payload.get("estimator_names"),
        "n_train_rows": payload.get("n_train_rows"),
        "final_estimator_name": payload.get("final_estimator_name"),
        "voting": payload.get("voting"),
        "cv": payload.get("cv"),
        "holdout_fraction": payload.get("holdout_fraction"),
        "blend_method": payload.get("blend_method"),
    }


def ensemble_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for native ensembles."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_voting",
            "fit_stacking",
            "fit_blending",
            "save_ensemble_bundle",
            "load_ensemble_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"EnsemblePlan strategy={getattr(plan, 'strategy', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"bases={list(getattr(plan, 'estimator_names', ()) or ())}.",
                "Session checkpoints do not embed EnsemblePlan; use "
                "save_ensemble_bundle / load_ensemble_bundle (or save_pipeline for "
                "preprocess + estimator).",
                "Stacking / blending meta-learners are fit inside train only; "
                "Session test is never used for meta features.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Ensemble operations appear in history, but no live EnsemblePlan is attached."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_ensemble_plan": enabled,
        "strategy": None if plan is None else getattr(plan, "strategy", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "estimator_names": (
            None if plan is None else list(getattr(plan, "estimator_names", ()) or ())
        ),
        "has_fit_result": fit_result is not None,
        "disclosures": disclosures,
        "boundary": (
            "Native ensembles are a Session domain path distinct from passing a single "
            "RandomForest to Session.fit."
        ),
    }


def ensemble_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return ensemble_status(
        getattr(session, "_ensemble_plan", None),
        fit_result=getattr(session, "_fit_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
