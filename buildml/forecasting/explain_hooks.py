"""History / catalog / walkthrough helpers for forecasting operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_forecast`` history."""
    if fit_result is None:
        return {}
    if hasattr(fit_result, "to_dict"):
        payload = fit_result.to_dict()
    else:
        payload = dict(fit_result)
    return {
        "method": payload.get("method"),
        "target_column": payload.get("target_column"),
        "time_column": payload.get("time_column"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_fit_rows": payload.get("n_fit_rows"),
        "horizon": payload.get("horizon"),
        "lags": payload.get("lags"),
        "univariate": payload.get("univariate"),
    }


def generate_result_summary(generate_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``generate_forecast`` history."""
    if generate_result is None:
        return {}
    if hasattr(generate_result, "to_dict"):
        payload = generate_result.to_dict()
    else:
        payload = dict(generate_result)
    return {
        "method": payload.get("method"),
        "horizon": payload.get("horizon"),
        "origin": payload.get("origin"),
        "n_predictions": payload.get("n_predictions"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_forecast`` history."""
    if eval_result is None:
        return {}
    if hasattr(eval_result, "to_dict"):
        payload = eval_result.to_dict()
    else:
        payload = dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "strategy": payload.get("strategy"),
        "n_points": payload.get("n_points"),
        "metrics": payload.get("metrics"),
    }


def forecasting_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    generate_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for classical forecasting."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_forecast",
            "generate_forecast",
            "evaluate_forecast",
            "save_forecast_bundle",
            "load_forecast_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"ForecastPlan method={getattr(plan, 'method', None)}, "
                f"horizon={getattr(plan, 'horizon', None)}, "
                f"univariate={getattr(plan, 'univariate', None)}.",
                "Session checkpoints do not embed ForecastPlan; use "
                "save_forecast_bundle / load_forecast_bundle.",
                "Forecasting refuses shuffled random/stratified/group splits; "
                "prefer time_split.",
                "Not a full econometrics suite and not a digital twin.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Forecasting operations appear in history, but no live ForecastPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last forecast eval: "
            f"partition={eval_payload.get('partition')}, "
            f"strategy={eval_payload.get('strategy')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    generate_payload = None
    if generate_result is not None:
        generate_payload = (
            generate_result.to_dict()
            if hasattr(generate_result, "to_dict")
            else dict(generate_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_forecast_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "horizon": None if plan is None else getattr(plan, "horizon", None),
        "univariate": None if plan is None else getattr(plan, "univariate", None),
        "target_column": None if plan is None else getattr(plan, "target_column", None),
        "time_column": None if plan is None else getattr(plan, "time_column", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_generate_result": generate_result is not None,
        "eval": eval_payload,
        "generate": generate_payload,
        "disclosures": disclosures,
        "boundary": (
            "Classical forecasting is a Session domain path distinct from "
            "supervised Session.fit on shuffled rows and from digital-twin claims."
        ),
    }


def forecasting_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return forecasting_status(
        getattr(session, "_forecast_plan", None),
        fit_result=getattr(session, "_forecast_fit_result", None),
        eval_result=getattr(session, "_forecast_eval_result", None),
        generate_result=getattr(session, "_forecast_generate_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
