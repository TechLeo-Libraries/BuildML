"""History / catalog / walkthrough helpers for forecasting operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a forecast fit result.

    Records method, column contract, train row counts, and horizon metadata
    for Session audit logs without embedding fitted estimators.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.forecasting.results.ForecastFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, columns, row counts, lags, and univariate flag summaries.
    """
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
    """Build a compact history summary from a forecast generate result.

    Extracts method, horizon, origin label, and prediction count for Session
    walkthrough panels without serialising full prediction arrays.

    Parameters
    ----------
    generate_result:
        :class:`~buildml.forecasting.results.ForecastGenerateResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, horizon, origin, and prediction-count summaries.
    """
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
    """Build a compact history summary from a forecast evaluation result.

    Captures partition, strategy, point counts, and headline metrics for
    history logs while omitting full prediction vectors when the input is ``None``.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.forecasting.results.ForecastEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, strategy, scored-point count, and metrics summaries.
    """
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
    """Build factual walkthrough disclosure for classical forecasting.

    Merges optional-backend availability, live plan metadata, and recent eval or
    generate summaries for capability-matrix teaching overlays.

    Parameters
    ----------
    plan:
        Active :class:`~buildml.forecasting.results.ForecastPlan`, if any.
    fit_result:
        Optional last fit summary attached to the Session.
    eval_result:
        Optional last holdout evaluation summary.
    generate_result:
        Optional last horizon generation summary.
    history:
        Session operation history used to detect forecasting activity without a
        live plan.

    Returns
    -------
    dict[str, Any]
        Enabled flags, backend availability, disclosures, and eval/generate
        payloads for walkthrough UI.
    """
    from buildml.forecasting.catalog import forecast_status_payload

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
    base = forecast_status_payload()
    disclosures: list[str] = list(base.get("disclosures", []))
    if enabled:
        disclosures.extend(
            [
                f"ForecastPlan method={getattr(plan, 'method', None)}, "
                f"backend={getattr(plan, 'backend', None)}, "
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

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
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
        "backends": {
            "statsmodels": base.get("statsmodels_available"),
            "prophet": base.get("prophet_available"),
            "neuralforecast": base.get("neuralforecast_available"),
        },
        "default_method": base.get("default_method"),
        "eval": eval_payload,
        "generate": generate_payload,
        "disclosures": disclosures,
        "boundary": (
            "Classical forecasting is a Session domain path distinct from "
            "supervised Session.fit on shuffled rows and from digital-twin claims."
        ),
    },
        "forecast_capability_matrix",
    )


def forecasting_status_for_session(session: Any) -> dict[str, Any]:
    """Build forecasting walkthrough status from a Session instance.

    Reads attached forecast plan, fit/eval/generate results, and operation
    history from standard Session private attributes.

    Parameters
    ----------
    session:
        BuildML Session object with optional forecasting state attached.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`forecasting_status` for the given Session.
    """
    return forecasting_status(
        getattr(session, "_forecast_plan", None),
        fit_result=getattr(session, "_forecast_fit_result", None),
        eval_result=getattr(session, "_forecast_eval_result", None),
        generate_result=getattr(session, "_forecast_generate_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
