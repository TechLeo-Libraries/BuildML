"""History / catalog / walkthrough helpers for probabilistic operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a probabilistic fit result.

    Strips full plan payloads while recording backend, conformal settings, and
    train carve sizes for Session audit logs.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.probabilistic.results.ProbabilisticFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Backend, estimator, task, alpha, and conformal metadata.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_fit_rows": payload.get("n_fit_rows"),
        "n_conformal_calib_rows": payload.get("n_conformal_calib_rows"),
        "alpha": payload.get("alpha"),
        "conformal": payload.get("conformal"),
        "interval_method": payload.get("interval_method"),
        "conformal_quantile": payload.get("conformal_quantile"),
        "target_column": payload.get("target_column"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a probabilistic evaluation result.

    Records partition metrics and interval coverage without embedding full
    prediction arrays in Session history.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.probabilistic.results.ProbabilisticEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, metrics, interval coverage, and width summaries.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "alpha": payload.get("alpha"),
        "metrics": payload.get("metrics"),
        "interval_coverage": payload.get("interval_coverage"),
        "mean_interval_width": payload.get("mean_interval_width"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a probabilistic predict result.

    Records prediction counts and uncertainty flags without listing every row.

    Parameters
    ----------
    predict_result:
        :class:`~buildml.probabilistic.results.ProbabilisticPredictResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, row counts, and std/probability availability flags.
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
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
        "has_std": payload.get("has_std"),
        "has_probabilities": payload.get("has_probabilities"),
    }


def interval_result_summary(interval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from an interval prediction result.

    Records interval method, alpha, and whether lower/upper bounds or prediction
    sets were produced.

    Parameters
    ----------
    interval_result:
        :class:`~buildml.probabilistic.results.ProbabilisticIntervalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, alpha, method, and interval/set availability flags.
    """
    if interval_result is None:
        return {}
    payload = (
        interval_result.to_dict()
        if hasattr(interval_result, "to_dict")
        else dict(interval_result)
    )
    return {
        "partition": payload.get("partition"),
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "alpha": payload.get("alpha"),
        "method": payload.get("method"),
        "has_lower_upper": payload.get("has_lower_upper"),
        "has_prediction_sets": payload.get("has_prediction_sets"),
    }


def probabilistic_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    interval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for Bayesian / probabilistic ML.

    Combines live plan metadata, optional evaluation summaries, history
    detection, and :func:`~buildml.probabilistic.catalog.probabilistic_capability_matrix`
    for teaching overlays.

    Parameters
    ----------
    plan:
        Active :class:`~buildml.probabilistic.results.ProbabilisticPlan`, if any.
    fit_result:
        Last fit report attached to the Session.
    eval_result:
        Last evaluation result.
    interval_result:
        Last interval or prediction-set result.
    history:
        Session operation records.

    Returns
    -------
    dict[str, Any]
        Enabled flags, backend metadata, embedded capability matrix, disclosures,
        and boundary text separating sklearn uncertainty from MCMC platforms.
    """
    from buildml.probabilistic.catalog import probabilistic_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_probabilistic",
            "evaluate_probabilistic",
            "predict_probabilistic",
            "predict_interval",
            "save_probabilistic_bundle",
            "load_probabilistic_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"ProbabilisticPlan backend={getattr(plan, 'backend', 'native')}, "
                f"estimator={getattr(plan, 'estimator_name', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"alpha={getattr(plan, 'alpha', None)}, "
                f"conformal={getattr(plan, 'conformal', None)}, "
                f"interval_method={getattr(plan, 'interval_method', None)}.",
                "Fit and optional split-conformal calibration use Session train "
                "only. Validation/test are evaluation / interval scoring only.",
                "Session.calibration() remains the classical FitResult diagnostic "
                "and is not replaced by this path; evaluate_probabilistic reports "
                "NLL/Brier/ECE/CRPS for probabilistic plans.",
                "Session checkpoints do not embed ProbabilisticPlan; use "
                "save_probabilistic_bundle / load_probabilistic_bundle.",
                "Honesty: native sklearn + optional MAPIE/NGBoost industry "
                "backends — not a PyMC/Stan MCMC platform.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Probabilistic operations appear in history, but no live "
            "ProbabilisticPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last probabilistic eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_rows={eval_payload.get('n_rows')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    interval_payload = None
    if interval_result is not None:
        interval_payload = (
            interval_result.to_dict()
            if hasattr(interval_result, "to_dict")
            else dict(interval_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_probabilistic_plan": enabled,
        "backend": None if plan is None else getattr(plan, "backend", "native"),
        "estimator_name": None if plan is None else getattr(plan, "estimator_name", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "alpha": None if plan is None else getattr(plan, "alpha", None),
        "conformal": None if plan is None else getattr(plan, "conformal", None),
        "interval_method": None if plan is None else getattr(plan, "interval_method", None),
        "capability_matrix": probabilistic_capability_matrix(),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_interval_result": interval_result is not None,
        "eval": eval_payload,
        "interval": interval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Bayesian / probabilistic ML fits uncertainty-aware tabular "
            "estimators (native sklearn, optional MAPIE conformal, optional "
            "NGBoost distributions) with train-only split conformal when "
            "enabled. Not a probabilistic-programming platform; not causal."
        ),
    }


def probabilistic_status_for_session(session: Any) -> dict[str, Any]:
    """Report probabilistic status for a Session walkthrough panel.

    Reads probabilistic plan and result slots without mutating the Session.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`probabilistic_status` for the Session's state.
    """
    return probabilistic_status(
        getattr(session, "_probabilistic_plan", None),
        fit_result=getattr(session, "_probabilistic_fit_result", None),
        eval_result=getattr(session, "_probabilistic_eval_result", None),
        interval_result=getattr(session, "_probabilistic_interval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
