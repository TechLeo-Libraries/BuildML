"""History / catalog / walkthrough helpers for probabilistic operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_probabilistic`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
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
    """Compact result_summary for ``evaluate_probabilistic`` history."""
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
    """Compact result_summary for ``predict_probabilistic`` history."""
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
    """Compact result_summary for ``predict_interval`` history."""
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
    """Factual walkthrough disclosure for Bayesian / probabilistic ML."""
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
                f"ProbabilisticPlan estimator={getattr(plan, 'estimator_name', None)}, "
                f"task={getattr(plan, 'task', None)}, "
                f"alpha={getattr(plan, 'alpha', None)}, "
                f"conformal={getattr(plan, 'conformal', None)}, "
                f"interval_method={getattr(plan, 'interval_method', None)}.",
                "Fit and optional split-conformal calibration use Session train "
                "only. Validation/test are evaluation / interval scoring only.",
                "Session.calibration() remains the classical FitResult diagnostic "
                "and is not replaced by this path; evaluate_probabilistic reports "
                "NLL/Brier/ECE for probabilistic classifiers.",
                "Session checkpoints do not embed ProbabilisticPlan; use "
                "save_probabilistic_bundle / load_probabilistic_bundle.",
                "Honesty: sklearn BayesianRidge / GaussianProcess* / GaussianNB "
                "+ optional split conformal — not a PyMC/Stan MCMC platform.",
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
        "estimator_name": None if plan is None else getattr(plan, "estimator_name", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "alpha": None if plan is None else getattr(plan, "alpha", None),
        "conformal": None if plan is None else getattr(plan, "conformal", None),
        "interval_method": None if plan is None else getattr(plan, "interval_method", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_interval_result": interval_result is not None,
        "eval": eval_payload,
        "interval": interval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Bayesian / probabilistic ML fits sklearn uncertainty-aware "
            "estimators with optional train-only split conformal intervals. "
            "Not a probabilistic-programming platform; not causal."
        ),
    }


def probabilistic_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return probabilistic_status(
        getattr(session, "_probabilistic_plan", None),
        fit_result=getattr(session, "_probabilistic_fit_result", None),
        eval_result=getattr(session, "_probabilistic_eval_result", None),
        interval_result=getattr(session, "_probabilistic_interval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
