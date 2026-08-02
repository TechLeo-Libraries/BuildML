"""History / catalog / walkthrough helpers for anomaly operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_anomaly`` history."""
    if fit_result is None:
        return {}
    if hasattr(fit_result, "to_dict"):
        payload = fit_result.to_dict()
    else:
        payload = dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_fit_rows": payload.get("n_fit_rows"),
        "threshold_policy": payload.get("threshold_policy"),
        "threshold": payload.get("threshold"),
        "contamination": payload.get("contamination"),
        "train_alert_rate": payload.get("train_alert_rate"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def score_result_summary(score_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``score_anomalies`` history."""
    if score_result is None:
        return {}
    if hasattr(score_result, "to_dict"):
        payload = score_result.to_dict()
    else:
        payload = dict(score_result)
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "n_flagged": payload.get("n_flagged"),
        "alert_rate": payload.get("alert_rate"),
        "threshold": payload.get("threshold"),
        "threshold_policy": payload.get("threshold_policy"),
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "attached": payload.get("attached"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_anomaly`` history."""
    if eval_result is None:
        return {}
    if hasattr(eval_result, "to_dict"):
        payload = eval_result.to_dict()
    else:
        payload = dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "n_rows": payload.get("n_rows"),
        "alert_rate": payload.get("alert_rate"),
        "threshold": payload.get("threshold"),
        "metrics": payload.get("metrics"),
        "labeled_metrics": payload.get("labeled_metrics"),
        "positive_rate": payload.get("positive_rate"),
    }


def anomaly_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for anomaly / fraud detection."""
    from buildml.anomaly.catalog import anomaly_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_anomaly",
            "score_anomalies",
            "evaluate_anomaly",
            "tune_anomaly_threshold",
            "save_anomaly_bundle",
            "load_anomaly_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"AnomalyPlan backend={getattr(plan, 'backend', None)}, "
                f"method={getattr(plan, 'method', None)}, "
                f"mode={getattr(plan, 'mode', None)}, "
                f"threshold={getattr(plan, 'threshold_', None)} "
                f"({getattr(plan, 'threshold_policy', None)}), "
                f"train_alert_rate={getattr(plan, 'train_alert_rate_', None)}.",
                "Session checkpoints do not embed AnomalyPlan; use "
                "save_anomaly_bundle / load_anomaly_bundle.",
                "EDA IsolationForest screens and handle_outliers fences are not this path.",
                "Not a full fraud platform (no graph fraud, no online streaming product).",
            ]
        )
        if getattr(plan, "used_reduce_components", False):
            disclosures.append("Detector was fit on reduce_dimensions component columns.")
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Anomaly operations appear in history, but no live AnomalyPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last anomaly eval: "
            f"partition={eval_payload.get('partition')}, "
            f"alert_rate={eval_payload.get('alert_rate')}, "
            f"labeled_metrics={eval_payload.get('labeled_metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_anomaly_plan": enabled,
        "backend": None if plan is None else getattr(plan, "backend", None),
        "method": None if plan is None else getattr(plan, "method", None),
        "mode": None if plan is None else getattr(plan, "mode", None),
        "threshold": None if plan is None else getattr(plan, "threshold_", None),
        "threshold_policy": None if plan is None else getattr(plan, "threshold_policy", None),
        "train_alert_rate": None if plan is None else getattr(plan, "train_alert_rate_", None),
        "used_reduce_components": (
            None if plan is None else getattr(plan, "used_reduce_components", None)
        ),
        "capability_matrix": anomaly_capability_matrix(),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Anomaly / fraud detection is a Session domain path distinct from EDA "
            "IsolationForest screens, preprocess outlier fences, unsupervised "
            "clustering, and supervised Session.fit unless mode='supervised'."
        ),
    }


def anomaly_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return anomaly_status(
        getattr(session, "_anomaly_plan", None),
        fit_result=getattr(session, "_anomaly_fit_result", None),
        eval_result=getattr(session, "_anomaly_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
