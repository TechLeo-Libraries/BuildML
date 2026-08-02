"""History / catalog / walkthrough helpers for federated learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_federated`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "estimator_name": payload.get("estimator_name"),
        "task": payload.get("task"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_clients": payload.get("n_clients"),
        "n_rounds": payload.get("n_rounds"),
        "local_epochs": payload.get("local_epochs"),
        "client_column": payload.get("client_column"),
        "final_train_metric": payload.get("final_train_metric"),
        "used_reduce_components": payload.get("used_reduce_components"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_federated`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "estimator_name": payload.get("estimator_name"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "n_clients_evaluated": payload.get("n_clients_evaluated"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``predict_federated`` history."""
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "estimator_name": payload.get("estimator_name"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
    }


def federated_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    predict_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for federated learning."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_federated",
            "evaluate_federated",
            "predict_federated",
            "save_federated_bundle",
            "load_federated_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"FederatedPlan method={getattr(plan, 'method', None)}, "
                f"estimator={getattr(plan, 'estimator_name', None)}, "
                f"client_column={getattr(plan, 'client_column', None)}, "
                f"n_clients={len(getattr(plan, 'client_ids', ()) or ())}, "
                f"n_rounds_completed={len(getattr(plan, 'round_history', ()) or ())}.",
                "Local client updates use train partition rows only; "
                "validation/test are evaluation-only.",
                "Session checkpoints do not embed FederatedPlan; use "
                "save_federated_bundle / load_federated_bundle.",
                "Honesty: local FedAvg-style simulation — not a distributed "
                "FL platform; not cryptographic secure aggregation.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Federated operations appear in history, but no live "
            "FederatedPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict()
            if hasattr(eval_result, "to_dict")
            else dict(eval_result)
        )
        disclosures.append(
            "Last federated eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_rows={eval_payload.get('n_rows')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    predict_payload = None
    if predict_result is not None:
        predict_payload = (
            predict_result.to_dict()
            if hasattr(predict_result, "to_dict")
            else dict(predict_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_federated_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "estimator_name": (
            None if plan is None else getattr(plan, "estimator_name", None)
        ),
        "client_column": (
            None if plan is None else getattr(plan, "client_column", None)
        ),
        "n_clients": (
            None if plan is None else len(getattr(plan, "client_ids", ()) or ())
        ),
        "n_rounds_completed": (
            None
            if plan is None
            else len(getattr(plan, "round_history", ()) or ())
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_predict_result": predict_result is not None,
        "eval": eval_payload,
        "predict": predict_payload,
        "disclosures": disclosures,
        "boundary": (
            "Federated learning provides a local FedAvg / FedProx simulation "
            "on Session data partitioned by a client/group column. Holdout is "
            "evaluation-only. Not a distributed FL platform (Flower/OpenFL); "
            "not cryptographic secure aggregation; not causal."
        ),
    }


def federated_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return federated_status(
        getattr(session, "_federated_plan", None),
        fit_result=getattr(session, "_federated_fit_result", None),
        eval_result=getattr(session, "_federated_eval_result", None),
        predict_result=getattr(session, "_federated_predict_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
