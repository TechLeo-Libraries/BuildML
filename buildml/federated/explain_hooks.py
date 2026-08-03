"""History / catalog / walkthrough helpers for federated learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a federated fit result.

    Strips heavy estimator objects so Session history records only the fields
    needed for walkthrough overlays and audit replay.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.federated.results.FederatedFitResult` or compatible
        mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Backend, method, client counts, round metadata, and train metric.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "backend": payload.get("backend"),
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
    """Build a compact history payload from a federated evaluation result.

    Captures partition-level holdout metrics and per-client evaluation counts
    for explain overlays without serializing full metric blobs.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.federated.results.FederatedEvalResult` or compatible
        mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Partition, row counts, aggregate metrics, and clients evaluated.
    """
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
    """Build a compact history payload from a federated predict result.

    Records partition and prediction counts without embedding full prediction
    tuples in Session history.

    Parameters
    ----------
    predict_result:
        :class:`~buildml.federated.results.FederatedPredictResult` or compatible
        mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Partition, row counts, and prediction count metadata.
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
    """Build factual walkthrough disclosure for federated learning state.

    Combines live plan fields, latest fit/eval/predict payloads, and history
    evidence into a teaching-oriented status dict with capability matrix
    attachment.

    Parameters
    ----------
    plan:
        Optional :class:`~buildml.federated.results.FederatedPlan`.
    fit_result:
        Optional latest fit result for ``has_fit_result``.
    eval_result:
        Optional latest eval result; metrics are summarized in disclosures.
    predict_result:
        Optional latest predict result.
    history:
        Session operation history used to detect federated activity when no
        plan is attached.

    Returns
    -------
    dict[str, Any]
        Enabled flags, client partition summary, disclosures, boundary text,
        and nested capability matrix from
        :func:`buildml.explain.capability_status.attach_capability_matrix`.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_federated",
            "evaluate_federated",
            "predict_federated",
            "save_federated_bundle",
            "load_federated_bundle",
            "export_round_history",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"FederatedPlan backend={getattr(plan, 'backend', 'native')}, "
                f"method={getattr(plan, 'method', None)}, "
                f"estimator={getattr(plan, 'estimator_name', None)}, "
                f"client_column={getattr(plan, 'client_column', None)}, "
                f"n_clients={len(getattr(plan, 'client_ids', ()) or ())}, "
                f"n_rounds_completed={len(getattr(plan, 'round_history', ()) or ())}.",
                "Local client updates use train partition rows only; "
                "validation/test are evaluation-only.",
                "Session checkpoints do not embed FederatedPlan; use "
                "save_federated_bundle / load_federated_bundle.",
                "Honesty: local FedAvg-style simulation: not a networked "
                "FL deployment unless you operate Flower separately; not "
                "cryptographic secure aggregation.",
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

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
        "enabled": enabled,
        "present": enabled or saw,
        "has_federated_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "backend": None if plan is None else getattr(plan, "backend", "native"),
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
            "on Session data partitioned by a client/group column. "
            "backend='native' uses in-process coef aggregation; "
            "backend='flower' uses Flower NumPyClient + flwr aggregation "
            "(still local unless you deploy Flower). Holdout is "
            "evaluation-only. Not cryptographic secure aggregation; not causal."
        ),
    },
        "federated_capability_matrix",
    )


def federated_status_for_session(session: Any) -> dict[str, Any]:
    """Build federated walkthrough status from a Session instance.

    Reads private Session attributes set by federated operations and delegates
    to :func:`federated_status`.

    Parameters
    ----------
    session:
        BuildML Session with optional ``_federated_*`` state attributes.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`federated_status` for the session's plan and
        results.
    """
    return federated_status(
        getattr(session, "_federated_plan", None),
        fit_result=getattr(session, "_federated_fit_result", None),
        eval_result=getattr(session, "_federated_eval_result", None),
        predict_result=getattr(session, "_federated_predict_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
