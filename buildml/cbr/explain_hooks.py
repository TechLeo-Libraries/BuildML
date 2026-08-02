"""History / catalog / walkthrough helpers for CBR operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for fit_cbr history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "task": payload.get("task"),
        "backend": payload.get("backend"),
        "metric": payload.get("metric"),
        "reuse": payload.get("reuse"),
        "k": payload.get("k"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_cases": payload.get("n_cases"),
        "target_column": payload.get("target_column"),
        "train_score": payload.get("train_score"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for evaluate_cbr history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "mean_neighbor_distance": payload.get("mean_neighbor_distance"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for predict_cbr history."""
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
        "n_traces": payload.get("n_traces"),
    }


def retrieve_result_summary(retrieve_result: Any) -> dict[str, Any]:
    """Compact result_summary for retrieve_cases history."""
    if retrieve_result is None:
        return {}
    payload = (
        retrieve_result.to_dict()
        if hasattr(retrieve_result, "to_dict")
        else dict(retrieve_result)
    )
    return {
        "partition": payload.get("partition"),
        "k": payload.get("k"),
        "metric": payload.get("metric"),
        "n_queries": payload.get("n_queries"),
        "n_traces": payload.get("n_traces"),
    }


def retain_result_summary(retain_result: Any) -> dict[str, Any]:
    """Compact result_summary for retain_cbr history."""
    if retain_result is None:
        return {}
    payload = (
        retain_result.to_dict()
        if hasattr(retain_result, "to_dict")
        else dict(retain_result)
    )
    return {
        "n_added": payload.get("n_added"),
        "n_cases_after": payload.get("n_cases_after"),
        "n_skipped": payload.get("n_skipped"),
    }


def cbr_status(
    cbr_plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for case-based reasoning."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_cbr",
            "retrieve_cases",
            "predict_cbr",
            "evaluate_cbr",
            "retain_cbr",
            "save_cbr_bundle",
            "load_cbr_bundle",
        }
        for r in records
    )
    enabled = cbr_plan is not None
    disclosures: list[str] = []
    if cbr_plan is not None:
        n_cases = None
        n_retained = None
        cb = getattr(cbr_plan, "case_base", None)
        if cb is not None:
            n_cases = getattr(cb, "n_cases", None)
            n_retained = getattr(cb, "n_retained", None)
        disclosures.extend(
            [
                f"CbrPlan task={getattr(cbr_plan, 'task', None)}, "
                f"backend={getattr(cbr_plan, 'backend', None)}, "
                f"metric={getattr(cbr_plan, 'metric', None)}, "
                f"reuse={getattr(cbr_plan, 'reuse', None)}, "
                f"k={getattr(cbr_plan, 'k', None)}, "
                f"n_cases={n_cases}, n_retained={n_retained}.",
                "Case memory is built from Session train only. "
                "Validation/test are retrieve / predict / evaluate only "
                "(retain refuses holdout indices).",
                "Session checkpoints do not embed CbrPlan; use "
                "save_cbr_bundle / load_cbr_bundle.",
                "Honesty: tabular case→solution CBR — not RAG document "
                "retrieval for generation, not a vector DB product.",
            ]
        )
        for note in getattr(cbr_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw and not enabled:
        disclosures.append(
            "CBR operations appear in history, but no live CbrPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last CBR eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_cbr_plan": cbr_plan is not None,
        "task": None if cbr_plan is None else getattr(cbr_plan, "task", None),
        "backend": None if cbr_plan is None else getattr(cbr_plan, "backend", None),
        "metric": None if cbr_plan is None else getattr(cbr_plan, "metric", None),
        "reuse": None if cbr_plan is None else getattr(cbr_plan, "reuse", None),
        "k": None if cbr_plan is None else getattr(cbr_plan, "k", None),
        "n_cases": (
            None
            if cbr_plan is None
            else getattr(getattr(cbr_plan, "case_base", None), "n_cases", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Case-based reasoning retrieves similar train cases and reuses/"
            "adapts their solutions for supervised-style tasks. Distinct from "
            "RAG (text corpus → generation). Not a vector DB product."
        ),
    }


def cbr_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return cbr_status(
        getattr(session, "_cbr_plan", None),
        fit_result=getattr(session, "_cbr_fit_result", None),
        eval_result=getattr(session, "_cbr_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
