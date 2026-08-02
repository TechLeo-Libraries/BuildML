"""History / catalog / walkthrough helpers for synthetic-data operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "n_columns": payload.get("n_columns"),
        "column_kinds": payload.get("column_kinds"),
    }


def sample_result_summary(sample_result: Any) -> dict[str, Any]:
    if sample_result is None:
        return {}
    payload = (
        sample_result.to_dict() if hasattr(sample_result, "to_dict") else dict(sample_result)
    )
    return {
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "merged": payload.get("merged"),
        "merge_mode": payload.get("merge_mode"),
        "provenance_column": payload.get("provenance_column"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "mode": payload.get("mode"),
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_real": payload.get("n_real"),
        "n_synthetic": payload.get("n_synthetic"),
        "metrics": payload.get("metrics"),
    }


def synthetic_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    sample_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for synthetic-data systems."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_synthesizer",
            "sample_synthetic",
            "evaluate_synthetic",
            "save_synthetic_bundle",
            "load_synthetic_bundle",
            "resample",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"SynthesizerPlan method={getattr(plan, 'method', None)}, "
                f"fitted_on={getattr(plan, 'partition_fitted', None)}, "
                f"n_rows={getattr(plan, 'n_rows_fitted', None)}.",
                "Session checkpoints do not embed SynthesizerPlan; use "
                "save_synthetic_bundle / load_synthetic_bundle.",
                "Cross-link: Session.resample remains class-balance "
                "preprocessing (buildml[imbalanced]); this path is the "
                "reusable synthetic-data product.",
                "Not a differential-privacy product.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Synthetic / resample operations appear in history, but no live "
            "SynthesizerPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last synthetic eval: "
            f"mode={eval_payload.get('mode')}, "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_synthesizer_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "partition_fitted": (
            None if plan is None else getattr(plan, "partition_fitted", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_sample_result": sample_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Synthetic-data systems are a Session domain path: train-fitted "
            "bootstrap / Gaussian copula / SMOTE generators with "
            "sample_synthetic, fidelity or TSTR evaluate_synthetic, and "
            "optional extend_train merge with provenance. Not SDV/CTGAN "
            "stacks in core; not differential privacy. "
            "Session.resample remains the class-balance preprocess path."
        ),
    }


def synthetic_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return synthetic_status(
        getattr(session, "_synthesizer_plan", None),
        fit_result=getattr(session, "_synthetic_fit_result", None),
        eval_result=getattr(session, "_synthetic_eval_result", None),
        sample_result=getattr(session, "_synthetic_sample_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
