"""History / catalog / walkthrough helpers for TDA operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a :class:`TdaFitResult`.

    Strips diagram arrays and NN indices so Session audit logs stay small while
    recording backend, vectorization, and head configuration.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.tda.results.TdaFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Backend, vectorization, train row count, feature dimension, and task
        metadata. Empty dict when ``fit_result`` is ``None``.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "vectorization": payload.get("vectorization"),
        "n_train_rows": payload.get("n_train_rows"),
        "feature_dim": payload.get("feature_dim"),
        "knn": payload.get("knn"),
        "head": payload.get("head"),
        "task": payload.get("task"),
        "train_score": payload.get("train_score"),
    }


def transform_result_summary(transform_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a :class:`TdaTransformResult`.

    Omits transformed feature matrices so Session history stays lightweight while
    recording partition and vectorization metadata.

    Parameters
    ----------
    transform_result:
        Last transform result or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, row count, feature dimension, and vectorization name.
    """
    if transform_result is None:
        return {}
    payload = (
        transform_result.to_dict()
        if hasattr(transform_result, "to_dict")
        else dict(transform_result)
    )
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "feature_dim": payload.get("feature_dim"),
        "vectorization": payload.get("vectorization"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a :class:`TdaPredictResult`.

    Records prediction counts and task type without embedding raw prediction
    arrays in Session audit logs.

    Parameters
    ----------
    predict_result:
        Last predict result or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, task, row count, and prediction count.
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
        "n_rows": payload.get("n_rows"),
        "task": payload.get("task"),
        "n_predictions": payload.get("n_predictions"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a :class:`TdaEvalResult`.

    Preserves headline metrics and optional diagram-distance summaries without
    full persistence diagrams in history payloads.

    Parameters
    ----------
    eval_result:
        Last evaluation result or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, metrics, optional diagram distances, and backend metadata.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "diagram_distances": payload.get("diagram_distances"),
        "vectorization": payload.get("vectorization"),
        "backend": payload.get("backend"),
    }


def tda_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    transform_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for TDA Session state.

    Combines live :class:`~buildml.tda.results.TdaPlan` facts, optional result
    summaries, Session history detection, and :func:`tda_capability_matrix` for
    teaching overlays and dashboard panels.

    Parameters
    ----------
    plan:
        Active train-fitted TDA plan, if any.
    fit_result, eval_result, transform_result:
        Last operation results attached to the Session.
    history:
        Session operation records to detect past TDA calls without a live plan.

    Returns
    -------
    dict[str, Any]
        Enabled flags, backend/vectorization metadata, embedded capability
        matrix, disclosures, and boundary text separating Session TDA from
        Mapper research tools.
    """
    from buildml.tda.catalog import tda_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_tda",
            "transform_tda",
            "predict_tda",
            "evaluate_tda",
            "save_tda_bundle",
            "load_tda_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"TdaPlan backend={getattr(plan, 'backend', 'native')}, "
                f"vectorization={getattr(plan, 'vectorization', None)}, "
                f"knn={getattr(plan, 'knn', None)}, "
                f"feature_dim={getattr(plan, 'feature_dim', None)}, "
                f"head={getattr(plan, 'head', None)}.",
                "Session checkpoints do not embed TdaPlan; use "
                "save_tda_bundle / load_tda_bundle.",
                "Native: buildml[tda] (ripser+persim). "
                "Industry: buildml[tda-industry] (giotto-tda).",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "TDA operations appear in history, but no live TdaPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last TDA eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_tda_plan": enabled,
        "backend": None if plan is None else getattr(plan, "backend", "native"),
        "vectorization": None if plan is None else getattr(plan, "vectorization", None),
        "knn": None if plan is None else getattr(plan, "knn", None),
        "feature_dim": None if plan is None else getattr(plan, "feature_dim", None),
        "head": None if plan is None else getattr(plan, "head", None),
        "task": None if plan is None else getattr(plan, "task", None),
        "mapper_summary": None
        if plan is None
        else getattr(plan, "mapper_summary_", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_transform_result": transform_result is not None,
        "eval": eval_payload,
        "capability_matrix": tda_capability_matrix(),
        "disclosures": disclosures,
        "boundary": (
            "TDA is a Session domain path: persistent homology + vectorization "
            "→ optional sklearn head. Native (ripser) or giotto (tda-industry). "
            "Not a Mapper research suite."
        ),
    }


def tda_status_for_session(session: Any) -> dict[str, Any]:
    """Report TDA status for a Session walkthrough panel.

    Reads ``_tda_plan``, result slots, and ``_history`` without mutating the
    Session. Convenience wrapper around :func:`tda_status`.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`tda_status` for the Session's TDA state.
    """
    return tda_status(
        getattr(session, "_tda_plan", None),
        fit_result=getattr(session, "_tda_fit_result", None),
        eval_result=getattr(session, "_tda_eval_result", None),
        transform_result=getattr(session, "_tda_transform_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
