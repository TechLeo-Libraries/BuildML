"""History / catalog / walkthrough helpers for active-learning operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history payload from an active-learning fit result.

    Strips heavy estimator objects so Session history records only the fields
    needed for walkthrough overlays and audit replay.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.activelearning.results.ActiveLearningFitResult` or
        compatible mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Strategy, backend, pool sizes, query budget, and target column.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "strategy": payload.get("strategy"),
        "backend": payload.get("backend"),
        "base_estimator": payload.get("base_estimator"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_labeled_train": payload.get("n_labeled_train"),
        "n_unlabeled_pool": payload.get("n_unlabeled_pool"),
        "n_queries_used": payload.get("n_queries_used"),
        "label_budget": payload.get("label_budget"),
        "target_column": payload.get("target_column"),
    }


def query_result_summary(query_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a query suggestion result.

    Captures suggested indices and budget state without embedding full score
    vectors in history.

    Parameters
    ----------
    query_result:
        :class:`~buildml.activelearning.results.ActiveLearningQueryResult` or
        compatible mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Strategy, batch size, pool size, budget remaining, and indices.
    """
    if query_result is None:
        return {}
    payload = (
        query_result.to_dict() if hasattr(query_result, "to_dict") else dict(query_result)
    )
    return {
        "strategy": payload.get("strategy"),
        "n_suggested": payload.get("n_suggested"),
        "n_unlabeled_pool": payload.get("n_unlabeled_pool"),
        "budget_remaining": payload.get("budget_remaining"),
        "indices": payload.get("indices"),
    }


def label_result_summary(label_result: Any) -> dict[str, Any]:
    """Build a compact history payload from a human labeling round.

    Records how many rows were newly labeled and whether a refit occurred.

    Parameters
    ----------
    label_result:
        :class:`~buildml.activelearning.results.ActiveLearningLabelResult` or
        compatible mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Newly labeled count, budget state, refit flag, and indices.
    """
    if label_result is None:
        return {}
    payload = (
        label_result.to_dict() if hasattr(label_result, "to_dict") else dict(label_result)
    )
    return {
        "n_newly_labeled": payload.get("n_newly_labeled"),
        "n_labeled_now": payload.get("n_labeled_now"),
        "n_queries_used": payload.get("n_queries_used"),
        "budget_remaining": payload.get("budget_remaining"),
        "refit": payload.get("refit"),
        "indices": payload.get("indices"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history payload from an active-learning evaluation result.

    Captures partition-level metrics and labeled/unlabeled mix for explain
    overlays without serializing per-row predictions.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.activelearning.results.ActiveLearningEvalResult` or
        compatible mapping; ``None`` yields an empty dict.

    Returns
    -------
    dict[str, Any]
        Partition, labeled/unlabeled counts, query usage, and metrics.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "strategy": payload.get("strategy"),
        "n_labeled_eval": payload.get("n_labeled_eval"),
        "n_unlabeled_eval": payload.get("n_unlabeled_eval"),
        "n_queries_used": payload.get("n_queries_used"),
        "metrics": payload.get("metrics"),
    }


def activelearning_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    query_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for active-learning state.

    Combines live plan fields, latest fit/query/eval payloads, and history
    evidence into a teaching-oriented status dict with capability matrix
    attachment.

    Parameters
    ----------
    plan:
        Optional :class:`~buildml.activelearning.results.ActiveLearningPlan`.
    fit_result:
        Optional latest fit result for ``has_fit_result``.
    query_result:
        Optional latest query result for ``last_query`` summary.
    eval_result:
        Optional latest eval result; metrics are summarized in disclosures.
    history:
        Session operation history used to detect active-learning activity when
        no plan is attached.

    Returns
    -------
    dict[str, Any]
        Enabled flags, pool/budget summary, disclosures, boundary text, and
        nested capability matrix.
    """
    from buildml.activelearning.catalog import activelearning_capability_matrix

    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_active_learner",
            "suggest_query",
            "label_rows",
            "evaluate_active_learning",
            "save_active_learning_bundle",
            "load_active_learning_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"ActiveLearningPlan backend={getattr(plan, 'backend', None)}, "
                f"strategy={getattr(plan, 'strategy', None)}, "
                f"n_labeled_train={getattr(plan, 'n_labeled_train', None)}, "
                f"n_unlabeled_pool={getattr(plan, 'n_unlabeled_pool', None)}, "
                f"n_queries_used={getattr(plan, 'n_queries_used', None)}, "
                f"label_budget={getattr(plan, 'label_budget', None)}.",
                "Unlabeled pool uses NaN missingness by default (train partition only).",
                "Labels come from the user — core never invents an oracle.",
                "Session checkpoints do not embed ActiveLearningPlan; use "
                "save_active_learning_bundle / load_active_learning_bundle.",
                "Distinct from semi-supervised propagation and self-supervised pretext.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "Active-learning operations appear in history, but no live "
            "ActiveLearningPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last active-learning eval: "
            f"partition={eval_payload.get('partition')}, "
            f"n_labeled_eval={eval_payload.get('n_labeled_eval')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    query_payload = None
    if query_result is not None:
        query_payload = (
            query_result.to_dict()
            if hasattr(query_result, "to_dict")
            else dict(query_result)
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_activelearning_plan": enabled,
        "strategy": None if plan is None else getattr(plan, "strategy", None),
        "backend": None if plan is None else getattr(plan, "backend", None),
        "capability_matrix": activelearning_capability_matrix(),
        "n_labeled_train": None if plan is None else getattr(plan, "n_labeled_train", None),
        "n_unlabeled_pool": (
            None if plan is None else getattr(plan, "n_unlabeled_pool", None)
        ),
        "n_queries_used": None if plan is None else getattr(plan, "n_queries_used", None),
        "label_budget": None if plan is None else getattr(plan, "label_budget", None),
        "has_fit_result": fit_result is not None,
        "has_query_result": query_result is not None,
        "has_eval_result": eval_result is not None,
        "last_query": query_payload,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Active learning is human-in-the-loop labeling on the train pool. "
            "Holdout labels are evaluation-only; the library never queries test "
            "and never invents oracle labels."
        ),
    }


def activelearning_status_for_session(session: Any) -> dict[str, Any]:
    """Build active-learning walkthrough status from a Session instance.

    Reads private Session attributes set by active-learning operations and
    delegates to :func:`activelearning_status`.

    Parameters
    ----------
    session:
        BuildML Session with optional ``_activelearning_*`` state attributes.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`activelearning_status` for the session's plan and
        results.
    """
    return activelearning_status(
        getattr(session, "_activelearning_plan", None),
        fit_result=getattr(session, "_activelearning_fit_result", None),
        query_result=getattr(session, "_activelearning_query_result", None),
        eval_result=getattr(session, "_activelearning_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
