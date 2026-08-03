"""History / catalog / walkthrough helpers for LTR operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from an LTR fit result.

    Strips estimator weights while recording backend, method, train query/row
    counts, and feature dimension for Session audit logs.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.ranking.results.RankerFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Backend, method, train counts, and estimator metadata.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "method": payload.get("method"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_train_queries": payload.get("n_train_queries"),
        "n_features": payload.get("n_features"),
        "pointwise_estimator": payload.get("pointwise_estimator"),
        "pairwise_estimator": payload.get("pairwise_estimator"),
        "n_pairwise_examples": payload.get("n_pairwise_examples"),
    }


def rank_result_summary(rank_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a rank operation result.

    Records top-k setting, query counts, and ranked item totals without
    embedding full per-query item lists in Session history.

    Parameters
    ----------
    rank_result:
        :class:`~buildml.ranking.results.RankResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, k, query count, and ranked item metadata.
    """
    if rank_result is None:
        return {}
    payload = (
        rank_result.to_dict() if hasattr(rank_result, "to_dict") else dict(rank_result)
    )
    return {
        "k": payload.get("k"),
        "n_queries": payload.get("n_queries"),
        "method": payload.get("method"),
        "n_ranked_items": payload.get("n_ranked_items"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a ranker evaluation result.

    Records partition, k, scored query counts, and macro metrics for
    walkthrough panels without repeating full disclosure text.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.ranking.results.RankerEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, method, k, query counts, and metric dictionary.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "k": payload.get("k"),
        "n_queries_scored": payload.get("n_queries_scored"),
        "metrics": payload.get("metrics"),
    }


def ranking_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    rank_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for tabular LTR.

    Combines live :class:`~buildml.ranking.results.RankerPlan` state, optional
    fit/eval/rank results, and Session history to produce an honest capability
    status payload for teaching overlays.

    Parameters
    ----------
    plan:
        Current train-fitted ranker plan attached to the Session, if any.
    fit_result:
        Last :class:`~buildml.ranking.results.RankerFitResult`, if recorded.
    eval_result:
        Last :class:`~buildml.ranking.results.RankerEvalResult`, if recorded.
    rank_result:
        Last :class:`~buildml.ranking.results.RankResult`, if recorded.
    history:
        Session operation history used to detect prior LTR operations.

    Returns
    -------
    dict[str, Any]
        Enabled flags, plan metadata, eval snapshot, disclosures, and the
        attached ranking capability matrix.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_ranker",
            "rank",
            "evaluate_ranker",
            "save_ranker_bundle",
            "load_ranker_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"RankerPlan backend={getattr(plan, 'backend', None)}, "
                f"method={getattr(plan, 'method', None)}, "
                f"queries={getattr(plan, 'n_train_queries', None)}, "
                f"rows={getattr(plan, 'n_train_rows', None)}, "
                f"features={getattr(plan, 'n_features', None)}.",
                "Session checkpoints do not embed RankerPlan; use "
                "save_ranker_bundle / load_ranker_bundle.",
                "Tabular LTR is not RAG retrieve/generate and not recommender CF.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "LTR operations appear in history, but no live RankerPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last ranker eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
        "enabled": enabled,
        "present": enabled or saw,
        "has_ranker_plan": enabled,
        "method": None if plan is None else getattr(plan, "method", None),
        "backend": None if plan is None else getattr(plan, "backend", None),
        "n_train_queries": None if plan is None else getattr(plan, "n_train_queries", None),
        "n_train_rows": None if plan is None else getattr(plan, "n_train_rows", None),
        "n_features": None if plan is None else getattr(plan, "n_features", None),
        "group_split_disclosed": (
            None if plan is None else getattr(plan, "group_split_disclosed", None)
        ),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_rank_result": rank_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Learning-to-rank is a Session domain path: query–item feature rows "
            "with relevance labels → pointwise or pairwise ranking metrics. "
            "Not RAG, not recommender CF, not a search-engine product."
        ),
    },
        "ranking_capability_matrix",
    )


def ranking_status_for_session(session: Any) -> dict[str, Any]:
    """Build LTR walkthrough status from a Session object.

    Reads private ranker plan and result slots on the Session and delegates to
    :func:`ranking_status` with the Session history log.

    Parameters
    ----------
    session:
        Active BuildML Session carrying optional ranker state attributes.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`ranking_status` for the Session's current LTR
        state.
    """
    return ranking_status(
        getattr(session, "_ranker_plan", None),
        fit_result=getattr(session, "_ranker_fit_result", None),
        eval_result=getattr(session, "_ranker_eval_result", None),
        rank_result=getattr(session, "_ranker_rank_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
