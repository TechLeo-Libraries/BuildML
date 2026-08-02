"""History / catalog / walkthrough helpers for knowledge-graph operations."""

from __future__ import annotations

from typing import Any

from buildml.kg.catalog import kg_capability_matrix
from buildml.kg.extras import pykeen_available


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``fit_kg`` history."""
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "backend": payload.get("backend"),
        "method": payload.get("method"),
        "n_train_triples": payload.get("n_train_triples"),
        "n_entities": payload.get("n_entities"),
        "n_relations": payload.get("n_relations"),
        "embedding_dim": payload.get("embedding_dim"),
        "epochs_run": payload.get("epochs_run"),
        "final_loss": payload.get("final_loss"),
        "neg_ratio": payload.get("neg_ratio"),
    }


def score_result_summary(score_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``score_triples`` history."""
    if score_result is None:
        return {}
    payload = (
        score_result.to_dict() if hasattr(score_result, "to_dict") else dict(score_result)
    )
    return {
        "method": payload.get("method"),
        "n_triples": payload.get("n_triples"),
        "unknown_entities": payload.get("unknown_entities"),
        "unknown_relations": payload.get("unknown_relations"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``predict_links`` history."""
    if predict_result is None:
        return {}
    payload = (
        predict_result.to_dict()
        if hasattr(predict_result, "to_dict")
        else dict(predict_result)
    )
    return {
        "mode": payload.get("mode"),
        "method": payload.get("method"),
        "k": payload.get("k"),
        "n_queries": payload.get("n_queries"),
        "n_predictions": payload.get("n_predictions"),
        "filtered": payload.get("filtered"),
    }


def query_result_summary(query_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``query_kg`` history."""
    if query_result is None:
        return {}
    payload = (
        query_result.to_dict() if hasattr(query_result, "to_dict") else dict(query_result)
    )
    return {
        "mode": payload.get("mode"),
        "n_results": payload.get("n_results"),
        "max_hops": payload.get("max_hops"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_kg`` history."""
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "k": payload.get("k"),
        "n_triples_scored": payload.get("n_triples_scored"),
        "metrics": payload.get("metrics"),
    }


def kg_status(
    plan: Any = None,
    *,
    fit_result: Any = None,
    eval_result: Any = None,
    score_result: Any = None,
    predict_result: Any = None,
    query_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Factual walkthrough disclosure for knowledge graphs."""
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "fit_kg",
            "score_triples",
            "predict_links",
            "query_kg",
            "evaluate_kg",
            "save_kg_bundle",
            "load_kg_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"KgPlan backend={getattr(plan, 'backend', None)}, "
                f"method={getattr(plan, 'method', None)}, "
                f"entities={getattr(plan, 'n_entities', None)}, "
                f"relations={getattr(plan, 'n_relations', None)}, "
                f"triples={getattr(plan, 'n_train_triples', None)}.",
                "Session checkpoints do not embed KgPlan; use "
                "save_kg_bundle / load_kg_bundle.",
                "KG is not Graph ML node classification, not Neo4j, not RAG.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif saw:
        disclosures.append(
            "KG operations appear in history, but no live KgPlan is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last KG eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    return {
        "enabled": enabled,
        "present": enabled or saw,
        "has_kg_plan": enabled,
        "backend": None if plan is None else getattr(plan, "backend", None),
        "method": None if plan is None else getattr(plan, "method", None),
        "n_entities": None if plan is None else getattr(plan, "n_entities", None),
        "n_relations": None if plan is None else getattr(plan, "n_relations", None),
        "n_train_triples": (
            None if plan is None else getattr(plan, "n_train_triples", None)
        ),
        "pykeen_available": pykeen_available(),
        "capability_matrix": kg_capability_matrix(),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_score_result": score_result is not None,
        "has_predict_result": predict_result is not None,
        "has_query_result": query_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Knowledge graphs are a Session domain path: (head, relation, tail) "
            "triples → native TransE/DistMult or PyKEEN RotatE/ComplEx link "
            "prediction + symbolic neighbors/path/typed query. Not Graph ML "
            "node-classify (set_graph/fit_graph), not Neo4j, not RAG."
        ),
    }


def kg_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing status helper."""
    return kg_status(
        getattr(session, "_kg_plan", None),
        fit_result=getattr(session, "_kg_fit_result", None),
        eval_result=getattr(session, "_kg_eval_result", None),
        score_result=getattr(session, "_kg_score_result", None),
        predict_result=getattr(session, "_kg_predict_result", None),
        query_result=getattr(session, "_kg_query_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
