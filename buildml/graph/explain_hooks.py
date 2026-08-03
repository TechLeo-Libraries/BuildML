"""History / catalog / walkthrough helpers for graph operations."""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a graph fit result.

    Strips estimator weights while recording method, mode, train counts, and
    train accuracy for Session audit logs.

    Parameters
    ----------
    fit_result:
        :class:`~buildml.graph.results.GraphFitResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Method, mode, task, train node/edge counts, and train accuracy.
    """
    if fit_result is None:
        return {}
    payload = fit_result.to_dict() if hasattr(fit_result, "to_dict") else dict(fit_result)
    return {
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "task": payload.get("task"),
        "n_train_nodes": payload.get("n_train_nodes"),
        "n_edges_fit": payload.get("n_edges_fit"),
        "n_classes": payload.get("n_classes"),
        "train_accuracy": payload.get("train_accuracy"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a graph prediction result.

    Records partition, method, mode, and node count without listing every
    predicted label in Session history.

    Parameters
    ----------
    predict_result:
        :class:`~buildml.graph.results.GraphPredictResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, method, mode, and scored node count.
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
        "mode": payload.get("mode"),
        "n_nodes": payload.get("n_nodes"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Build a compact history summary from a graph evaluation result.

    Records partition, method, mode, node count, and holdout metrics for
    walkthrough panels without embedding full prediction arrays.

    Parameters
    ----------
    eval_result:
        :class:`~buildml.graph.results.GraphEvalResult` or ``None``.

    Returns
    -------
    dict[str, Any]
        Partition, method, mode, node count, and metrics dict.
    """
    if eval_result is None:
        return {}
    payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "n_nodes": payload.get("n_nodes"),
        "metrics": payload.get("metrics"),
    }


def graph_spec_summary(spec: Any) -> dict[str, Any]:
    """Build a compact history summary from a :class:`GraphSpec`.

    Records node-id column, edge/node counts, and directed flag for Session
    history without serialising the full edge table.

    Parameters
    ----------
    spec:
        :class:`~buildml.graph.types.GraphSpec` or ``None``.

    Returns
    -------
    dict[str, Any]
        Node-id column, edge/node counts, and directed flag.
    """
    if spec is None:
        return {}
    payload = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    return {
        "node_id_col": payload.get("node_id_col"),
        "n_edges": payload.get("n_edges"),
        "n_nodes_in_edges": payload.get("n_nodes_in_edges"),
        "directed": payload.get("directed"),
    }


def graph_status(
    plan: Any = None,
    *,
    graph_spec: Any = None,
    fit_result: Any = None,
    predict_result: Any = None,
    eval_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build factual walkthrough disclosure for Graph ML Session state.

    Combines live plan metadata, optional result summaries, history detection,
    and the graph capability matrix for teaching overlays.

    Parameters
    ----------
    plan:
        Active :class:`~buildml.graph.results.GraphPlan`, if any.
    graph_spec:
        Attached :class:`~buildml.graph.types.GraphSpec`, if any.
    fit_result, predict_result, eval_result:
        Last operation reports attached to the Session.
    history:
        Session operation records.

    Returns
    -------
    dict[str, Any]
        Enabled flags, backend metadata, embedded capability matrix,
        disclosures, and boundary text separating Graph ML from KG / Neo4j.
    """
    records = list(history or [])
    saw = any(
        str(r.get("operation_id") or r.get("action"))
        in {
            "set_graph",
            "fit_graph",
            "predict_graph",
            "evaluate_graph",
            "save_graph_bundle",
            "load_graph_bundle",
        }
        for r in records
    )
    enabled = plan is not None
    has_spec = graph_spec is not None or (
        plan is not None and getattr(plan, "graph_spec", None) is not None
    )
    disclosures: list[str] = []
    if enabled:
        disclosures.extend(
            [
                f"GraphPlan method={getattr(plan, 'method', None)}, "
                f"mode={getattr(plan, 'mode', None)}, "
                f"n_train_nodes={getattr(plan, 'n_train_nodes', None)}.",
                "Node splits via Session.split; structure via set_graph.",
                "Session checkpoints do not embed GraphPlan; use "
                "save_graph_bundle / load_graph_bundle.",
                "Honesty: classical NetworkX+sklearn, pure-Torch GCN, "
                "and/or PyG GCN/GraphSAGE/GAT: not Neo4j/KG.",
            ]
        )
        for note in getattr(plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    elif has_spec:
        disclosures.append(
            "GraphSpec is attached, but no GraphPlan is fitted yet. "
            "Call fit_graph after split + roles."
        )
    elif saw:
        disclosures.append(
            "Graph operations appear in history, but no live GraphPlan "
            "or GraphSpec is attached."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = (
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else dict(eval_result)
        )
        disclosures.append(
            "Last graph eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
        "enabled": enabled,
        "present": enabled or saw or has_spec,
        "has_graph_plan": enabled,
        "has_graph_spec": has_spec,
        "method": None if plan is None else getattr(plan, "method", None),
        "mode": None if plan is None else getattr(plan, "mode", None),
        "has_fit_result": fit_result is not None,
        "has_predict_result": predict_result is not None,
        "has_eval_result": eval_result is not None,
        "eval": eval_payload,
        "disclosures": disclosures,
        "boundary": (
            "Graph ML is node classification over an edge list + node feature "
            "table. Inductive mode fits on the train-induced subgraph; "
            "transductive uses full topology with train-label-only supervision. "
            "Not a knowledge-graph / Neo4j product; PyG path is GCN/SAGE/GAT only."
        ),
    },
        "graph_capability_matrix",
    )


def graph_status_for_session(session: Any) -> dict[str, Any]:
    """Report graph-ML status for a Session walkthrough panel.

    Reads graph plan, spec, and result slots without mutating the Session.

    Parameters
    ----------
    session:
        :class:`~buildml.session.session.Session` instance.

    Returns
    -------
    dict[str, Any]
        Same payload as :func:`graph_status` for the Session's graph state.
    """
    return graph_status(
        getattr(session, "_graph_plan", None),
        graph_spec=getattr(session, "_graph_spec", None),
        fit_result=getattr(session, "_graph_fit_result", None),
        predict_result=getattr(session, "_graph_predict_result", None),
        eval_result=getattr(session, "_graph_eval_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )
