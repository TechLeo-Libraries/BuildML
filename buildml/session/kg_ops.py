"""Thin Session facades over buildml.kg (knowledge graphs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.kg.checkpoint import load_kg_bundle, save_kg_bundle
from buildml.kg.evaluate import evaluate_kg
from buildml.kg.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
    query_result_summary,
    score_result_summary,
)
from buildml.kg.fit import fit_kg
from buildml.kg.predict import predict_links, score_triples
from buildml.kg.query import query_kg
from buildml.kg.types import (
    KgBackend,
    KgMethod,
    KgNorm,
    KgQueryMode,
    LinkPredictionMode,
)

PartitionOrAll = PartitionName | Literal["all"]
Direction = Literal["out", "in", "both"]


def fit_kg_op(
    session,
    *,
    backend: KgBackend | None = None,
    method: KgMethod = "transe",
    head_column: str | None = None,
    relation_column: str | None = None,
    tail_column: str | None = None,
    embedding_dim: int = 50,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    neg_ratio: int = 1,
    norm: KgNorm = "l1",
    random_state: int | None = 0,
):
    """Fit a KG embedding model on Session train triples only.

    Backends: ``native`` (numpy TransE/DistMult) or ``pykeen`` (RotatE/ComplEx
    when ``buildml[kg-industry]`` is installed).

    Notes
    -----
    **Leakage:** Requires a split. Vocabularies, embeddings, and adjacency
    use train triples only. Holdout triples never update the model.
    Distinct from Graph ML (``set_graph`` / ``fit_graph``) and from RAG.
    Honesty: Session KG learning — not a Neo4j / graph-DB product.
    """
    session.assert_can_fit("train")
    plan, result = fit_kg(
        session.dataset,
        session._split_plan,
        backend=backend,
        method=method,
        head_column=head_column,
        relation_column=relation_column,
        tail_column=tail_column,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        margin=margin,
        neg_ratio=neg_ratio,
        norm=norm,
        random_state=random_state,
    )
    session._kg_plan = plan
    session._kg_fit_result = result
    session._kg_eval_result = None
    session._kg_score_result = None
    session._kg_predict_result = None
    session._kg_query_result = None
    session._record(
        "fit_kg",
        {
            "backend": backend,
            "method": method,
            "head_column": head_column,
            "relation_column": relation_column,
            "tail_column": tail_column,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "margin": margin,
            "neg_ratio": neg_ratio,
            "norm": norm,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def score_triples_op(
    session,
    *,
    partition: PartitionOrAll | None = None,
    triples: pd.DataFrame | Sequence[tuple[Any, Any, Any]] | None = None,
):
    """Score (head, relation, tail) triples with the frozen KgPlan."""
    plan = getattr(session, "_kg_plan", None)
    if plan is None:
        raise ValidationError("No KgPlan. Call fit_kg(...) first.")
    result = score_triples(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        triples=triples,
    )
    session._kg_score_result = result
    session._record(
        "score_triples",
        {
            "partition": partition,
            "n_explicit_triples": None if triples is None else len(list(triples)),
        },
        warnings=tuple(result.warnings),
        result_summary=score_result_summary(result),
    )
    return result


def predict_links_op(
    session,
    *,
    mode: LinkPredictionMode = "tail",
    heads: Sequence[Any] | None = None,
    relations: Sequence[Any] | None = None,
    tails: Sequence[Any] | None = None,
    k: int = 10,
    filtered: bool = True,
):
    """Predict missing link components (tail / head / relation)."""
    plan = getattr(session, "_kg_plan", None)
    if plan is None:
        raise ValidationError("No KgPlan. Call fit_kg(...) first.")
    result = predict_links(
        plan,
        mode=mode,
        heads=heads,
        relations=relations,
        tails=tails,
        k=k,
        filtered=filtered,
    )
    session._kg_predict_result = result
    session._record(
        "predict_links",
        {
            "mode": mode,
            "k": k,
            "filtered": filtered,
            "n_heads": None if heads is None else len(list(heads)),
            "n_relations": None if relations is None else len(list(relations)),
            "n_tails": None if tails is None else len(list(tails)),
        },
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def query_kg_op(
    session,
    *,
    mode: KgQueryMode = "neighbors",
    entity: Any | None = None,
    source: Any | None = None,
    target: Any | None = None,
    relation: Any | None = None,
    direction: Direction = "out",
    max_hops: int = 3,
):
    """Symbolic neighbors / path / typed query over the train KG."""
    plan = getattr(session, "_kg_plan", None)
    if plan is None:
        raise ValidationError("No KgPlan. Call fit_kg(...) first.")
    result = query_kg(
        plan,
        mode=mode,
        entity=entity,
        source=source,
        target=target,
        relation=relation,
        direction=direction,
        max_hops=max_hops,
    )
    session._kg_query_result = result
    session._record(
        "query_kg",
        {
            "mode": mode,
            "entity": entity,
            "source": source,
            "target": target,
            "relation": relation,
            "direction": direction,
            "max_hops": max_hops,
        },
        warnings=tuple(result.warnings),
        result_summary=query_result_summary(result),
    )
    return result


def evaluate_kg_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    k: int = 10,
):
    """Filtered MRR / Hits@K on a holdout partition."""
    plan = getattr(session, "_kg_plan", None)
    if plan is None:
        raise ValidationError("No KgPlan. Call fit_kg(...) first.")
    result = evaluate_kg(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        k=k,
    )
    session._kg_eval_result = result
    session._record(
        "evaluate_kg",
        {"partition": partition, "k": k},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_kg_bundle_op(session, path: str | Path) -> Path:
    plan = getattr(session, "_kg_plan", None)
    if plan is None:
        raise ValidationError("No KgPlan. Call fit_kg(...) first.")
    out = save_kg_bundle(
        path,
        plan,
        fit_result=getattr(session, "_kg_fit_result", None),
        eval_result=getattr(session, "_kg_eval_result", None),
        score_result=getattr(session, "_kg_score_result", None),
        predict_result=getattr(session, "_kg_predict_result", None),
        query_result=getattr(session, "_kg_query_result", None),
    )
    session._record(
        "save_kg_bundle",
        {"path": str(path)},
        result_summary={"path": str(out), "format": "buildml.kg_bundle.v1"},
    )
    return out


def load_kg_bundle_op(session, path: str | Path):
    plan = load_kg_bundle(path)
    session._kg_plan = plan
    session._kg_fit_result = None
    session._kg_eval_result = None
    session._kg_score_result = None
    session._kg_predict_result = None
    session._kg_query_result = None
    session._record(
        "load_kg_bundle",
        {"path": str(path)},
        result_summary={"path": str(path), "format": "buildml.kg_bundle.v1"},
    )
    return session
