"""Thin Session facades over buildml.kg (knowledge graphs)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
    """Fit a knowledge-graph embedding model on Session train triples only.

    Delegates to :func:`buildml.kg.fit.fit_kg`, stores the
    :class:`~buildml.kg.results.KgPlan` on Session, and records the fit.
    Follow with :func:`score_triples_op`, :func:`predict_links_op`, or
    :func:`evaluate_kg_op`.

    Parameters
    ----------
    session:
        Active Session with triple columns and a split plan.
    backend:
        Optional backend override (``native`` or ``pykeen``).
    method:
        Embedding method (``transe``, ``distmult``, ``rotate``, etc.).
    head_column:
        Subject/head entity column; inferred from roles when omitted.
    relation_column:
        Relation/predicate column.
    tail_column:
        Object/tail entity column.
    embedding_dim:
        Latent embedding dimensionality.
    epochs:
        Training epochs over positive triples.
    batch_size:
        Minibatch size for stochastic training.
    learning_rate:
        Optimizer learning rate.
    margin:
        Margin for ranking-loss methods like TransE.
    neg_ratio:
        Negative samples per positive triple per batch.
    norm:
        Distance norm for TransE (``l1`` or ``l2``).
    random_state:
        Seed for negative sampling and initialization.

    Returns
    -------
    KgFitResult
        Serializable fit summary including vocab sizes and disclosures.

    Notes
    -----
    **Leakage:** Requires a split. Vocabularies, embeddings, and adjacency
    use train triples only. Holdout triples never update the model.
    Distinct from Graph ML (``set_graph`` / ``fit_graph``) and from RAG.
    Honesty: Session KG learning: not a Neo4j / graph-DB product.
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
    """Score head-relation-tail triples with the frozen KgPlan.

    Delegates to :func:`buildml.kg.predict.score_triples` without refitting
    embeddings. Use explicit ``triples`` or a Session partition.

    Parameters
    ----------
    session:
        Active Session with a KgPlan from :func:`fit_kg_op`.
    partition:
        Optional split partition whose triples to score.
    triples:
        Optional explicit triples as a DataFrame or sequence of tuples.

    Returns
    -------
    KgScoreResult
        Plausibility scores for each triple.

    Raises
    ------
    ValidationError
        When no KgPlan exists on the Session.
    """
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
    """Predict missing link components using the frozen KgPlan.

    Delegates to :func:`buildml.kg.predict.predict_links` to rank candidate
    tails, heads, or relations for given query entities.

    Parameters
    ----------
    session:
        Active Session with a KgPlan from :func:`fit_kg_op`.
    mode:
        Which component to predict (``tail``, ``head``, or ``relation``).
    heads:
        Optional head entities to query; defaults to all known heads.
    relations:
        Optional relations to constrain predictions.
    tails:
        Optional tail entities for head/relation prediction modes.
    k:
        Number of top-ranked candidates to return per query.
    filtered:
        When True, filter out triples already present in the train graph.

    Returns
    -------
    KgPredictResult
        Ranked link predictions and scores for each query.

    Raises
    ------
    ValidationError
        When no KgPlan exists on the Session.
    """
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
    """Run symbolic KG queries over the train-fitted graph structure.

    Delegates to :func:`buildml.kg.query.query_kg` for neighbor lookup,
    path finding, or typed queries over the frozen KgPlan.

    Parameters
    ----------
    session:
        Active Session with a KgPlan from :func:`fit_kg_op`.
    mode:
        Query mode (``neighbors``, ``path``, or ``typed``).
    entity:
        Anchor entity for neighbor queries.
    source:
        Path query source entity.
    target:
        Path query target entity.
    relation:
        Optional relation filter for neighbor/path queries.
    direction:
        Edge direction to traverse (``out``, ``in``, or ``both``).
    max_hops:
        Maximum path length for path queries.

    Returns
    -------
    KgQueryResult
        Query results as neighbor lists or paths.

    Raises
    ------
    ValidationError
        When no KgPlan exists on the Session.
    """
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
    """Evaluate link prediction with filtered MRR and Hits@K.

    Delegates to :func:`buildml.kg.evaluate.evaluate_kg` on a holdout
    partition without updating embeddings.

    Parameters
    ----------
    session:
        Active Session with a KgPlan from :func:`fit_kg_op`.
    partition:
        Holdout partition containing test triples (default ``test``).
    k:
        Cutoff for Hits@K metrics.

    Returns
    -------
    KgEvalResult
        Filtered ranking metrics (MRR, Hits@K) for the partition.

    Raises
    ------
    ValidationError
        When no KgPlan exists on the Session.
    """
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
    """Persist the active KgPlan as ``buildml.kg_bundle.v1``.

    Delegates to :func:`buildml.kg.checkpoint.save_kg_bundle`.
    Reload with :func:`load_kg_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a KgPlan from :func:`fit_kg_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no KgPlan exists on the Session.
    """
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


def load_kg_bundle_op(session, path: str | Path, *, trusted: bool = False):
    """Load a knowledge-graph bundle into this Session.

    Delegates to :func:`buildml.kg.checkpoint.load_kg_bundle` and clears
    prior score/predict/query/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded KgPlan.
    path:
        Path to a ``buildml.kg_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with KgPlan attached for chaining.
    """
    plan = load_kg_bundle(path, trusted=trusted)
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
    return cast("Session", session)