"""Thin Session facades over buildml.ranking (tabular LTR)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.ranking.checkpoint import load_ranker_bundle, save_ranker_bundle
from buildml.ranking.evaluate import evaluate_ranker
from buildml.ranking.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    rank_result_summary,
)
from buildml.ranking.fit import fit_ranker
from buildml.ranking.rank import rank
from buildml.ranking.types import (
    PairwiseEstimator,
    PointwiseEstimator,
    RankerBackend,
    RankerMethod,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_ranker_op(
    session,
    *,
    backend: RankerBackend | None = None,
    method: RankerMethod | str | None = None,
    query_column: str | None = None,
    item_column: str | None = None,
    relevance_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    pointwise_estimator: PointwiseEstimator = "ridge",
    pairwise_estimator: PairwiseEstimator = "ranksvm",
    max_pairs_per_query: int = 80,
    relevance_threshold: float = 0.0,
    alpha: float = 1.0,
    C: float = 1.0,
    n_estimators: int = 120,
    learning_rate: float = 0.08,
    hidden_dim: int = 64,
    epochs: int = 40,
    device: str = "cpu",
    random_state: int | None = 0,
):
    """Fit a tabular ranker on Session train rows only.

    Delegates to :func:`buildml.ranking.fit.fit_ranker`, stores the
    :class:`~buildml.ranking.results.RankerPlan` on Session, and records
    the fit. Follow with :func:`rank_op` or :func:`evaluate_ranker_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    backend:
        Optional ranker backend override.
    method:
        Optional method override (pointwise/pairwise/listwise identifiers).
    query_column:
        Query/group column for LTR examples.
    item_column:
        Item/document column for LTR examples.
    relevance_column:
        Relevance or grade column for supervised ranking.
    feature_columns:
        Optional explicit feature columns for ranker inputs.
    pointwise_estimator:
        Pointwise base estimator when method is pointwise.
    pairwise_estimator:
        Pairwise base estimator when method is pairwise.
    max_pairs_per_query:
        Cap on generated pairs per query for pairwise training.
    relevance_threshold:
        Threshold for binarizing graded relevance in some metrics.
    alpha:
        Regularization strength for linear rankers.
    C:
        Inverse regularization for SVM-style pairwise rankers.
    n_estimators:
        Number of trees for GBDT rankers.
    learning_rate:
        Learning rate for GBDT/torch rankers.
    hidden_dim:
        Hidden width for torch listwise ranker.
    epochs:
        Training epochs for torch backend.
    device:
        Torch device string.
    random_state:
        Seed for stochastic training steps.

    Notes
    -----
    **Leakage:** Requires a split. Prefer ``group_split`` on ``query_column``
    so test queries' labels never appear in train. Distinct from RAG and from
    recommender CF. See ``ranking_capability_matrix()`` for backends.
    """
    session.assert_can_fit("train")
    plan, result = fit_ranker(
        session.dataset,
        session._split_plan,
        backend=backend,
        method=method,
        query_column=query_column,
        item_column=item_column,
        relevance_column=relevance_column,
        feature_columns=feature_columns,
        pointwise_estimator=pointwise_estimator,
        pairwise_estimator=pairwise_estimator,
        max_pairs_per_query=max_pairs_per_query,
        relevance_threshold=relevance_threshold,
        alpha=alpha,
        C=C,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
        device=device,
        random_state=random_state,
    )
    session._ranker_plan = plan
    session._ranker_fit_result = result
    session._ranker_eval_result = None
    session._ranker_rank_result = None
    session._record(
        "fit_ranker",
        {
            "backend": plan.backend,
            "method": plan.method,
            "query_column": query_column,
            "item_column": item_column,
            "relevance_column": relevance_column,
            "feature_columns": (
                None if feature_columns is None else list(feature_columns)
            ),
            "pointwise_estimator": pointwise_estimator,
            "pairwise_estimator": pairwise_estimator,
            "max_pairs_per_query": max_pairs_per_query,
            "relevance_threshold": relevance_threshold,
            "alpha": alpha,
            "C": C,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "device": device,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def rank_op(
    session,
    *,
    partition: PartitionOrAll | None = None,
    query_ids: Sequence[Any] | None = None,
    k: int = 10,
    backend: RankerBackend | None = None,
):
    """Order items for queries in a partition or an explicit query id list.

    Delegates to :func:`buildml.ranking.rank.rank` using the fitted ranker
    plan. Defaults to the ``test`` partition when neither ``partition`` nor
    ``query_ids`` is supplied.

    Parameters
    ----------
    session:
        Active Session with a ranker plan from :func:`fit_ranker_op`.
    partition:
        Optional partition to rank (``train``, ``validation``, ``test``,
        or ``all``).
    query_ids:
        Optional explicit query identifiers to rank.
    k:
        Top-k items to return per query.
    backend:
        Optional backend override for ranking.

    Raises
    ------
    ValidationError
        When no ranker plan exists on the Session.
    """
    plan = getattr(session, "_ranker_plan", None)
    if plan is None:
        raise ValidationError("No RankerPlan. Call fit_ranker(...) first.")
    if query_ids is None and partition is None:
        partition = "test"
    result = rank(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        query_ids=query_ids,
        k=k,
        backend=backend,
    )
    session._ranker_rank_result = result
    session._record(
        "rank",
        {
            "partition": partition,
            "query_ids": None if query_ids is None else list(query_ids),
            "k": k,
            "backend": backend,
        },
        warnings=tuple(result.warnings),
        result_summary=rank_result_summary(result),
    )
    return result


def evaluate_ranker_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    k: int = 10,
    backend: RankerBackend | None = None,
):
    """Evaluate per-query ranking metrics on a holdout partition.

    Delegates to :func:`buildml.ranking.evaluate.evaluate_ranker` using the
    frozen train ranker without refitting.

    Parameters
    ----------
    session:
        Active Session with a ranker plan from :func:`fit_ranker_op`.
    partition:
        Holdout partition for evaluation (``test`` by default).
    k:
        Cutoff k for ranking metrics (NDCG@k, etc.).
    backend:
        Optional backend override for evaluation.

    Raises
    ------
    ValidationError
        When no ranker plan exists on the Session.
    """
    plan = getattr(session, "_ranker_plan", None)
    if plan is None:
        raise ValidationError("No RankerPlan. Call fit_ranker(...) first.")
    result = evaluate_ranker(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        k=k,
        backend=backend,
    )
    session._ranker_eval_result = result
    session._record(
        "evaluate_ranker",
        {"partition": partition, "k": k, "backend": backend},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_ranker_bundle_op(session, path: str | Path) -> Path:
    """Persist the active RankerPlan as ``buildml.ranker_bundle.v1``.

    Delegates to :func:`buildml.ranking.checkpoint.save_ranker_bundle`.
    Reload with :func:`load_ranker_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a ranker plan from :func:`fit_ranker_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no ranker plan exists on the Session.
    """
    plan = getattr(session, "_ranker_plan", None)
    if plan is None:
        raise ValidationError("No RankerPlan. Call fit_ranker(...) first.")
    out = save_ranker_bundle(
        path,
        plan,
        fit_result=getattr(session, "_ranker_fit_result", None),
        eval_result=getattr(session, "_ranker_eval_result", None),
        rank_result=getattr(session, "_ranker_rank_result", None),
    )
    session._record(
        "save_ranker_bundle",
        {"path": str(path)},
        result_summary={"path": str(out), "format": "buildml.ranker_bundle.v1"},
    )
    return out


def load_ranker_bundle_op(session, path: str | Path, *, trusted: bool = False):
    """Load a ranker bundle into this Session.

    Delegates to :func:`buildml.ranking.checkpoint.load_ranker_bundle` and
    clears prior fit/eval/rank results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded ranker plan.
    path:
        Path to a ``buildml.ranker_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with ranker plan attached for chaining.
    """
    plan = load_ranker_bundle(path, trusted=trusted)
    session._ranker_plan = plan
    session._ranker_fit_result = None
    session._ranker_eval_result = None
    session._ranker_rank_result = None
    session._record(
        "load_ranker_bundle",
        {"path": str(path)},
        result_summary={"path": str(path), "format": "buildml.ranker_bundle.v1"},
    )
    return cast("Session", session)