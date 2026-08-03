"""Thin Session facades over buildml.recommenders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.recommenders.checkpoint import (
    load_recommender_bundle,
    save_recommender_bundle,
)
from buildml.recommenders.evaluate import evaluate_recommender
from buildml.recommenders.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    recommend_result_summary,
)
from buildml.recommenders.fit import fit_recommender
from buildml.recommenders.recommend import recommend
from buildml.recommenders.types import (
    ColdStartPolicy,
    FeedbackMode,
    RecommenderBackend,
    RecommenderMethod,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_recommender_op(
    session,
    *,
    method: RecommenderMethod | None = None,
    backend: RecommenderBackend | None = None,
    user_column: str | None = None,
    item_column: str | None = None,
    rating_column: str | None = None,
    feedback: FeedbackMode = "explicit",
    n_neighbors: int = 40,
    n_factors: int = 32,
    min_rating: float | None = None,
    item_feature_columns: Sequence[str] | None = None,
    user_feature_columns: Sequence[str] | None = None,
    cold_start: ColdStartPolicy = "popularity",
    random_state: int | None = 0,
    n_iterations: int = 15,
    lightfm_epochs: int = 10,
):
    """Fit a recommender on Session train interactions only.

    Delegates to :func:`buildml.recommenders.fit.fit_recommender`, stores the
    :class:`~buildml.recommenders.results.RecommenderPlan` on Session, and
    records the fit. Follow with :func:`recommend_op` or
    :func:`evaluate_recommender_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    method:
        Optional recommender method override.
    backend:
        Optional backend override (sklearn, implicit, lightfm).
    user_column:
        User identifier column.
    item_column:
        Item identifier column.
    rating_column:
        Rating or interaction strength column.
    feedback:
        Feedback mode (``explicit`` or ``implicit``).
    n_neighbors:
        Neighborhood size for kNN-style recommenders.
    n_factors:
        Latent factor dimension for matrix-factorization methods.
    min_rating:
        Optional minimum rating threshold for explicit feedback.
    item_feature_columns:
        Optional item-side content feature columns.
    user_feature_columns:
        Optional user-side content feature columns.
    cold_start:
        Cold-start policy for unseen users/items.
    random_state:
        Seed for stochastic training steps.
    n_iterations:
        Iteration count for ALS-style trainers.
    lightfm_epochs:
        Epoch count for LightFM backend.

    Notes
    -----
    **Leakage:** Requires a split. Similarities / factors / content profiles
    use train interactions only. Holdout items may appear as cold catalog
    misses (known-item protocol). Distinct from RAG and EDA Recommendation
    Findings.

    When ``feedback='implicit'`` and ``method`` is omitted, defaults to ALS
    (``implicit`` library) when ``buildml[recommenders-industry]`` is installed.
    """
    session.assert_can_fit("train")
    plan, result = fit_recommender(
        session.dataset,
        session._split_plan,
        method=method,
        backend=backend,
        user_column=user_column,
        item_column=item_column,
        rating_column=rating_column,
        feedback=feedback,
        n_neighbors=n_neighbors,
        n_factors=n_factors,
        min_rating=min_rating,
        item_feature_columns=item_feature_columns,
        user_feature_columns=user_feature_columns,
        cold_start=cold_start,
        random_state=random_state,
        n_iterations=n_iterations,
        lightfm_epochs=lightfm_epochs,
    )
    session._recommender_plan = plan
    session._recommender_fit_result = result
    session._recommender_eval_result = None
    session._recommender_recommend_result = None
    session._record(
        "fit_recommender",
        {
            "method": result.method,
            "backend": result.backend,
            "user_column": user_column,
            "item_column": item_column,
            "rating_column": rating_column,
            "feedback": feedback,
            "n_neighbors": n_neighbors,
            "n_factors": n_factors,
            "min_rating": min_rating,
            "item_feature_columns": (
                None if item_feature_columns is None else list(item_feature_columns)
            ),
            "user_feature_columns": (
                None if user_feature_columns is None else list(user_feature_columns)
            ),
            "cold_start": cold_start,
            "random_state": random_state,
            "n_iterations": n_iterations,
            "lightfm_epochs": lightfm_epochs,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def recommend_op(
    session,
    *,
    partition: PartitionOrAll | None = None,
    user_ids: Sequence[Any] | None = None,
    k: int = 10,
    exclude_train_items: bool = True,
):
    """Top-K recommendations for partition users or an explicit user id list.

    Delegates to :func:`buildml.recommenders.recommend.recommend` using the
    fitted plan. Defaults to the ``test`` partition when neither ``partition``
    nor ``user_ids`` is supplied.

    Parameters
    ----------
    session:
        Active Session with a recommender plan from :func:`fit_recommender_op`.
    partition:
        Optional partition whose users receive recommendations.
    user_ids:
        Optional explicit user identifiers to recommend for.
    k:
        Number of items to recommend per user.
    exclude_train_items:
        When True, exclude items seen in train interactions.

    Raises
    ------
    ValidationError
        When no recommender plan exists on the Session.
    """
    plan = getattr(session, "_recommender_plan", None)
    if plan is None:
        raise ValidationError("No RecommenderPlan. Call fit_recommender(...) first.")
    if user_ids is None and partition is None:
        partition = "test"
    result = recommend(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        user_ids=user_ids,
        k=k,
        exclude_train_items=exclude_train_items,
    )
    session._recommender_recommend_result = result
    session._record(
        "recommend",
        {
            "partition": partition,
            "user_ids": None if user_ids is None else list(user_ids),
            "k": k,
            "exclude_train_items": exclude_train_items,
        },
        warnings=tuple(result.warnings),
        result_summary=recommend_result_summary(result),
    )
    return result


def evaluate_recommender_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    k: int = 10,
):
    """Evaluate ranking metrics on a holdout partition (frozen train plan).

    Delegates to :func:`buildml.recommenders.evaluate.evaluate_recommender`
    without refitting the recommender on holdout interactions.

    Parameters
    ----------
    session:
        Active Session with a recommender plan from :func:`fit_recommender_op`.
    partition:
        Holdout partition for evaluation (``test`` by default).
    k:
        Cutoff k for ranking metrics.

    Raises
    ------
    ValidationError
        When no recommender plan exists on the Session.
    """
    plan = getattr(session, "_recommender_plan", None)
    if plan is None:
        raise ValidationError("No RecommenderPlan. Call fit_recommender(...) first.")
    result = evaluate_recommender(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        k=k,
    )
    session._recommender_eval_result = result
    session._record(
        "evaluate_recommender",
        {"partition": partition, "k": k},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_recommender_bundle_op(session, path: str | Path) -> Path:
    """Persist the active RecommenderPlan as ``buildml.recommender_bundle.v1``.

    Delegates to :func:`buildml.recommenders.checkpoint.save_recommender_bundle`.
    Reload with :func:`load_recommender_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a recommender plan from :func:`fit_recommender_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no recommender plan exists on the Session.
    """
    plan = getattr(session, "_recommender_plan", None)
    if plan is None:
        raise ValidationError("No RecommenderPlan. Call fit_recommender(...) first.")
    out = save_recommender_bundle(
        path,
        plan,
        fit_result=getattr(session, "_recommender_fit_result", None),
        eval_result=getattr(session, "_recommender_eval_result", None),
        recommend_result=getattr(session, "_recommender_recommend_result", None),
    )
    session._record(
        "save_recommender_bundle",
        {"path": str(path)},
        result_summary={"path": str(out), "format": "buildml.recommender_bundle.v1"},
    )
    return out


def load_recommender_bundle_op(session, path: str | Path):
    """Load a recommender bundle into this Session.

    Delegates to :func:`buildml.recommenders.checkpoint.load_recommender_bundle`
    and clears prior fit/eval/recommend results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded recommender plan.
    path:
        Path to a ``buildml.recommender_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with recommender plan attached for chaining.
    """
    plan = load_recommender_bundle(path)
    session._recommender_plan = plan
    session._recommender_fit_result = None
    session._recommender_eval_result = None
    session._recommender_recommend_result = None
    session._record(
        "load_recommender_bundle",
        {"path": str(path)},
        result_summary={"path": str(path), "format": "buildml.recommender_bundle.v1"},
    )
    return session
