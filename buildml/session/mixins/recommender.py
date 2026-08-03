"""Session mixin: recommender domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import recommender_ops
from buildml.session.mixins._shared import *  # noqa: F403


class RecommenderSessionMixin:
    """Public Session methods for the recommender domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _recommender_eval_result: Any
        _recommender_fit_result: Any
        _recommender_plan: Any
        _recommender_recommend_result: Any

    @staticmethod
    def recommender_capability_matrix() -> dict[str, Any]:
        """
        Report which recommender-system backends are available on this machine.

        Call before :meth:`fit_recommender` to confirm implicit, LightFM, or sklearn
        matrix-factorization paths on this install. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Recommender backends, interaction models, and install hints from
            :func:`buildml.recommenders.catalog.recommender_capability_matrix`.
        """
        from buildml.recommenders.catalog import recommender_capability_matrix

        return cast("dict[str, Any]", recommender_capability_matrix())

    def fit_recommender(
        self,
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
        item_feature_columns: list[str] | None = None,
        user_feature_columns: list[str] | None = None,
        cold_start: ColdStartPolicy = "popularity",
        random_state: int | None = 0,
        n_iterations: int = 15,
        lightfm_epochs: int = 10,
    ) -> RecommenderFitResult:
        """Fit a recommender on Session train interactions only.

        Session facade over :func:`buildml.session.recommender_ops.fit_recommender_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
            RecommenderFitResult
                    Fit report with method, backend, and training disclosures.

        See Also
        --------
        :func:`buildml.session.recommender_ops.fit_recommender_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RecommenderFitResult", recommender_ops.fit_recommender_op(
            self,
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
        ))

    def recommend(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = None,
        user_ids: Sequence[Any] | None = None,
        k: int = 10,
        exclude_train_items: bool = True,
    ) -> RecommendResult:
        """Top-K recommendations for partition users or an explicit user id list.

        Session facade over :func:`buildml.session.recommender_ops.recommend_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
            RecommendResult
                    Top-k recommendations per user with provenance and warnings.

        See Also
        --------
        :func:`buildml.session.recommender_ops.recommend_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RecommendResult", recommender_ops.recommend_op(
            self,
            partition=partition,
            user_ids=user_ids,
            k=k,
            exclude_train_items=exclude_train_items,
        ))

    def evaluate_recommender(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int = 10,
    ) -> RecommenderEvalResult:
        """Evaluate ranking metrics on a holdout partition (frozen train plan).

        Session facade over :func:`buildml.session.recommender_ops.evaluate_recommender_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
            RecommenderEvalResult
                    Ranking metrics on the holdout partition.

        See Also
        --------
        :func:`buildml.session.recommender_ops.evaluate_recommender_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RecommenderEvalResult", recommender_ops.evaluate_recommender_op(
            self, partition=partition, k=k
        ))

    @property
    def recommender_plan(self) -> RecommenderPlan | None:
        """Return the recommender plan built by the most recent fit_recommender call.

        Session-held result for ``recommender_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RecommenderPlan | None", self._recommender_plan)

    @property
    def recommender_fit_result(self) -> RecommenderFitResult | None:
        """Return the report from the most recent recommender fit.

        Session-held result for ``recommender_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RecommenderFitResult | None", self._recommender_fit_result)

    @property
    def recommender_eval_result(self) -> RecommenderEvalResult | None:
        """Return the metrics from the most recent recommender evaluation.

        Session-held result for ``recommender_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RecommenderEvalResult | None", self._recommender_eval_result)

    @property
    def recommender_recommend_result(self) -> RecommendResult | None:
        """Return the recommendations from the most recent recommend call.

        Session-held result for ``recommender_recommend_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RecommendResult | None", self._recommender_recommend_result)

    def save_recommender_bundle(self, path: str | Path) -> Path:
        """Persist the active RecommenderPlan as ``buildml.recommender_bundle.v1``.

        Session facade over :func:`buildml.session.recommender_ops.save_recommender_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.recommender_ops.save_recommender_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", recommender_ops.save_recommender_bundle_op(self, path=path))

    def load_recommender_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a recommender bundle into this Session.

        Session facade over :func:`buildml.session.recommender_ops.load_recommender_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with recommender plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.recommender_ops.load_recommender_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", recommender_ops.load_recommender_bundle_op(self, path=path, trusted=trusted))
