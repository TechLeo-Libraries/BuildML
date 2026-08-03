"""Session mixin: ranking domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import ranking_ops
from buildml.session.mixins._shared import *  # noqa: F403


class RankingSessionMixin:
    """Public Session methods for the ranking domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _ranker_eval_result: Any
        _ranker_fit_result: Any
        _ranker_plan: Any
        _ranker_rank_result: Any

    @staticmethod
    def ranking_capability_matrix() -> dict[str, Any]:
        """
        Report which learning-to-rank backends and objectives are available here.

        Call before :meth:`fit_ranker` to confirm LightGBM/XGBoost/CatBoost or
        sklearn fallbacks before writing a fit call that will fail on this install.
        Read-only: no dataset required.

        Returns
        -------
        dict[str, Any]
            Ranker backends, supported objectives, and install hints from
            :func:`buildml.ranking.catalog.ranking_capability_matrix`.
        """
        from buildml.ranking.catalog import ranking_capability_matrix

        return cast("dict[str, Any]", ranking_capability_matrix())

    def fit_ranker(
        self,
        *,
        backend: RankerBackend | None = None,
        method: RankerMethod | str | None = None,
        query_column: str | None = None,
        item_column: str | None = None,
        relevance_column: str | None = None,
        feature_columns: list[str] | None = None,
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
    ) -> RankerFitResult:
        """Fit a tabular ranker on Session train rows only.

        Session facade over :func:`buildml.session.ranking_ops.fit_ranker_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
            RankerFitResult
                    Fit report with backend, method, and training disclosures.

        See Also
        --------
        :func:`buildml.session.ranking_ops.fit_ranker_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RankerFitResult", ranking_ops.fit_ranker_op(
            self,
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
        ))

    def rank(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = None,
        query_ids: Sequence[Any] | None = None,
        k: int = 10,
        backend: RankerBackend | None = None,
    ) -> RankResult:
        """Order items for queries in a partition or an explicit query id list.

        Session facade over :func:`buildml.session.ranking_ops.rank_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
            RankResult
                    Ranked items per query with scores and provenance.

        See Also
        --------
        :func:`buildml.session.ranking_ops.rank_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RankResult", ranking_ops.rank_op(
            self,
            partition=partition,
            query_ids=query_ids,
            k=k,
            backend=backend,
        ))

    def evaluate_ranker(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int = 10,
        backend: RankerBackend | None = None,
    ) -> RankerEvalResult:
        """Evaluate per-query ranking metrics on a holdout partition.

        Session facade over :func:`buildml.session.ranking_ops.evaluate_ranker_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
            RankerEvalResult
                    Per-query ranking metrics on the holdout partition.

        See Also
        --------
        :func:`buildml.session.ranking_ops.evaluate_ranker_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RankerEvalResult", ranking_ops.evaluate_ranker_op(
            self, partition=partition, k=k, backend=backend
        ))

    @property
    def ranker_plan(self) -> RankerPlan | None:
        """Return the ranker plan built by the most recent fit_ranker call.

        Session-held result for ``ranker_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RankerPlan | None", self._ranker_plan)

    @property
    def ranker_fit_result(self) -> RankerFitResult | None:
        """Return the report from the most recent ranker fit.

        Session-held result for ``ranker_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RankerFitResult | None", self._ranker_fit_result)

    @property
    def ranker_eval_result(self) -> RankerEvalResult | None:
        """Return the metrics from the most recent ranker evaluation.

        Session-held result for ``ranker_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RankerEvalResult | None", self._ranker_eval_result)

    @property
    def ranker_rank_result(self) -> RankResult | None:
        """
        Return the rankings from the most recent rank call.

        Stored on Session after :meth:`rank` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.ranking.results.RankResult` or None
            ``None`` until :meth:`rank` has run.
        """
        return cast("RankResult | None", self._ranker_rank_result)

    def save_ranker_bundle(self, path: str | Path) -> Path:
        """Persist the active RankerPlan as ``buildml.ranker_bundle.v1``.

        Session facade over :func:`buildml.session.ranking_ops.save_ranker_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.ranking_ops.save_ranker_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", ranking_ops.save_ranker_bundle_op(self, path=path))

    def load_ranker_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a ranker bundle into this Session.

        Session facade over :func:`buildml.session.ranking_ops.load_ranker_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with ranker plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.ranking_ops.load_ranker_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", ranking_ops.load_ranker_bundle_op(self, path=path, trusted=trusted))
