"""Session mixin: kg domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import kg_ops
from buildml.session.mixins._shared import *  # noqa: F403


class KgSessionMixin:
    """Public Session methods for the kg domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _kg_eval_result: Any
        _kg_fit_result: Any
        _kg_plan: Any
        _kg_predict_result: Any
        _kg_query_result: Any
        _kg_score_result: Any

    @staticmethod
    def kg_capability_matrix() -> dict[str, Any]:
        """
        Report which knowledge-graph learning backends are available on this machine.

        Call before link-prediction or embedding fit methods to confirm PyKEEN,
        DGL-KE, or native paths on this install. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            KG backends, tasks, and install hints from
            :func:`buildml.kg.catalog.kg_capability_matrix`.
        """
        from buildml.kg.catalog import kg_capability_matrix

        return cast("dict[str, Any]", kg_capability_matrix())

    def fit_kg(
        self,
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
    ) -> KgFitResult:
        """Fit a knowledge-graph embedding model on Session train triples only.

        Session facade over :func:`buildml.session.kg_ops.fit_kg_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        KgFitResult
            Serializable fit summary including vocab sizes and disclosures.

        See Also
        --------
        :func:`buildml.session.kg_ops.fit_kg_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("KgFitResult", kg_ops.fit_kg_op(
            self,
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
        ))

    def score_triples(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = None,
        triples: Any | None = None,
    ) -> ScoreTriplesResult:
        """Score head-relation-tail triples with the frozen KgPlan.

        Session facade over :func:`buildml.session.kg_ops.score_triples_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        KgScoreResult
            Plausibility scores for each triple.

        See Also
        --------
        :func:`buildml.session.kg_ops.score_triples_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ScoreTriplesResult", kg_ops.score_triples_op(
            self, partition=partition, triples=triples
        ))

    def predict_links(
        self,
        *,
        mode: LinkPredictionMode = "tail",
        heads: Sequence[Any] | None = None,
        relations: Sequence[Any] | None = None,
        tails: Sequence[Any] | None = None,
        k: int = 10,
        filtered: bool = True,
    ) -> PredictLinksResult:
        """Predict missing link components using the frozen KgPlan.

        Session facade over :func:`buildml.session.kg_ops.predict_links_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        KgPredictResult
            Ranked link predictions and scores for each query.

        See Also
        --------
        :func:`buildml.session.kg_ops.predict_links_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("PredictLinksResult", kg_ops.predict_links_op(
            self,
            mode=mode,
            heads=heads,
            relations=relations,
            tails=tails,
            k=k,
            filtered=filtered,
        ))

    def query_kg(
        self,
        *,
        mode: KgQueryMode = "neighbors",
        entity: Any | None = None,
        source: Any | None = None,
        target: Any | None = None,
        relation: Any | None = None,
        direction: Literal["out", "in", "both"] = "out",
        max_hops: int = 3,
    ) -> KgQueryResult:
        """Run symbolic KG queries over the train-fitted graph structure.

        Session facade over :func:`buildml.session.kg_ops.query_kg_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        KgQueryResult
            Query results as neighbor lists or paths.

        See Also
        --------
        :func:`buildml.session.kg_ops.query_kg_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("KgQueryResult", kg_ops.query_kg_op(
            self,
            mode=mode,
            entity=entity,
            source=source,
            target=target,
            relation=relation,
            direction=direction,
            max_hops=max_hops,
        ))

    def evaluate_kg(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int = 10,
    ) -> KgEvalResult:
        """Evaluate link prediction with filtered MRR and Hits@K.

        Session facade over :func:`buildml.session.kg_ops.evaluate_kg_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        KgEvalResult
            Filtered ranking metrics (MRR, Hits@K) for the partition.

        See Also
        --------
        :func:`buildml.session.kg_ops.evaluate_kg_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("KgEvalResult", kg_ops.evaluate_kg_op(self, partition=partition, k=k))

    @property
    def kg_plan(self) -> KgPlan | None:
        """Return the knowledge-graph plan built by the most recent fit_kg call.

        Session-held result for ``kg_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("KgPlan | None", self._kg_plan)

    @property
    def kg_fit_result(self) -> KgFitResult | None:
        """
        Return the report from the most recent KG fit.

        Stored on Session after :meth:`fit_kg` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgFitResult` or None
            ``None`` until :meth:`fit_kg` has run.
        """
        return cast("KgFitResult | None", self._kg_fit_result)

    @property
    def kg_eval_result(self) -> KgEvalResult | None:
        """
        Return the metrics from the most recent KG evaluation.

        Stored on Session after :meth:`evaluate_kg` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgEvalResult` or None
            ``None`` until :meth:`evaluate_kg` has run.
        """
        return cast("KgEvalResult | None", self._kg_eval_result)

    @property
    def kg_score_result(self) -> ScoreTriplesResult | None:
        """Return the triple scores from the most recent score_triples call.

        Session-held result for ``kg_score_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ScoreTriplesResult | None", self._kg_score_result)

    @property
    def kg_predict_result(self) -> PredictLinksResult | None:
        """Return the link predictions from the most recent predict_links call.

        Session-held result for ``kg_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("PredictLinksResult | None", self._kg_predict_result)

    @property
    def kg_query_result(self) -> KgQueryResult | None:
        """
        Return the graph query from the most recent query_kg call.

        Stored on Session after :meth:`query_kg` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.kg.results.KgQueryResult` or None
            ``None`` until :meth:`query_kg` has run.
        """
        return cast("KgQueryResult | None", self._kg_query_result)

    def save_kg_bundle(self, path: str | Path) -> Path:
        """Persist the active KgPlan as ``buildml.kg_bundle.v1``.

        Session facade over :func:`buildml.session.kg_ops.save_kg_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.kg_ops.save_kg_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", kg_ops.save_kg_bundle_op(self, path=path))

    def load_kg_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a knowledge-graph bundle into this Session.

        Session facade over :func:`buildml.session.kg_ops.load_kg_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with KgPlan attached for chaining.

        See Also
        --------
        :func:`buildml.session.kg_ops.load_kg_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", kg_ops.load_kg_bundle_op(self, path=path, trusted=trusted))
