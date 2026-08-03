"""Session mixin: graph domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import graph_ops
from buildml.session.mixins._shared import *  # noqa: F403


class GraphSessionMixin:
    """Public Session methods for the graph domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _graph_eval_result: Any
        _graph_fit_result: Any
        _graph_plan: Any
        _graph_predict_result: Any
        _graph_spec: Any

    def set_graph(
        self,
        edges: Any,
        *,
        source_col: str = "source",
        target_col: str = "target",
        node_id_col: str = "node_id",
        directed: bool = False,
    ) -> GraphSpec:
        """Attach an edge list to the Session with dataset rows as nodes.

        Session facade over :func:`buildml.session.graph_ops.set_graph_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        GraphSpec
            Validated graph specification stored on Session as ``_graph_spec``.

        See Also
        --------
        :func:`buildml.session.graph_ops.set_graph_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("GraphSpec", graph_ops.set_graph_op(
            self,
            edges,
            source_col=source_col,
            target_col=target_col,
            node_id_col=node_id_col,
            directed=directed,
        ))

    def fit_graph(
        self,
        *,
        method: GraphMethod = "classical",
        task: GraphTask = "node_classification",
        mode: GraphMode = "inductive",
        columns: Sequence[str] | None = None,
        classical_estimator: ClassicalEstimator = "logistic_regression",
        hidden_dim: int = 32,
        n_layers: int = 2,
        epochs: int = 80,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        dropout: float = 0.1,
        random_state: int | None = 0,
        include_graph_metrics: bool = True,
        pyg_model: PyGModel = "gcn",
        heads: int = 4,
    ) -> GraphFitResult:
        """Fit graph node classification on Session train nodes.

        Session facade over :func:`buildml.session.graph_ops.fit_graph_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        GraphFitResult
            Serializable fit summary including method and mode disclosures.

        See Also
        --------
        :func:`buildml.session.graph_ops.fit_graph_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("GraphFitResult", graph_ops.fit_graph_op(
            self,
            method=method,
            task=task,
            mode=mode,
            columns=columns,
            classical_estimator=classical_estimator,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            random_state=random_state,
            include_graph_metrics=include_graph_metrics,
            pyg_model=pyg_model,
            heads=heads,
        ))

    def predict_graph(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> GraphPredictResult:
        """Predict node labels with the fitted GraphPlan on a partition.

        Session facade over :func:`buildml.session.graph_ops.predict_graph_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        GraphPredictResult
            Node predictions and optional probabilities for the partition.

        See Also
        --------
        :func:`buildml.session.graph_ops.predict_graph_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("GraphPredictResult", graph_ops.predict_graph_op(self, partition=partition))

    def evaluate_graph(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> GraphEvalResult:
        """Evaluate node classification on a holdout graph partition.

        Session facade over :func:`buildml.session.graph_ops.evaluate_graph_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        GraphEvalResult
            Classification metrics for nodes in the partition.

        See Also
        --------
        :func:`buildml.session.graph_ops.evaluate_graph_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("GraphEvalResult", graph_ops.evaluate_graph_op(self, partition=partition))

    @property
    def graph_spec(self) -> GraphSpec | None:
        """Return the graph specification attached by the most recent set_graph call.

        Session-held result for ``graph_spec``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("GraphSpec | None", self._graph_spec)

    @property
    def graph_plan(self) -> GraphPlan | None:
        """Return the graph plan built by the most recent graph fit.

        Session-held result for ``graph_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("GraphPlan | None", self._graph_plan)

    @property
    def graph_fit_result(self) -> GraphFitResult | None:
        """
        Return the report from the most recent graph fit.

        Stored on Session after :meth:`fit_graph` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.graph.results.GraphFitResult` or None
            ``None`` until :meth:`fit_graph` has run.
        """
        return cast("GraphFitResult | None", self._graph_fit_result)

    @property
    def graph_predict_result(self) -> GraphPredictResult | None:
        """Return the node predictions from the most recent graph scoring call.

        Session-held result for ``graph_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("GraphPredictResult | None", self._graph_predict_result)

    @property
    def graph_eval_result(self) -> GraphEvalResult | None:
        """Return the metrics from the most recent graph evaluation.

        Session-held result for ``graph_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("GraphEvalResult | None", self._graph_eval_result)

    def save_graph_bundle(self, path: str | Path) -> Path:
        """Persist the active GraphPlan as ``buildml.graph_bundle.v1``.

        Session facade over :func:`buildml.session.graph_ops.save_graph_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.graph_ops.save_graph_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", graph_ops.save_graph_bundle_op(self, path=path))

    def load_graph_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a graph bundle into this Session.

        Session facade over :func:`buildml.session.graph_ops.load_graph_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with GraphPlan and GraphSpec attached for chaining.

        See Also
        --------
        :func:`buildml.session.graph_ops.load_graph_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", graph_ops.load_graph_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def graph_capability_matrix() -> dict[str, Any]:
        """
        Report which graph machine-learning backends are available here.

        Call before :meth:`fit_graph` to see whether PyTorch Geometric, DGL, or
        sklearn graph kernels imported successfully. Read-only — no dataset required.

        Returns
        -------
        dict[str, Any]
            Graph backends, tasks, and install hints from
            :func:`buildml.graph.catalog.graph_capability_matrix`.
        """
        from buildml.graph.catalog import graph_capability_matrix

        return cast("dict[str, Any]", graph_capability_matrix())
