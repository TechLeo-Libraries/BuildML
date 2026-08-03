"""Session mixin: activelearning domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import activelearning_ops
from buildml.session.mixins._shared import *  # noqa: F403


class ActivelearningSessionMixin:
    """Public Session methods for the activelearning domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _activelearning_eval_result: Any
        _activelearning_fit_result: Any
        _activelearning_label_result: Any
        _activelearning_plan: Any
        _activelearning_query_result: Any

    def fit_active_learner(
        self,
        *,
        backend: ActiveLearningBackend | None = None,
        strategy: ActiveLearningStrategy = "margin",
        base_estimator: ActiveLearningEstimator = "logistic_regression",
        columns: list[str] | None = None,
        random_state: int | None = 0,
        batch_size: int = 5,
        label_budget: int | None = 50,
        unlabeled_marker: Any = None,
        prefer_reduce_components: bool = True,
        committee_size: int = 5,
        auto_refit: bool = True,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        mc_samples: int = 20,
        device: str = "cpu",
    ) -> ActiveLearningFitResult:
        """Fit or initialize the active learner on labeled train rows only.

        Session facade over :func:`buildml.session.activelearning_ops.fit_active_learner_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ActiveLearningFitResult
            Serializable fit summary including labeled/unlabeled pool sizes.

        See Also
        --------
        :func:`buildml.session.activelearning_ops.fit_active_learner_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ActiveLearningFitResult", activelearning_ops.fit_active_learner_op(
            self,
            backend=backend,
            strategy=strategy,
            base_estimator=base_estimator,
            columns=columns,
            random_state=random_state,
            batch_size=batch_size,
            label_budget=label_budget,
            unlabeled_marker=unlabeled_marker,
            prefer_reduce_components=prefer_reduce_components,
            committee_size=committee_size,
            auto_refit=auto_refit,
            epochs=epochs,
            learning_rate=learning_rate,
            mc_samples=mc_samples,
            device=device,
        ))

    def suggest_query(
        self,
        *,
        batch_size: int | None = None,
        strategy: ActiveLearningStrategy | None = None,
    ) -> ActiveLearningQueryResult:
        """Suggest unlabeled train-pool indices for human labeling without an oracle.

        Session facade over :func:`buildml.session.activelearning_ops.suggest_query_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ActiveLearningQueryResult
            Suggested train-pool indices, scores, and strategy metadata.

        See Also
        --------
        :func:`buildml.session.activelearning_ops.suggest_query_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ActiveLearningQueryResult", activelearning_ops.suggest_query_op(
            self,
            batch_size=batch_size,
            strategy=strategy,
        ))

    def label_rows(
        self,
        *,
        indices: list[Any] | tuple[Any, ...],
        labels: list[Any] | tuple[Any, ...],
        refit: bool | None = None,
    ) -> ActiveLearningLabelResult:
        """Incorporate user-provided labels on train-pool rows and optionally refit.

        Session facade over :func:`buildml.session.activelearning_ops.label_rows_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ActiveLearningLabelResult
            Labeling summary including whether a refit occurred.

        See Also
        --------
        :func:`buildml.session.activelearning_ops.label_rows_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ActiveLearningLabelResult", activelearning_ops.label_rows_op(
            self,
            indices=indices,
            labels=labels,
            refit=refit,
        ))

    def evaluate_active_learning(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        unlabeled_marker: Any = None,
    ) -> ActiveLearningEvalResult:
        """Evaluate the active learner on labeled rows of a holdout partition.

        Session facade over :func:`buildml.session.activelearning_ops.evaluate_active_learning_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ActiveLearningEvalResult
            Holdout metrics computed on labeled rows only.

        See Also
        --------
        :func:`buildml.session.activelearning_ops.evaluate_active_learning_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ActiveLearningEvalResult", activelearning_ops.evaluate_active_learning_op(
            self,
            partition=partition,
            unlabeled_marker=unlabeled_marker,
        ))

    @property
    def activelearning_plan(self) -> ActiveLearningPlan | None:
        """Return the last active-learning plan, if any.

        Session-held result for ``activelearning_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ActiveLearningPlan | None", self._activelearning_plan)

    @property
    def activelearning_fit_result(self) -> ActiveLearningFitResult | None:
        """Return the last active-learning fit result, if any.

        Session-held result for ``activelearning_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ActiveLearningFitResult | None", self._activelearning_fit_result)

    @property
    def activelearning_query_result(self) -> ActiveLearningQueryResult | None:
        """Return the last active-learning query result, if any.

        Session-held result for ``activelearning_query_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ActiveLearningQueryResult | None", self._activelearning_query_result)

    @property
    def activelearning_label_result(self) -> ActiveLearningLabelResult | None:
        """Return the last active-learning labeling result, if any.

        Session-held result for ``activelearning_label_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ActiveLearningLabelResult | None", self._activelearning_label_result)

    @property
    def activelearning_eval_result(self) -> ActiveLearningEvalResult | None:
        """Return the last active-learning evaluation result, if any.

        Session-held result for ``activelearning_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ActiveLearningEvalResult | None", self._activelearning_eval_result)

    def save_active_learning_bundle(self, path: str | Path) -> Path:
        """Persist the active-learning plan as ``buildml.activelearning_bundle.v1``.

        Session facade over :func:`buildml.session.activelearning_ops.save_active_learning_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.activelearning_ops.save_active_learning_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", activelearning_ops.save_active_learning_bundle_op(self, path=path))

    def load_active_learning_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an active-learning bundle into this Session.

        Session facade over :func:`buildml.session.activelearning_ops.load_active_learning_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with active-learning plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.activelearning_ops.load_active_learning_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", activelearning_ops.load_active_learning_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def activelearning_capability_matrix() -> dict[str, Any]:
        """
        Report which active-learning query strategies are available on this machine.

        Call before pool-based query loops to confirm modAL, sklearn uncertainty
        samplers, or native strategies on this install. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Query strategies, backends, and install hints from
            :func:`buildml.activelearning.catalog.activelearning_capability_matrix`.
        """
        from buildml.activelearning.catalog import activelearning_capability_matrix

        return cast("dict[str, Any]", activelearning_capability_matrix())
