"""Session mixin: metalearning domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import metalearning_ops
from buildml.session.mixins._shared import *  # noqa: F403


class MetalearningSessionMixin:
    """Public Session methods for the metalearning domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _metalearning_adapt_result: Any
        _metalearning_eval_result: Any
        _metalearning_fit_result: Any
        _metalearning_plan: Any

    def fit_metalearning(
        self,
        *,
        backend: str | None = None,
        method: MetaLearningMethod = "prototypical",
        task_column: str | None = None,
        columns: list[str] | None = None,
        n_way: int | None = None,
        k_shot: int = 5,
        n_query: int = 10,
        n_episodes: int = 20,
        base_estimator: MetaLearningBaseEstimator | str = "logistic_regression",
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        task_holdout_fraction: float = 0.25,
        meta_epochs: int = 40,
        inner_lr: float = 0.05,
        inner_steps: int = 5,
        meta_lr: float = 1e-3,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> MetaLearningFitResult:
        """Meta-train on episodic tasks carved from the train partition only.

        Session facade over :func:`buildml.session.metalearning_ops.fit_metalearning_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        MetaLearningFitResult
            Serializable fit summary including task counts and disclosures.

        See Also
        --------
        :func:`buildml.session.metalearning_ops.fit_metalearning_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MetaLearningFitResult", metalearning_ops.fit_metalearning_op(
            self,
            backend=backend,
            method=method,
            task_column=task_column,
            columns=columns,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes,
            base_estimator=base_estimator,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            task_holdout_fraction=task_holdout_fraction,
            meta_epochs=meta_epochs,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            meta_lr=meta_lr,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=device,
        ))

    def adapt_to_task(
        self,
        *,
        task_id: Any | None = None,
        partition: PartitionName = "train",
        support_frame: Any | None = None,
        max_support_per_class: int | None = None,
        random_state: int | None = 0,
    ) -> MetaAdaptResult:
        """Fast-adapt the meta-learner to one task's labeled support set.

        Session facade over :func:`buildml.session.metalearning_ops.adapt_to_task_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        MetaLearningAdaptResult
            Adapted predictions and support-set summary for the task.

        See Also
        --------
        :func:`buildml.session.metalearning_ops.adapt_to_task_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MetaAdaptResult", metalearning_ops.adapt_to_task_op(
            self,
            task_id=task_id,
            partition=partition,
            support_frame=support_frame,
            max_support_per_class=max_support_per_class,
            random_state=random_state,
        ))

    def evaluate_metalearning(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        k_shot: int | None = None,
        n_query: int | None = None,
        n_way: int | None = None,
        prefer_novel_tasks: bool = True,
        random_state: int | None = 0,
    ) -> MetaLearningEvalResult:
        """Run episodic holdout evaluation without meta-training on holdout.

        Session facade over :func:`buildml.session.metalearning_ops.evaluate_metalearning_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        MetaLearningEvalResult
            Episodic accuracy metrics on the holdout partition.

        See Also
        --------
        :func:`buildml.session.metalearning_ops.evaluate_metalearning_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MetaLearningEvalResult", metalearning_ops.evaluate_metalearning_op(
            self,
            partition=partition,
            k_shot=k_shot,
            n_query=n_query,
            n_way=n_way,
            prefer_novel_tasks=prefer_novel_tasks,
            random_state=random_state,
        ))

    @property
    def metalearning_plan(self) -> MetaLearningPlan | None:
        """Return the last meta-learning plan, if any.

        Stored on this Session after :meth:`fit_metalearning` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        MetaLearningPlan or None
            ``None`` before the first :meth:`fit_metalearning` call on this session.
        """
        return cast("MetaLearningPlan | None", self._metalearning_plan)

    @property
    def metalearning_fit_result(self) -> MetaLearningFitResult | None:
        """Return the last meta-learning fit result, if any.

        Session-held result for ``metalearning_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("MetaLearningFitResult | None", self._metalearning_fit_result)

    @property
    def metalearning_adapt_result(self) -> MetaAdaptResult | None:
        """Return the last meta-learning adaptation result, if any.

        Session-held result for ``metalearning_adapt_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("MetaAdaptResult | None", self._metalearning_adapt_result)

    @property
    def metalearning_eval_result(self) -> MetaLearningEvalResult | None:
        """Return the last meta-learning evaluation result, if any.

        Session-held result for ``metalearning_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("MetaLearningEvalResult | None", self._metalearning_eval_result)

    def save_metalearning_bundle(self, path: str | Path) -> Path:
        """Persist the active MetaLearningPlan as ``buildml.metalearning_bundle.v1``.

        Session facade over :func:`buildml.session.metalearning_ops.save_metalearning_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.metalearning_ops.save_metalearning_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", metalearning_ops.save_metalearning_bundle_op(self, path=path))

    def load_metalearning_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a meta-learning bundle into this Session.

        Session facade over :func:`buildml.session.metalearning_ops.load_metalearning_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with MetaLearningPlan attached for chaining.

        See Also
        --------
        :func:`buildml.session.metalearning_ops.load_metalearning_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", metalearning_ops.load_metalearning_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def metalearning_capability_matrix() -> dict[str, Any]:
        """
        Report which meta-learning backends and algorithms are available here.

        Call before few-shot or MAML-style fit methods to confirm learn2learn,
        torch meta modules, or sklearn fallbacks. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Meta-learning backends and install hints from
            :func:`buildml.metalearning.catalog.metalearning_capability_matrix`.
        """
        from buildml.metalearning.catalog import metalearning_capability_matrix

        return cast("dict[str, Any]", metalearning_capability_matrix())
