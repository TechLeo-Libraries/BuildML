"""Session mixin: rl domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import rl_ops
from buildml.session.mixins._shared import *  # noqa: F403


class RlSessionMixin:
    """Public Session methods for the rl domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _imitation_eval_result: Any
        _imitation_fit_result: Any
        _imitation_plan: Any
        _imitation_predict_result: Any
        _rl_act_result: Any
        _rl_eval_result: Any
        _rl_fit_result: Any
        _rl_plan: Any

    def fit_imitation(
        self,
        *,
        backend: str | None = None,
        task: ImitationTask | None = None,
        estimator: ImitationEstimator | None = None,
        method: str | None = None,
        columns: list[str] | None = None,
        action_column: str | None = None,
        env_id: str | None = None,
        n_epochs: int = 40,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
    ) -> ImitationFitResult:
        """Fit behavioral cloning on Session train demonstrations.

        Session facade over :func:`buildml.session.rl_ops.fit_imitation_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        ImitationFitResult
            Serializable fit summary including action-space disclosures.

        See Also
        --------
        :func:`buildml.session.rl_ops.fit_imitation_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ImitationFitResult", rl_ops.fit_imitation_op(
            self,
            backend=backend,
            task=task,
            estimator=estimator,
            method=method,
            columns=columns,
            action_column=action_column,
            env_id=env_id,
            n_epochs=n_epochs,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
        ))

    def predict_imitation_action(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> ImitationPredictResult:
        """Predict actions under the fitted BC policy.

        Session facade over :func:`buildml.session.rl_ops.predict_imitation_action_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        ImitationPredictResult
            Predicted actions and optional quality disclosures.

        See Also
        --------
        :func:`buildml.session.rl_ops.predict_imitation_action_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ImitationPredictResult", rl_ops.predict_imitation_action_op(self, partition=partition))

    def evaluate_imitation(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> ImitationEvalResult:
        """Evaluate BC against held-out demonstration actions.

        Session facade over :func:`buildml.session.rl_ops.evaluate_imitation_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        ImitationEvalResult
            Held-out action prediction metrics and disclosures.

        See Also
        --------
        :func:`buildml.session.rl_ops.evaluate_imitation_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ImitationEvalResult", rl_ops.evaluate_imitation_op(self, partition=partition))

    @property
    def imitation_plan(self) -> ImitationPlan | None:
        """Return the behavioral-cloning plan built by the most recent imitation fit.

        Session-held result for ``imitation_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ImitationPlan | None", self._imitation_plan)

    @property
    def imitation_fit_result(self) -> ImitationFitResult | None:
        """Return the report from the most recent imitation fit.

        Session-held result for ``imitation_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ImitationFitResult | None", self._imitation_fit_result)

    @property
    def imitation_eval_result(self) -> ImitationEvalResult | None:
        """Return the metrics from the most recent imitation evaluation.

        Session-held result for ``imitation_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ImitationEvalResult | None", self._imitation_eval_result)

    @property
    def imitation_predict_result(self) -> ImitationPredictResult | None:
        """Return the actions from the most recent imitation prediction call.

        Session-held result for ``imitation_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ImitationPredictResult | None", self._imitation_predict_result)

    def save_imitation_bundle(self, path: str | Path) -> Path:
        """Persist the active ImitationPlan as ``buildml.imitation_bundle.v1``.

        Session facade over :func:`buildml.session.rl_ops.save_imitation_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.rl_ops.save_imitation_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", rl_ops.save_imitation_bundle_op(self, path=path))

    def load_imitation_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an imitation bundle into this Session.

        Session facade over :func:`buildml.session.rl_ops.load_imitation_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with imitation plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.rl_ops.load_imitation_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rl_ops.load_imitation_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def rl_capability_matrix() -> dict[str, Any]:
        """Return the RL / imitation capability matrix for this installation.

        Session facade over :func:`buildml.session.rl_ops.rl_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported modes and methods.

        See Also
        --------
        :func:`buildml.session.rl_ops.rl_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", rl_ops.rl_capability_matrix_op())

    def fit_rl(
        self,
        *,
        backend: str | None = None,
        mode: RlMode | None = None,
        algorithm: BanditAlgorithm | str = "linucb",
        columns: list[str] | None = None,
        action_column: str | None = None,
        reward_column: str | None = None,
        alpha: float = 1.0,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        env_id: str = "CartPole-v1",
        n_episodes: int = 200,
        max_steps: int = 500,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        total_timesteps: int = 20_000,
        n_bins: int = 8,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> RlFitResult:
        """Fit a contextual bandit (core) or a Gymnasium env policy (``buildml[rl]``).

        Session facade over :func:`buildml.session.rl_ops.fit_rl_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        RlFitResult
            Serializable fit summary including mode and algorithm disclosures.

        See Also
        --------
        :func:`buildml.session.rl_ops.fit_rl_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RlFitResult", rl_ops.fit_rl_op(
            self,
            backend=backend,
            mode=mode,
            algorithm=algorithm,
            columns=columns,
            action_column=action_column,
            reward_column=reward_column,
            alpha=alpha,
            epsilon=epsilon,
            temperature=temperature,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            env_id=env_id,
            n_episodes=n_episodes,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            total_timesteps=total_timesteps,
            n_bins=n_bins,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        ))

    def act_rl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        observations: Sequence[Any] | Any | None = None,
        deterministic: bool = True,
        random_state: int | None = 0,
    ) -> RlActResult:
        """Choose actions under the fitted RL policy.

        Session facade over :func:`buildml.session.rl_ops.act_rl_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        RlActResult
            Selected actions and policy disclosures.

        See Also
        --------
        :func:`buildml.session.rl_ops.act_rl_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RlActResult", rl_ops.act_rl_op(
            self,
            partition=partition,
            observations=observations,
            deterministic=deterministic,
            random_state=random_state,
        ))

    def evaluate_rl(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        n_episodes: int | None = None,
        max_steps: int | None = None,
        random_state: int | None = 0,
        deterministic: bool = True,
    ) -> RlEvalResult:
        """Evaluate RL (offline bandit metrics or Gymnasium rollouts).

        Session facade over :func:`buildml.session.rl_ops.evaluate_rl_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        RlEvalResult
            Offline or env evaluation metrics and disclosures.

        See Also
        --------
        :func:`buildml.session.rl_ops.evaluate_rl_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("RlEvalResult", rl_ops.evaluate_rl_op(
            self,
            partition=partition,
            n_episodes=n_episodes,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=deterministic,
        ))

    @property
    def rl_plan(self) -> RlPlan | None:
        """Return the RL plan built by the most recent fit_rl call.

        Session-held result for ``rl_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("RlPlan | None", self._rl_plan)

    @property
    def rl_fit_result(self) -> RlFitResult | None:
        """
        Return the report from the most recent RL fit.

        Stored on Session after :meth:`fit_rl` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlFitResult` or None
            ``None`` until :meth:`fit_rl` has run.
        """
        return cast("RlFitResult | None", self._rl_fit_result)

    @property
    def rl_eval_result(self) -> RlEvalResult | None:
        """
        Return the metrics from the most recent RL evaluation.

        Stored on Session after :meth:`evaluate_rl` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlEvalResult` or None
            ``None`` until :meth:`evaluate_rl` has run.
        """
        return cast("RlEvalResult | None", self._rl_eval_result)

    @property
    def rl_act_result(self) -> RlActResult | None:
        """
        Return the actions from the most recent act_rl call.

        Stored on Session after :meth:`act_rl` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.rl.results.RlActResult` or None
            ``None`` until :meth:`act_rl` has run.
        """
        return cast("RlActResult | None", self._rl_act_result)

    def save_rl_bundle(self, path: str | Path) -> Path:
        """Persist the active RlPlan as ``buildml.rl_bundle.v1``.

        Session facade over :func:`buildml.session.rl_ops.save_rl_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.rl_ops.save_rl_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", rl_ops.save_rl_bundle_op(self, path=path))

    def load_rl_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an RL bundle into this Session.

        Session facade over :func:`buildml.session.rl_ops.load_rl_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with RL plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.rl_ops.load_rl_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rl_ops.load_rl_bundle_op(self, path=path, trusted=trusted))
