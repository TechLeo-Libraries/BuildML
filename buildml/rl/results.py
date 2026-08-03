"""What the decision-making operations hand back.

Two shapes recur. A **plan** is a fitted policy: it holds live model objects, is
what you save and reload, and is what every later operation takes as input. A
**result** is a report about something that happened, holds no models, and is
safe to log or serialise.

Every one of them carries ``disclosures`` and ``warnings``, and in this domain
those are not decoration. An offline bandit estimate without the note that it is
offline invites being read as a measured result; a tabular return without the
note about unvisited states invites being read as a general capability. The
numbers alone do not carry their own caveats, so the caveats travel beside them.

``to_dict`` on each type returns JSON-safe values suitable for a manifest or a
history entry. Large payloads — chosen actions, score matrices — are summarised
by count rather than included.

See Also
--------
buildml.rl.checkpoint : Saving and reloading the plans.
buildml.rl.explain_hooks : Condensing these for history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ImitationPlan:
    """A fitted policy that reproduces demonstrated decisions.

    Holds the trained model plus everything needed to apply it consistently: the
    state columns in order, the action vocabulary, and the encoder that maps
    integer predictions back to recognisable actions. Save it with
    :func:`~buildml.rl.checkpoint.save_imitation_bundle`.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    estimator:
        Which model carries the policy.
    columns:
        The state features, in matrix order. Scoring frames must supply these.
    action_column:
        The column the demonstrated actions came from.
    n_train_rows:
        How many demonstrations were fitted on.
    classes_:
        The action vocabulary for classification, ``None`` for regression. The
        policy cannot produce an action outside it.
    backend:
        ``'sklearn'`` or ``'industry'``.
    method:
        The industry method, or ``None``.
    label_encoder_:
        Maps between action labels and integer codes.
    estimator_:
        The fitted model.
    disclosures:
        What the fit did, including the train-only contract.
    warnings:
        Anything that went not-quite-right.
    used_reduce_components:
        Whether reduction components stood in for raw columns.
    config:
        The resolved settings.
    train_score:
        In-sample agreement with the demonstrator. Not a quality measure — see
        :func:`~buildml.rl.imitation.fit_imitation`.

    Notes
    -----
    This is behavioural cloning and nothing more. It does not infer the reward
    the demonstrator was pursuing (inverse RL), does not query the demonstrator
    on states the policy reaches (DAgger), and carries no robotics stack.

    See Also
    --------
    buildml.rl.imitation.fit_imitation : Produces this plan.
    """

    task: str
    estimator: str
    columns: tuple[str, ...]
    action_column: str
    n_train_rows: int
    classes_: tuple[Any, ...] | None
    backend: str = "sklearn"
    method: str | None = None
    label_encoder_: Any = field(repr=False, default=None)
    estimator_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    train_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Describe the fitted policy as JSON-safe values.

        The metadata half of the plan, separated from the live model objects so
        it can go into a bundle manifest, a history entry, or a diff against
        another policy.

        Returns
        -------
        dict
            Task, backend, estimator, columns, action vocabulary, training
            size, in-sample score, disclosures, and configuration. The model
            objects and encoder are omitted — they are not JSON-representable,
            and a bundle is the way to persist them.
        """
        return {
            "kind": "imitation",
            "mode": "behavioral_cloning",
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "method": self.method,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "n_train_rows": self.n_train_rows,
            "classes": None if self.classes_ is None else list(self.classes_),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class ImitationFitResult:
    """A report on what a cloning fit saw and produced.

    The plan is what you use; this is what you read. It answers how many
    demonstrations there were, which actions appeared, and how closely the
    policy reproduces them in-sample.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    estimator:
        Which model was fitted.
    n_train_rows:
        Demonstrations used. Too few and the policy has memorised rather than
        learned.
    columns:
        The state features.
    action_column:
        Where the demonstrated actions came from.
    backend:
        ``'sklearn'`` or ``'industry'``.
    method:
        The industry method, or ``None``.
    classes:
        The action vocabulary, for classification.
    train_score:
        In-sample agreement with the demonstrator. High values are expected and
        say nothing about holdout behaviour.
    disclosures:
        What the fit did.
    warnings:
        Anything worth a second look.

    See Also
    --------
    ImitationEvalResult : The honest measurement, on holdout rows.
    """

    task: str
    estimator: str
    n_train_rows: int
    columns: tuple[str, ...]
    action_column: str
    backend: str = "sklearn"
    method: str | None = None
    classes: tuple[Any, ...] | None = None
    train_score: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the fit report as JSON-safe values.

        Nothing is dropped, because a cloning fit report is small enough to
        record whole — which makes two runs directly comparable field by field.

        Returns
        -------
        dict
            Every field, with tuples widened to lists.
        """
        return {
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "method": self.method,
            "n_train_rows": self.n_train_rows,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "classes": None if self.classes is None else list(self.classes),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ImitationEvalResult:
    """How closely the clone matched demonstrations it had not seen.

    Attributes
    ----------
    partition:
        Which rows were scored.
    task:
        ``'classification'`` or ``'regression'``.
    n_rows:
        How many demonstrations were compared.
    metrics:
        ``accuracy`` and ``macro_f1`` for discrete actions; ``rmse``, ``mae``,
        and ``r2`` for continuous ones.
    disclosures:
        Including that these rows never touched the fit.
    warnings:
        Anything that limits the reading, such as an empty partition.

    Notes
    -----
    **This measures similarity to the demonstrator, not quality.** A clone that
    agrees with a poor demonstrator 95% of the time scores 0.95 here. Whether
    that is good depends on the demonstrator, which no metric in this result can
    assess.

    See Also
    --------
    buildml.rl.imitation.evaluate_imitation : Produces this result.
    """

    partition: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as JSON-safe values.

        Metrics are kept in full rather than summarised, since a holdout score
        is the thing you will most want to look up later.

        Returns
        -------
        dict
            Partition, task, row count, the full metric mapping, disclosures,
            and warnings.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ImitationPredictResult:
    """The actions a cloned policy chose for a partition.

    Attributes
    ----------
    partition:
        Which rows were scored.
    task:
        ``'classification'`` or ``'regression'``.
    n_rows:
        How many rows were scored.
    actions:
        One action per row, in row order. Classification actions are the
        original labels, not internal codes; regression actions are floats.
    disclosures:
        Including that the policy was fitted on train alone.
    warnings:
        Anything worth noting, such as an empty partition.

    See Also
    --------
    buildml.rl.imitation.predict_imitation_action : Produces this result.
    ImitationEvalResult : The same actions, scored.
    """

    partition: str
    task: str
    n_rows: int
    actions: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the prediction run as JSON-safe values.

        Records that scoring happened and at what scale, not what it produced.
        A history entry should stay small however many rows were scored.

        Returns
        -------
        dict
            Partition, task, row count, how many actions were produced,
            disclosures, and warnings. The actions themselves are counted
            rather than included — read them from ``actions``.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_actions": len(self.actions),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RlPlan:
    """A fitted policy that chooses actions to earn reward.

    Covers all four modes, so which fields are populated depends on how it was
    fitted. A bandit plan carries columns, an action and reward column, an arm
    vocabulary, and a propensity model. An environment plan carries an
    ``env_id`` and ``obs_dim`` instead, with the tabular fields empty. Save it
    with :func:`~buildml.rl.checkpoint.save_rl_bundle`.

    Attributes
    ----------
    mode:
        Which of the four problems this policy solves.
    algorithm:
        The specific algorithm.
    columns:
        Bandit context features, in matrix order. Empty for environment modes.
    action_column, reward_column:
        Where the bandit's logged actions and rewards came from. ``None`` for
        environment modes.
    n_train_rows:
        Logged rows for bandits; episodes or timesteps for environment modes.
    n_arms:
        How many actions are available.
    arms_:
        The action vocabulary in code order. Encoding holdout actions against
        this is what keeps codes aligned with the fit.
    backend:
        ``'sklearn'``, ``'native'``, or ``'industry'``.
    label_encoder_:
        Maps between action labels and codes, for bandits.
    policy_:
        The fitted policy object, whose type follows the mode.
    propensity_model_:
        The logging-policy model that inverse propensity scoring needs.
        ``None`` when it could not be fitted, in which case IPS is unavailable.
    env_id, obs_dim:
        The environment and its observation width, for environment modes.
    disclosures:
        What the fit did and what its results mean.
    warnings:
        Anything that went not-quite-right, such as a failed propensity fit.
    used_reduce_components:
        Whether reduction components stood in for raw columns.
    config:
        The resolved settings.
    train_metrics:
        Per-mode training numbers. For bandits these describe the *log*, not
        the new policy.

    Notes
    -----
    This is a Session-shaped bandit and small-environment RL surface. It is not
    a MuJoCo, multi-agent, or robotics platform, and it is not batch offline RL.

    See Also
    --------
    buildml.rl.fit.fit_rl : Produces this plan.
    """

    mode: str
    algorithm: str
    columns: tuple[str, ...]
    action_column: str | None
    reward_column: str | None
    n_train_rows: int
    n_arms: int
    arms_: tuple[Any, ...]
    backend: str = "sklearn"
    label_encoder_: Any = field(repr=False, default=None)
    policy_: Any = field(repr=False, default=None)
    propensity_model_: Any = field(repr=False, default=None)
    env_id: str | None = None
    obs_dim: int | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Describe the fitted policy as JSON-safe values.

        The metadata half of the plan, separated from the live policy object so
        it can go into a bundle manifest or a history entry.

        Returns
        -------
        dict
            Mode, backend, algorithm, columns, action vocabulary, environment
            details, training metrics, disclosures, and configuration. The
            policy, encoder, and propensity model are omitted — a bundle is how
            those are persisted.
        """
        return {
            "kind": "rl",
            "mode": self.mode,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "reward_column": self.reward_column,
            "n_train_rows": self.n_train_rows,
            "n_arms": self.n_arms,
            "arms": list(self.arms_),
            "env_id": self.env_id,
            "obs_dim": self.obs_dim,
            "train_metrics": dict(self.train_metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RlFitResult:
    """A report on what an RL fit saw and produced.

    Attributes
    ----------
    mode:
        Which problem was solved.
    algorithm:
        The algorithm used.
    n_train_rows:
        Logged rows for bandits; episodes or timesteps otherwise.
    n_arms:
        How many actions are available.
    columns:
        Bandit context features. Empty for environment modes.
    backend:
        ``'sklearn'``, ``'native'``, or ``'industry'``.
    action_column, reward_column:
        The bandit's logged action and reward columns.
    env_id:
        The environment, for environment modes.
    train_metrics:
        Per-mode training numbers.
    disclosures:
        What the fit did.
    warnings:
        Anything worth a second look — a failed propensity fit shows up here,
        and it means IPS will be unavailable at evaluation.

    Notes
    -----
    **For bandits, ``train_metrics`` describes the log, not the policy.**
    ``mean_logged_reward`` is what the *existing* policy earned, and serves as
    the baseline the new one must beat. Nothing here says whether it does; that
    requires :func:`~buildml.rl.evaluate.evaluate_rl`.

    See Also
    --------
    RlEvalResult : What the policy is estimated or measured to earn.
    """

    mode: str
    algorithm: str
    n_train_rows: int
    n_arms: int
    columns: tuple[str, ...]
    backend: str = "sklearn"
    action_column: str | None = None
    reward_column: str | None = None
    env_id: str | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the fit report as JSON-safe values.

        Complete rather than summarised, so two fits — possibly in different
        modes — can be compared field by field afterwards.

        Returns
        -------
        dict
            Every field, with tuples widened to lists.
        """
        return {
            "mode": self.mode,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "n_train_rows": self.n_train_rows,
            "n_arms": self.n_arms,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "reward_column": self.reward_column,
            "env_id": self.env_id,
            "train_metrics": dict(self.train_metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RlEvalResult:
    """What a policy is estimated — or measured — to earn.

    The ``offline`` flag decides how everything else should be read, and is the
    first field to look at.

    Attributes
    ----------
    partition:
        The scored partition for bandits; ``None`` for environment rollouts,
        which have no partition.
    mode:
        Which problem the policy solves.
    n_rows:
        Rows scored, or episodes rolled out.
    metrics:
        For bandits: ``direct_method``, ``ips``, ``action_match_rate``,
        ``mean_logged_reward_on_match``, and ``mean_logged_reward``. For
        environment modes: mean and standard deviation of episode return, plus
        ``unseen_state_rate`` for tabular control.
    offline:
        ``True`` when the numbers are counterfactual estimates from a log;
        ``False`` when the policy was actually run.
    disclosures:
        Including, for bandits, that these are not A/B test results.
    warnings:
        Reasons to distrust the numbers — a missing propensity model, an empty
        partition, or a high unseen-state rate.

    Notes
    -----
    **``offline=True`` means nobody ran this policy.** The numbers are estimates
    of what would have happened, built from a log generated by a different
    policy. They are the best available evidence before deployment and no
    substitute for it.

    **Compare against ``mean_logged_reward``.** An estimate in isolation says
    nothing; the question is always whether it beats what you already run.

    See Also
    --------
    buildml.rl.evaluate : Both estimators, and when each misleads.
    """

    partition: str | None
    mode: str
    n_rows: int | None
    metrics: dict[str, float]
    offline: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as JSON-safe values.

        Carries the ``offline`` flag alongside the metrics, deliberately. A
        recorded number that has lost track of whether it was estimated or
        measured is a number that will eventually be misread.

        Returns
        -------
        dict
            Partition, mode, row count, the full metric mapping, the
            ``offline`` flag, disclosures, and warnings.
        """
        return {
            "partition": self.partition,
            "mode": self.mode,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "offline": self.offline,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RlActResult:
    """The actions a policy chose, and the scores behind them.

    Attributes
    ----------
    partition:
        The scored partition for bandits; ``None`` when acting on raw
        observations.
    mode:
        Which problem the policy solves.
    n_rows:
        How many situations were acted on.
    actions:
        One action per situation, in order. Bandit actions come back as the
        original labels; environment actions as integer indices.
    scores:
        One score tuple per situation, aligned with ``actions``. What the
        numbers mean depends on the mode — see
        :func:`~buildml.rl.act.act_rl`.
    disclosures:
        What produced the actions and how to read the scores.
    warnings:
        Anything worth noting.

    Notes
    -----
    **The scores are usually more informative than the actions.** Four arms
    scoring 0.51, 0.50, 0.50, 0.49 mean the policy has almost no preference, and
    the action it happened to pick is close to arbitrary. The action alone hides
    that.

    See Also
    --------
    buildml.rl.act.act_rl : Produces this result.
    """

    partition: str | None
    mode: str
    n_rows: int
    actions: tuple[Any, ...]
    scores: tuple[tuple[float, ...], ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the acting run as JSON-safe values.

        Records the scale of the run rather than its output. The score matrix
        alone is rows times arms, which has no business in a history entry.

        Returns
        -------
        dict
            Partition, mode, row count, how many actions and score rows were
            produced, disclosures, and warnings. Read the actions and scores
            from ``actions`` and ``scores``.
        """
        return {
            "partition": self.partition,
            "mode": self.mode,
            "n_rows": self.n_rows,
            "n_actions": len(self.actions),
            "n_score_rows": len(self.scores),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
