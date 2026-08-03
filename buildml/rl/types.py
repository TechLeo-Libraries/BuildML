"""Name the choices available in imitation learning and reinforcement learning.

Two families of decision-making live in this package, and the type aliases here
draw the line between them.

**Imitation learning** copies a demonstrator. You have a table of situations and
the action a person or an existing system took in each, and you fit a model that
predicts the action. It is ordinary supervised learning wearing different words:
the "label" is an action. It never asks whether the demonstrated action was
good — it only learns to reproduce it.

**Reinforcement learning** learns from outcomes. Instead of being told the right
action, the learner tries actions and sees rewards. That is a genuinely harder
problem, and BuildML addresses two tractable slices of it: contextual bandits
learned from logged ``(context, action, reward)`` rows, and episodic control
against a Gymnasium environment.

The aliases are :class:`~typing.Literal` types, so a mistyped algorithm name is
caught by a type checker before it reaches a validation error at runtime.

See Also
--------
buildml.rl.imitation : The behavioural cloning path.
buildml.rl.fit : The reinforcement learning path.
buildml.rl.catalog : What each option supports, as data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ImitationTask = Literal["classification", "regression"]

ImitationEstimator = Literal[
    "logistic_regression",
    "hist_gradient_boosting",
    "ridge",
    "hist_gradient_boosting_regressor",
]

ImitationBackend = Literal["sklearn", "industry"]

ImitationMethod = Literal["bc_mlp", "gail_lite"]

RlMode = Literal["contextual_bandit", "gym_reinforce", "tabular_q", "gym_sb3"]

RlBackend = Literal["sklearn", "native", "industry"]

BanditAlgorithm = Literal["linucb", "epsilon_greedy", "softmax"]

TabularAlgorithm = Literal[
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "double_q_learning",
]

Sb3Algorithm = Literal["ppo", "dqn", "a2c"]


@dataclass(slots=True)
class ImitationConfig:
    """The settings a behavioural cloning fit ran with, kept for the record.

    Written by :func:`~buildml.rl.imitation.fit_imitation` after it resolves
    every default, so it records what actually happened rather than what was
    asked for. Two fits that disagree can be compared field by field.

    Attributes
    ----------
    task:
        ``'classification'`` for discrete actions, ``'regression'`` for
        continuous ones. Inferred from the action column when not given.
    backend:
        ``'sklearn'`` for the always-available scikit-learn path, or
        ``'industry'`` for the neural policies behind ``buildml[rl-industry]``.
    estimator:
        Which scikit-learn model carries the policy.
    method:
        The industry method (``'bc_mlp'`` or ``'gail_lite'``), or ``None`` on
        the scikit-learn path.
    columns:
        The state features the policy reads, in order. The policy is only valid
        for frames carrying these columns.
    action_column:
        The column holding the demonstrated action. Defaults to the Dataset
        target.
    env_id:
        The Gymnasium environment, for methods that need one.
    n_epochs:
        Neural training passes. Ignored by scikit-learn estimators.
    random_state:
        Seed, so a fit can be reproduced.
    prefer_reduce_components:
        Whether an attached dimensionality reduction was used for the state
        features in place of raw columns.

    See Also
    --------
    buildml.rl.imitation.fit_imitation : Produces this configuration.
    """

    task: ImitationTask = "classification"
    backend: ImitationBackend = "sklearn"
    estimator: ImitationEstimator = "logistic_regression"
    method: ImitationMethod | None = None
    columns: tuple[str, ...] | None = None
    action_column: str | None = None
    env_id: str | None = None
    n_epochs: int = 40
    random_state: int | None = 0
    prefer_reduce_components: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as JSON-safe values.

        Written into the plan and into bundle manifests, so that a policy
        reloaded months later still says what it was fitted with.

        Returns
        -------
        dict
            Every field, with the column tuple widened to a list so the result
            survives a JSON round trip. Suitable for a bundle manifest or a
            history entry.
        """
        return {
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "method": self.method,
            "columns": None if self.columns is None else list(self.columns),
            "action_column": self.action_column,
            "env_id": self.env_id,
            "n_epochs": self.n_epochs,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
        }


@dataclass(slots=True)
class RlConfig:
    """The settings a reinforcement learning fit ran with, kept for the record.

    Covers all four modes in one dataclass, so most fields are irrelevant to any
    given fit — a contextual bandit ignores every episode setting, and a
    Gymnasium run ignores the logged-data columns. The grouping below says which
    fields matter when.

    Attributes
    ----------
    mode:
        Which problem is being solved. ``'contextual_bandit'`` learns from
        logged rows; ``'gym_reinforce'``, ``'tabular_q'``, and ``'gym_sb3'``
        learn by interacting with an environment.
    backend:
        ``'sklearn'`` for the bandit path, ``'native'`` for BuildML's own
        environment loops, ``'industry'`` for Stable-Baselines3.
    algorithm:
        The specific algorithm within the mode.
    columns, action_column, reward_column:
        Bandit only. The context features, the action that was taken, and the
        reward observed. All three must be present in the logged table.
    alpha:
        LinUCB only. How much unexplored actions are favoured. Higher explores
        more; lower exploits the current estimate harder.
    epsilon:
        Epsilon-greedy only. The probability of choosing at random.
    temperature:
        Softmax only. High values flatten the action distribution toward
        uniform, low values sharpen it toward the best-scoring action.
    random_state:
        Seed. RL is unusually seed-sensitive; two seeds can differ more than
        two algorithms.
    prefer_reduce_components:
        Whether an attached dimensionality reduction supplies the context
        features.
    env_id, n_episodes, max_steps:
        Environment modes. Which environment, how many episodes to train for,
        and the per-episode step cap that stops a non-terminating episode from
        hanging the run.
    learning_rate, gamma:
        Environment modes. Step size, and the discount factor — ``gamma`` near
        1.0 values distant reward almost as much as immediate reward, lower
        values make the policy short-sighted.
    hidden_seed:
        A second seed for policy initialisation, separate from the environment
        seed, so the two sources of randomness can be varied independently.
    total_timesteps:
        Stable-Baselines3 only. The interaction budget.
    n_bins:
        Tabular only. How finely each continuous observation dimension is
        discretised. The table grows as ``n_bins`` raised to the number of
        dimensions, so this is the setting that decides whether tabular control
        is feasible at all.
    epsilon_min, epsilon_decay:
        Tabular only. Exploration starts at ``epsilon`` and decays by
        ``epsilon_decay`` each episode, never below ``epsilon_min`` — early
        randomness to discover, later greediness to exploit.

    See Also
    --------
    buildml.rl.fit.fit_rl : Produces this configuration.
    """

    mode: RlMode = "contextual_bandit"
    backend: RlBackend = "sklearn"
    algorithm: BanditAlgorithm | TabularAlgorithm | Sb3Algorithm | str = "linucb"
    columns: tuple[str, ...] | None = None
    action_column: str | None = None
    reward_column: str | None = None
    alpha: float = 1.0
    epsilon: float = 0.1
    temperature: float = 1.0
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    # Gymnasium REINFORCE-lite
    env_id: str = "CartPole-v1"
    n_episodes: int = 200
    max_steps: int = 500
    learning_rate: float = 0.01
    gamma: float = 0.99
    hidden_seed: int | None = None
    total_timesteps: int = 20_000
    # Tabular TD control (Q-learning / SARSA family)
    n_bins: int = 8
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as JSON-safe values.

        Written into the plan and into bundle manifests, so a reloaded policy
        still carries the settings it was trained under.

        Returns
        -------
        dict
            Every field, including those the active mode ignores, with the
            column tuple widened to a list. Nothing is filtered by mode: a
            record that silently dropped fields would be harder to diff against
            another run.
        """
        return {
            "mode": self.mode,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "columns": None if self.columns is None else list(self.columns),
            "action_column": self.action_column,
            "reward_column": self.reward_column,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "env_id": self.env_id,
            "n_episodes": self.n_episodes,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "hidden_seed": self.hidden_seed,
            "total_timesteps": self.total_timesteps,
            "n_bins": self.n_bins,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
