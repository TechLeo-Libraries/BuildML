"""Learn a policy directly, by making good actions more likely.

REINFORCE is the simplest thing that could possibly work in sequential
reinforcement learning, and understanding it makes every fancier method easier
to read.

The idea: run an episode, see what it earned, then adjust the policy so that the
actions taken in a good episode become more likely and those in a bad one become
less. There is no value function and no model of the environment: just the
policy, nudged by outcomes.

Two refinements make it usable. **Returns-to-go** credit each action with the
reward that came *after* it, not the whole episode's reward, since an action
cannot have caused what preceded it. **A mean baseline** subtracts the average
return before updating, so what matters is whether an action did better than
typical rather than whether the reward was positive. Without the baseline, an
environment where all rewards are positive reinforces every action, including
the bad ones.

The policy here is linear: action scores are a matrix times the observation,
passed through a softmax. That is deliberately modest. It cannot represent
anything a linear model cannot, and it will not solve a hard control problem :
but it needs no deep learning framework, it trains in seconds, and every
parameter is inspectable. Use ``'gym_sb3'`` when you need PPO or DQN on a real
task; use this to see the mechanism.

Requires ``buildml[rl]``. The imitation and bandit paths do not.

See Also
--------
buildml.rl.tabular : Value-based control, the other classical approach.
buildml.rl.adapters.stable_baselines3 : Deep RL, when linear is not enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium
from buildml.rl.features import softmax


@dataclass
class LinearSoftmaxPolicy:
    """A policy that scores actions linearly and picks among them by chance.

    Holds one weight vector per action. Scoring an observation is a matrix
    multiply; turning the scores into a distribution is a softmax. Sampling from
    that distribution is how the policy acts, and it is also where exploration
    comes from: nothing else in REINFORCE explores, so a policy that becomes
    confident too early stops learning.

    Attributes
    ----------
    n_actions:
        How many discrete actions the environment offers.
    obs_dim:
        The flattened observation width.
    learning_rate:
        Step size for policy updates. Too large and the policy swings between
        extremes; too small and it never leaves its initialisation. This is the
        setting most worth trying twice.
    gamma:
        Discount factor for returns-to-go. Near 1.0 credits an action with
        reward far in its future; lower values keep credit local, which is
        easier to learn from but short-sighted.
    weights:
        The ``(n_actions, obs_dim)`` parameter matrix, initialised to small
        random values. Exact zeros would make every action equiprobable and
        every gradient identical, so the policy would never differentiate.

    Notes
    -----
    **Linear means linear.** If the best action depends on an interaction
    between observation dimensions, this policy cannot represent it, however
    long it trains. That is a property of the model, not of the training run.

    See Also
    --------
    train_gym_reinforce : Fit one of these against an environment.
    """

    n_actions: int
    obs_dim: int
    learning_rate: float = 0.01
    gamma: float = 0.99
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.weights.size == 0:
            # Small init helps early exploration without torch.
            rng = np.random.default_rng(0)
            self.weights = rng.normal(
                scale=0.01, size=(self.n_actions, self.obs_dim)
            ).astype(float)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """Score each action for an observation, before normalising.

        The raw linear scores. Useful for inspection; :meth:`probs` is what
        acting uses.

        Parameters
        ----------
        obs:
            An observation, flattened to length ``obs_dim``.

        Returns
        -------
        numpy.ndarray
            One unnormalised score per action. These are on an arbitrary scale
           : only their differences matter.
        """
        x = np.asarray(obs, dtype=float).reshape(-1)
        return self.weights @ x

    def probs(self, obs: np.ndarray) -> np.ndarray:
        """Give the probability of choosing each action in this state.

        The policy itself, ``π(a|s)``. How spread out this distribution is tells
        you how much the policy is still exploring: near-uniform means it has
        not committed, near-one-hot means it has.

        Parameters
        ----------
        obs:
            An observation, flattened to length ``obs_dim``.

        Returns
        -------
        numpy.ndarray
            One probability per action, summing to 1.0.
        """
        return softmax(self.logits(obs))

    def act(
        self, obs: np.ndarray, *, rng: np.random.Generator, deterministic: bool = False
    ) -> int:
        """Choose one action for this observation.

        Samples from :meth:`probs` by default, which is what makes training
        work: a policy that always took its current best action would never
        discover a better one.

        Parameters
        ----------
        obs:
            An observation, flattened to length ``obs_dim``.
        rng:
            The generator to sample from. Required, so that the caller controls
            reproducibility across a whole rollout rather than per call.
        deterministic:
            ``True`` takes the most probable action instead of sampling. Use it
            when evaluating or serving; leave it ``False`` while training.

        Returns
        -------
        int
            The chosen action index.
        """
        probs = self.probs(obs)
        if deterministic:
            return int(np.argmax(probs))
        return int(rng.choice(self.n_actions, p=probs))

    def update_episode(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
    ) -> float:
        """Learn from one finished episode.

        Where REINFORCE actually happens. Each action is credited with the
        discounted reward that followed it, the episode's mean is subtracted as
        a baseline, and the weights move so that better-than-average actions
        become more probable in the states they were taken in.

        Parameters
        ----------
        observations:
            The states visited, in order.
        actions:
            The action taken in each.
        rewards:
            The reward received at each step.

        Returns
        -------
        float
            The undiscounted total reward for the episode: what you plot to
            see whether learning is happening.

        Notes
        -----
        **The baseline is what makes this stable.** Subtracting the mean return
        means an action is reinforced only if it did better than the episode's
        average, rather than merely earning positive reward. In an environment
        where every step pays +1, without a baseline every action is reinforced
        and the policy learns nothing useful.

        **Nothing is learned until the episode ends**, because returns-to-go
        cannot be computed before then. This is what makes REINFORCE
        high-variance: one lucky episode moves the weights as confidently as a
        genuinely good one.
        """
        if not rewards:
            return 0.0
        returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = float(r) + self.gamma * g
            returns.insert(0, g)
        ret_arr = np.asarray(returns, dtype=float)
        # Baseline = mean return (variance reduction lite).
        ret_arr = ret_arr - float(np.mean(ret_arr))
        for obs, action, gt in zip(observations, actions, ret_arr, strict=True):
            x = np.asarray(obs, dtype=float).reshape(-1)
            probs = self.probs(x)
            # ∇ log π(a|s) = x * (1_a - π)
            grad = np.zeros_like(self.weights)
            for a in range(self.n_actions):
                indicator = 1.0 if a == int(action) else 0.0
                grad[a] = x * (indicator - probs[a])
            self.weights = self.weights + self.learning_rate * float(gt) * grad
        return float(sum(rewards))


def train_gym_reinforce(
    *,
    env_id: str = "CartPole-v1",
    n_episodes: int = 200,
    max_steps: int = 500,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    random_state: int | None = 0,
) -> tuple[LinearSoftmaxPolicy, dict[str, float], list[str], list[str]]:
    """Run episodes against an environment and improve the policy after each.

    The training loop: reset, act until the episode ends or the step cap is
    reached, then update the policy from what happened. Repeated ``n_episodes``
    times.

    Parameters
    ----------
    env_id:
        The Gymnasium environment. Must have a discrete action space and a
        shaped observation space.
    n_episodes:
        How many episodes to train for. REINFORCE is sample-hungry; a few
        hundred is a starting point, not a guarantee.
    max_steps:
        Per-episode step cap, so an episode that never terminates cannot hang
        the run.
    learning_rate:
        Policy step size.
    gamma:
        Discount factor for returns-to-go.
    random_state:
        Seeds weight initialisation, action sampling, and each episode's
        environment reset.

    Returns
    -------
    LinearSoftmaxPolicy
        The trained policy.
    dict
        ``n_episodes``, ``mean_return`` over all episodes, ``last_return``, and
        ``mean_return_last_20``.
    list of str
        Disclosures describing the run and its scope.
    list of str
        Warnings, including a note when returns look too low to indicate
        learning.

    Raises
    ------
    MissingExtraError
        If ``buildml[rl]`` is not installed.
    ValidationError
        If the environment cannot be created, its action space is not discrete,
        or its observation space has no shape.

    Notes
    -----
    **Read ``mean_return_last_20``, not ``mean_return``.** The overall mean
    includes the early episodes when the policy was random, so it understates a
    policy that did learn. The trailing mean is where it ended up.

    **Training returns are not an evaluation.** They come from a policy that was
    still sampling and still changing. Use :func:`evaluate_gym_policy` for a
    clean measurement of the finished policy.

    See Also
    --------
    evaluate_gym_policy : Roll out the trained policy without learning.
    """
    gymnasium = require_gymnasium(feature="fit_rl(mode='gym_reinforce')")
    disclosures = [
        "Gymnasium REINFORCE-lite trains a linear softmax policy in an env loop.",
        "This path requires buildml[rl] (gymnasium). Core BC/bandit paths do not.",
        "Honesty: small discrete-action env teaching loop: not MuJoCo/robotics.",
        f"env_id={env_id!r}; n_episodes={n_episodes}; gamma={gamma}.",
    ]
    warnings: list[str] = []
    try:
        env = gymnasium.make(env_id)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Failed to create Gymnasium env_id={env_id!r}: {exc}"
        ) from exc

    try:
        obs_space = env.observation_space
        act_space = env.action_space
        if not hasattr(act_space, "n"):
            raise ValidationError(
                "gym_reinforce requires a discrete action space (action_space.n)."
            )
        obs_shape = getattr(obs_space, "shape", None)
        if not obs_shape:
            raise ValidationError(
                "gym_reinforce requires a Box-like observation space with a shape."
            )
        obs_dim = int(np.prod(obs_shape))
        n_actions = int(act_space.n)
        policy = LinearSoftmaxPolicy(
            n_actions=n_actions,
            obs_dim=obs_dim,
            learning_rate=learning_rate,
            gamma=gamma,
        )
        # Re-seed weights with caller random_state.
        rng = np.random.default_rng(random_state)
        policy.weights = rng.normal(
            scale=0.01, size=(n_actions, obs_dim)
        ).astype(float)

        returns: list[float] = []
        for ep in range(int(n_episodes)):
            reset_out = env.reset(seed=None if random_state is None else int(random_state) + ep)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            observations: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            for _ in range(int(max_steps)):
                flat = np.asarray(obs, dtype=float).reshape(-1)
                action = policy.act(flat, rng=rng, deterministic=False)
                step_out = env.step(action)
                if len(step_out) == 5:
                    next_obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_obs, reward, done, _info = step_out
                observations.append(flat)
                actions.append(int(action))
                rewards.append(float(reward))
                obs = next_obs
                if done:
                    break
            ep_return = policy.update_episode(observations, actions, rewards)
            returns.append(ep_return)
    finally:
        env.close()

    metrics = {
        "n_episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "last_return": float(returns[-1]) if returns else float("nan"),
        "mean_return_last_20": float(np.mean(returns[-20:])) if returns else float("nan"),
    }
    if metrics["mean_return_last_20"] < 20 and env_id.startswith("CartPole"):
        warnings.append(
            "CartPole mean return (last 20) is still low; try more episodes "
            "or a different learning_rate: this is a lite teaching loop."
        )
    return policy, metrics, disclosures, warnings


def evaluate_gym_policy(
    policy: LinearSoftmaxPolicy,
    *,
    env_id: str,
    n_episodes: int = 20,
    max_steps: int = 500,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """Run a trained policy for a few episodes and see what it earns.

    Unlike training, nothing is updated: the policy is fixed and simply
    executed, so the returns measure what it does rather than what it was doing
    while learning.

    Parameters
    ----------
    policy:
        The trained policy.
    env_id:
        The environment to roll out in. Normally the one it was trained on :
        a different environment measures transfer, not performance.
    n_episodes:
        How many episodes to run. Returns vary a great deal episode to episode,
        so a handful gives a noisy mean.
    max_steps:
        Per-episode step cap.
    random_state:
        Seeds the rollouts. Offset from the training seeds, so evaluation does
        not replay the exact episodes the policy trained on.
    deterministic:
        ``True`` (default) always takes the most probable action, which is what
        you would deploy. ``False`` samples, which is what training looked like.

    Returns
    -------
    dict
        ``n_eval_episodes``, ``mean_return``, ``std_return``, ``min_return``,
        and ``max_return``.

    Raises
    ------
    MissingExtraError
        If ``buildml[rl]`` is not installed.

    Notes
    -----
    **Read ``std_return`` alongside the mean.** A policy averaging 200 with a
    standard deviation of 10 is reliable; one averaging 200 with a standard
    deviation of 150 succeeds sometimes and fails badly otherwise, and the two
    are indistinguishable from the mean alone.

    See Also
    --------
    train_gym_reinforce : Produce the policy.
    """
    gymnasium = require_gymnasium(feature="evaluate_rl(mode='gym_reinforce')")
    env = gymnasium.make(env_id)
    rng = np.random.default_rng(random_state)
    returns: list[float] = []
    try:
        for ep in range(int(n_episodes)):
            reset_out = env.reset(
                seed=None if random_state is None else int(random_state) + 10_000 + ep
            )
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            total = 0.0
            for _ in range(int(max_steps)):
                flat = np.asarray(obs, dtype=float).reshape(-1)
                action = policy.act(flat, rng=rng, deterministic=deterministic)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _info = step_out
                total += float(reward)
                if done:
                    break
            returns.append(total)
    finally:
        env.close()
    return {
        "n_eval_episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "std_return": float(np.std(returns)) if returns else float("nan"),
        "min_return": float(np.min(returns)) if returns else float("nan"),
        "max_return": float(np.max(returns)) if returns else float("nan"),
    }


def act_gym_observation(
    policy: LinearSoftmaxPolicy,
    observation: Any,
    *,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> tuple[int, tuple[float, ...]]:
    """Ask the policy what to do in one state, and how sure it is.

    The single-step form used when serving a policy outside an environment
    loop. Returns the action probabilities alongside the choice, so a caller can
    see whether the decision was clear-cut.

    Parameters
    ----------
    policy:
        The trained policy.
    observation:
        One observation. Flattened, and its size must match ``policy.obs_dim``.
    random_state:
        Seed, used only when sampling.
    deterministic:
        ``True`` (default) takes the most probable action.

    Returns
    -------
    int
        The chosen action index.
    tuple of float
        The action probabilities, summing to 1.0.

    Raises
    ------
    ValidationError
        If the observation's size does not match the policy's. This normally
        means the observation came from a different environment.

    Notes
    -----
    **The probabilities are the interesting part.** ``(0.26, 0.25, 0.25, 0.24)``
    means the policy has essentially no opinion and the returned action is close
    to arbitrary: worth knowing before acting on it.

    See Also
    --------
    buildml.rl.act.act_rl : The Session-level entry point.
    """
    rng = np.random.default_rng(random_state)
    flat = np.asarray(observation, dtype=float).reshape(-1)
    if flat.size != policy.obs_dim:
        raise ValidationError(
            f"Observation dim {flat.size} != policy.obs_dim={policy.obs_dim}."
        )
    probs = policy.probs(flat)
    action = policy.act(flat, rng=rng, deterministic=deterministic)
    return int(action), tuple(float(p) for p in probs.tolist())
