"""Bridge to Stable-Baselines3 for deep reinforcement learning.

BuildML's own environment loops are deliberately simple: a linear policy trained
by REINFORCE, and a Q-table over discretised states. Both are readable and both
have hard ceilings. When a problem genuinely needs a neural policy, this adapter
hands off to Stable-Baselines3 rather than reimplementing PPO.

Three algorithms are exposed. **PPO** is the general-purpose default: on-policy,
stable across a wide range of problems, and forgiving of hyperparameters. **DQN**
is off-policy and reuses past experience, which makes it more sample-efficient
when interaction is expensive, but it only handles discrete actions and is
fussier to tune. **A2C** is the lightweight on-policy option, faster per step
than PPO and correspondingly less stable.

The adapter is thin by design. BuildML supplies the environment, the seed, and a
uniform result shape; Stable-Baselines3 does the learning. That keeps the surface
small and the behaviour identical to using the library directly.

Two limits are worth stating. The scope is small discrete-action environments,
for learning and for modest problems: not robotics, autonomous driving, or
multi-agent simulation. And ``act_sb3_observation`` cannot return true action
probabilities, because Stable-Baselines3 does not expose them uniformly across
algorithms; see that function for what it returns instead.

Requires ``buildml[rl-industry]``.

See Also
--------
buildml.rl.gym_reinforce : The transparent linear alternative.
buildml.rl.tabular : Value-based control without a neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium, require_stable_baselines3

Sb3Algorithm = Literal["ppo", "dqn", "a2c"]


@dataclass
class SB3PolicyWrapper:
    """Hold a Stable-Baselines3 model along with what BuildML needs to know.

    A raw SB3 model does not record which environment it was trained on, how
    wide its observations are, or how long it trained. Bundling that alongside
    the model is what lets a saved policy be validated and described later,
    rather than being an opaque object.

    Attributes
    ----------
    model:
        The trained Stable-Baselines3 model.
    env_id:
        The Gymnasium environment it was trained on.
    algorithm:
        ``'ppo'``, ``'dqn'``, or ``'a2c'``.
    obs_dim:
        The flattened observation width, used to reject mismatched
        observations before they reach the model.
    n_actions:
        How many discrete actions the environment offers.
    total_timesteps:
        The interaction budget it was trained with.
    disclosures:
        What the training run did and what its scope is.

    See Also
    --------
    train_sb3_policy : Produce one of these.
    """

    model: Any
    env_id: str
    algorithm: Sb3Algorithm
    obs_dim: int
    n_actions: int
    total_timesteps: int = 0
    disclosures: tuple[str, ...] = ()

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[int, np.ndarray | None]:
        """Ask the underlying model what to do in one state.

        Reshapes the observation into the batch form Stable-Baselines3 expects
        and narrows the action to a plain ``int``.

        Parameters
        ----------
        observation:
            One observation, flattened.
        deterministic:
            ``True`` (default) takes the model's best action. ``False`` samples,
            which is what training looked like.

        Returns
        -------
        int
            The chosen action index.
        numpy.ndarray or None
            The recurrent state, for policies that keep one. ``None`` for the
            feed-forward policies used here.
        """
        obs = np.asarray(observation, dtype=float).reshape(1, -1)
        action, state = self.model.predict(obs, deterministic=deterministic)
        return int(action), state


def _make_sb3_model(
    algorithm: Sb3Algorithm,
    env: Any,
    *,
    learning_rate: float,
    gamma: float,
    seed: int | None,
) -> Any:
    sb3 = require_stable_baselines3(feature=f"fit_rl SB3 {algorithm}")
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "seed": seed,
        "gamma": gamma,
    }
    if algorithm == "ppo":
        return sb3.PPO(learning_rate=learning_rate, **common)
    if algorithm == "dqn":
        return sb3.DQN(learning_rate=learning_rate, **common)
    if algorithm == "a2c":
        return sb3.A2C(learning_rate=learning_rate, **common)
    raise ValidationError(
        f"Unknown SB3 algorithm={algorithm!r}. Supported: ppo, dqn, a2c."
    )


def train_sb3_policy(
    *,
    env_id: str = "CartPole-v1",
    algorithm: Sb3Algorithm = "ppo",
    total_timesteps: int = 20_000,
    max_steps: int = 500,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    random_state: int | None = 0,
) -> tuple[SB3PolicyWrapper, dict[str, float], list[str], list[str]]:
    """Train a neural policy on a Gymnasium environment.

    Creates the environment, builds the requested algorithm, trains for the
    given interaction budget, then runs a short greedy probe so the returned
    metrics say something about the finished policy rather than about the
    training process.

    Parameters
    ----------
    env_id:
        The Gymnasium environment. Must have a discrete action space and a
        shaped observation space.
    algorithm:
        ``'ppo'`` (default) is the safe general choice. ``'dqn'`` reuses past
        experience and needs fewer interactions, at the cost of being harder to
        tune. ``'a2c'`` is faster per step and less stable.
    total_timesteps:
        The interaction budget: the single setting that most determines
        whether the policy learns anything. Deep RL needs far more steps than
        feels reasonable; 20,000 is a starting point, not a sufficient one.
    max_steps:
        Per-episode cap for the evaluation probe.
    learning_rate:
        Optimiser step size, passed through to the algorithm.
    gamma:
        Discount factor.
    random_state:
        Seed. Deep RL is notoriously seed-sensitive; two seeds can differ more
        than two algorithms, so a single run is an anecdote.

    Returns
    -------
    SB3PolicyWrapper
        The trained policy with its metadata.
    dict
        ``total_timesteps`` plus the probe's ``n_eval_episodes``,
        ``mean_return``, and ``std_return``.
    list of str
        Disclosures describing the run and its scope.
    list of str
        Warnings, including a note when returns suggest undertraining.

    Raises
    ------
    MissingExtraError
        If ``buildml[rl-industry]`` is not installed.
    ValidationError
        If the environment cannot be created, its action space is not discrete,
        its observation space has no shape, or the algorithm is unknown.

    Notes
    -----
    **The reported returns come from a ten-episode probe**, which is a small
    sample for something as variable as an RL return. Treat them as a smoke
    test; run :func:`evaluate_sb3_policy` with more episodes for a number worth
    quoting.

    **Low returns usually mean too few timesteps**, not a wrong algorithm.
    Increase the budget before changing anything else.

    See Also
    --------
    evaluate_sb3_policy : A proper measurement of the trained policy.
    """
    gymnasium = require_gymnasium(feature="fit_rl(mode='gym_sb3')")
    require_stable_baselines3(feature="fit_rl(mode='gym_sb3')")
    disclosures = [
        "SB3 industry path trains PPO/DQN/A2C on a Gymnasium env loop.",
        "Requires buildml[rl-industry] (stable-baselines3 + gymnasium).",
        "Honesty: small discrete-action env teaching: not MuJoCo/robotics/AV.",
        "Offline RL / batch-constrained Q-learning are out of scope here.",
        f"env_id={env_id!r}; algorithm={algorithm}; "
        f"total_timesteps={total_timesteps}.",
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
                "gym_sb3 requires a discrete action space (action_space.n)."
            )
        obs_shape = getattr(obs_space, "shape", None)
        if not obs_shape:
            raise ValidationError(
                "gym_sb3 requires a Box-like observation space with a shape."
            )
        obs_dim = int(np.prod(obs_shape))
        n_actions = int(act_space.n)

        model = _make_sb3_model(
            algorithm,
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            seed=random_state,
        )
        model.learn(total_timesteps=int(total_timesteps))

        # Quick in-env return probe for train metrics.
        eval_metrics = evaluate_sb3_policy(
            SB3PolicyWrapper(
                model=model,
                env_id=env_id,
                algorithm=algorithm,
                obs_dim=obs_dim,
                n_actions=n_actions,
                total_timesteps=total_timesteps,
            ),
            n_episodes=10,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=True,
        )
    finally:
        env.close()

    wrapper = SB3PolicyWrapper(
        model=model,
        env_id=env_id,
        algorithm=algorithm,
        obs_dim=obs_dim,
        n_actions=n_actions,
        total_timesteps=int(total_timesteps),
        disclosures=tuple(disclosures),
    )
    metrics = {
        "total_timesteps": float(total_timesteps),
        "n_eval_episodes": eval_metrics.get("n_eval_episodes", 0.0),
        "mean_return": eval_metrics.get("mean_return", float("nan")),
        "std_return": eval_metrics.get("std_return", float("nan")),
    }
    if (
        env_id.startswith("CartPole")
        and metrics["mean_return"] < 50.0
        and total_timesteps < 50_000
    ):
        warnings.append(
            "CartPole mean return is still low; increase total_timesteps: "
            "this is an honest small-env teaching loop, not a robotics product."
        )
    return wrapper, metrics, disclosures, warnings


def evaluate_sb3_policy(
    policy: SB3PolicyWrapper,
    *,
    env_id: str | None = None,
    n_episodes: int = 20,
    max_steps: int = 500,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """Run a trained policy for several episodes and see what it earns.

    Nothing is updated: the policy is fixed and simply executed, so the returns
    measure what it does rather than what it was doing while learning.

    Parameters
    ----------
    policy:
        The trained policy wrapper.
    env_id:
        Override the environment. Defaults to the one the policy was trained
        on; a different one measures transfer, not performance.
    n_episodes:
        How many episodes to run. Returns vary a great deal, so twenty is a
        reasonable floor rather than a generous sample.
    max_steps:
        Per-episode step cap.
    random_state:
        Seeds the rollouts, offset from the training seeds so evaluation does
        not replay the episodes the policy trained on.
    deterministic:
        ``True`` (default) evaluates the greedy policy, which is what you would
        deploy.

    Returns
    -------
    dict
        ``n_eval_episodes``, ``mean_return``, ``std_return``, ``min_return``,
        and ``max_return``.

    Raises
    ------
    MissingExtraError
        If ``buildml[rl-industry]`` is not installed.

    Notes
    -----
    **Read ``std_return`` and ``min_return`` alongside the mean.** A policy
    averaging 400 that occasionally scores 20 fails badly some of the time, and
    the mean alone conceals that. For anything that will actually be deployed,
    the worst case usually matters more than the average.

    See Also
    --------
    train_sb3_policy : Produce the policy.
    """
    gymnasium = require_gymnasium(feature="evaluate_rl(mode='gym_sb3')")
    resolved_env = env_id or policy.env_id
    env = gymnasium.make(resolved_env)
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
                action, _state = policy.model.predict(
                    flat.reshape(1, -1),
                    deterministic=deterministic,
                )
                step_out = env.step(int(action))
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


def act_sb3_observation(
    policy: SB3PolicyWrapper,
    observation: Any,
    *,
    deterministic: bool = True,
) -> tuple[int, tuple[float, ...]]:
    """Ask the policy what to do in one state.

    The single-step form used when serving a policy outside an environment
    loop. The observation is checked against the policy's expected width first,
    so a mismatched vector fails with a clear message rather than producing a
    plausible-looking wrong action.

    Parameters
    ----------
    policy:
        The trained policy wrapper.
    observation:
        One observation. Flattened, and its size must match
        ``policy.obs_dim``.
    deterministic:
        ``True`` (default) takes the policy's best action.

    Returns
    -------
    int
        The chosen action index.
    tuple of float
        A one-hot vector marking the chosen action: **not** action
        probabilities.

    Raises
    ------
    ValidationError
        If the observation's size does not match the policy's. This normally
        means it came from a different environment.

    Notes
    -----
    **The scores are one-hot, and that is a real limitation.** The other modes
    return genuine distributions, and code that reads confidence from
    :class:`~buildml.rl.results.RlActResult` scores will see false certainty
    here. Stable-Baselines3 does not expose action probabilities uniformly :
    DQN has Q-values rather than probabilities, and PPO's distribution is not
    surfaced by ``predict``: so a one-hot marker is returned rather than a
    number that would look like a probability without being one.

    See Also
    --------
    buildml.rl.gym_reinforce.act_gym_observation : Returns real probabilities.
    """
    flat = np.asarray(observation, dtype=float).reshape(-1)
    if flat.size != policy.obs_dim:
        raise ValidationError(
            f"Observation dim {flat.size} != policy.obs_dim={policy.obs_dim}."
        )
    action, _state = policy.predict(flat, deterministic=deterministic)
    # SB3 does not expose action probs for all algos; return one-hot-ish scores.
    scores = np.zeros(policy.n_actions, dtype=float)
    scores[int(action)] = 1.0
    return int(action), tuple(float(v) for v in scores.tolist())
