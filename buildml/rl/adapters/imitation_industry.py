"""Bridge to the ``imitation`` library for neural cloning.

The scikit-learn cloning path fits a logistic regression or a boosted tree,
which is enough for most tabular demonstration data. This adapter offers two
alternatives for when it is not.

**``bc_mlp``** is the same idea with a neural policy: a multi-layer network
mapping state to action, trained by supervised learning on the demonstrations.
Worth reaching for when the mapping is genuinely non-linear and there are enough
demonstrations to fit a network without memorising them. On ordinary tabular
data, the boosted-tree default is usually competitive and far cheaper.

**``gail_lite``** is a different thing entirely. Rather than matching actions
row by row, generative adversarial imitation learning trains a discriminator to
tell the demonstrator's behaviour from the policy's, and trains the policy to
fool it. It needs a live environment, because the policy must act to be judged :
and that is what lets it address the compounding-error problem that plain
cloning cannot: the policy is evaluated in the states it actually reaches, not
only in the states the demonstrator visited.

The ``lite`` is meant literally. GAIL is adversarial training, which is unstable
at the best of times, and the budgets here are small. Expect more variance than
from cloning, and treat a good result as encouraging rather than settled.

Requires ``buildml[rl-industry]``.

See Also
--------
buildml.rl.imitation : The always-available scikit-learn path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium, require_imitation, require_stable_baselines3


@dataclass
class TabularMlpPolicy:
    """Make a neural policy look like the estimator the rest of the code expects.

    Both methods here produce a Stable-Baselines3 policy, whose ``predict``
    signature and return shape differ from scikit-learn's. This wrapper adapts
    it, so that :mod:`buildml.rl.imitation` can treat a neural policy and a
    boosted tree identically.

    Attributes
    ----------
    model:
        The trained Stable-Baselines3 policy.
    obs_dim:
        The state-feature width.
    n_actions:
        How many discrete actions the policy can produce.
    method:
        ``'bc_mlp'`` or ``'gail_lite'``.
    classes_:
        The action vocabulary, when one was carried through.

    See Also
    --------
    fit_tabular_bc_mlp : Produce one by supervised cloning.
    fit_tabular_gail_lite : Produce one adversarially.
    """

    model: Any
    obs_dim: int
    n_actions: int
    method: str
    classes_: tuple[Any, ...] | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Choose an action for each row of a state matrix.

        The scikit-learn-shaped entry point: a 2-D array in, a 1-D array of
        action codes out. Acting is always deterministic here, since a cloned
        policy is being asked what the demonstrator would do rather than
        exploring.

        Parameters
        ----------
        x:
            A ``(n_rows, obs_dim)`` state matrix, or a single 1-D state, which
            is treated as one row.

        Returns
        -------
        numpy.ndarray
            One integer action code per row. Decode these with
            :func:`~buildml.rl.features.decode_discrete_actions` to recover the
            original action labels.
        """
        obs = np.asarray(x, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        actions, _states = self.model.predict(obs, deterministic=True)
        return np.asarray(actions).reshape(-1)


def _build_spaces(obs_dim: int, n_actions: int) -> tuple[Any, Any]:
    gymnasium = require_gymnasium(feature="imitation industry BC/GAIL")
    spaces = gymnasium.spaces
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32,
    )
    act_space = spaces.Discrete(n_actions)
    return obs_space, act_space


def fit_tabular_bc_mlp(
    x: np.ndarray,
    y_codes: np.ndarray,
    *,
    n_actions: int,
    n_epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    random_state: int | None = 0,
) -> tuple[TabularMlpPolicy, float, list[str], list[str]]:
    """Clone demonstrations with a neural policy instead of a linear one.

    Wraps the demonstrations as single-step transitions and trains a
    multi-layer policy on them. Supervised learning throughout: there is no
    environment and no reward, just states paired with the actions taken in
    them.

    Parameters
    ----------
    x:
        The state matrix, one row per demonstration.
    y_codes:
        Integer action codes, aligned with the rows.
    n_actions:
        How many distinct actions exist. Defines the policy's output width.
    n_epochs:
        Passes over the demonstrations. Too many and the network memorises
        rather than generalises.
    batch_size:
        Rows per gradient step.
    learning_rate:
        Optimiser step size.
    random_state:
        Seed for initialisation and shuffling.

    Returns
    -------
    TabularMlpPolicy
        The trained policy, wrapped for the scikit-learn-shaped interface.
    float
        In-sample agreement with the demonstrator.
    list of str
        Disclosures describing the run and its scope.
    list of str
        Warnings. Empty in practice for this path.

    Raises
    ------
    MissingExtraError
        If ``buildml[rl-industry]`` is not installed.

    Notes
    -----
    **The demonstrations are presented as one-step episodes**, each marked
    ``done``. That is accurate for tabular data, where rows are independent
    situations rather than a trajectory: but it also means no sequential
    structure is available to learn from, even if your rows happen to have some.

    **The returned score is in-sample and will be high.** A network with enough
    capacity can fit almost any set of demonstrations. Judge the policy with
    :func:`~buildml.rl.imitation.evaluate_imitation` on holdout rows.

    See Also
    --------
    fit_tabular_gail_lite : The adversarial alternative.
    """
    require_imitation(feature="fit_imitation(backend='industry')")
    require_stable_baselines3(feature="fit_imitation(backend='industry')")
    from imitation.algorithms import bc
    from imitation.data.types import Transitions

    obs_dim = int(x.shape[1])
    n = int(x.shape[0])
    obs_space, act_space = _build_spaces(obs_dim, n_actions)

    transitions = Transitions(
        obs=np.asarray(x, dtype=np.float32),
        acts=np.asarray(y_codes, dtype=np.int64),
        infos=np.array([{}] * n, dtype=object),
        next_obs=np.asarray(x, dtype=np.float32),
        dones=np.ones(n, dtype=bool),
    )

    rng = np.random.default_rng(random_state)
    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=transitions,
        rng=rng,
        policy_kwargs={"net_arch": [64, 64]},
    )
    bc_trainer.train(
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
    )
    policy_model = bc_trainer.policy
    pred = policy_model.predict(np.asarray(x, dtype=np.float32), deterministic=True)[0]
    pred = np.asarray(pred).reshape(-1)
    train_score = float(np.mean(pred == y_codes))

    disclosures = [
        "Industry imitation BC trains an MLP policy via imitation+SB3 on train demos.",
        "Requires buildml[rl-industry] (imitation + stable-baselines3 + gymnasium).",
        "Honesty: tabular BC from tables: not inverse RL, not DAgger, not robotics.",
        "Offline RL / batch RL are out of scope; this is supervised cloning depth.",
        f"method=bc_mlp; n_epochs={n_epochs}; obs_dim={obs_dim}; n_actions={n_actions}.",
    ]
    warnings: list[str] = []
    wrapper = TabularMlpPolicy(
        model=policy_model,
        obs_dim=obs_dim,
        n_actions=n_actions,
        method="bc_mlp",
    )
    return wrapper, train_score, disclosures, warnings


def fit_tabular_gail_lite(
    x: np.ndarray,
    y_codes: np.ndarray,
    *,
    env_id: str,
    n_actions: int,
    total_timesteps: int = 8_000,
    random_state: int | None = 0,
) -> tuple[TabularMlpPolicy, float, list[str], list[str]]:
    """Imitate by learning to be indistinguishable from the demonstrator.

    Trains a discriminator to separate demonstrated behaviour from the policy's,
    and a PPO policy to fool it. Because the policy must act in an environment
    to be judged, it is evaluated in the states it actually reaches: which is
    what lets adversarial imitation address the compounding-error problem that
    plain cloning cannot.

    The demonstrations must therefore be environment-compatible: same
    observation width, same action count. Both are checked before training
    starts.

    Parameters
    ----------
    x:
        The state matrix, one row per demonstration. Its width must match the
        environment's observation space.
    y_codes:
        Integer action codes, aligned with the rows.
    env_id:
        The Gymnasium environment the policy will act in. Required: without
        one there is nothing for the discriminator to judge.
    n_actions:
        How many distinct actions exist. Must match the environment's action
        space.
    total_timesteps:
        The interaction budget. Small by deliberate default; GAIL normally
        wants far more.
    random_state:
        Seed for the generator policy.

    Returns
    -------
    TabularMlpPolicy
        The trained policy.
    float
        In-sample agreement with the demonstrator.
    list of str
        Disclosures describing the run and its scope.
    list of str
        Warnings, always including a note about the small budget.

    Raises
    ------
    MissingExtraError
        If ``buildml[rl-industry]`` is not installed.
    ValidationError
        If the environment cannot be created, its action space is not discrete,
        or its observation width or action count disagrees with the
        demonstrations.

    Notes
    -----
    **Adversarial training is unstable, and this budget is small.** Two runs with
    different seeds can differ substantially, and a run may end mid-oscillation
    between generator and discriminator. Compare against
    :func:`fit_tabular_bc_mlp` before concluding that GAIL helped.

    **The score measures action agreement, which is not what GAIL optimises.**
    GAIL matches the *distribution* of behaviour, so a policy that reaches
    demonstrator-like states by a different route is a success by its own
    objective and scores poorly here. A low score is not necessarily a failure :
    but it does mean this number is the wrong one to judge it by.

    See Also
    --------
    fit_tabular_bc_mlp : Supervised cloning, stabler and much cheaper.
    """
    require_imitation(feature="fit_imitation(method='gail_lite')")
    sb3 = require_stable_baselines3(feature="fit_imitation(method='gail_lite')")
    gymnasium = require_gymnasium(feature="fit_imitation(method='gail_lite')")

    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.data.types import Transitions
    from imitation.rewards.reward_nets import BasicRewardNet

    obs_dim = int(x.shape[1])
    n = int(x.shape[0])

    try:
        env = gymnasium.make(env_id)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"gail_lite requires a valid Gymnasium env_id={env_id!r}: {exc}"
        ) from exc

    try:
        env_obs_dim = int(np.prod(env.observation_space.shape))
        if env_obs_dim != obs_dim:
            raise ValidationError(
                f"gail_lite demo obs_dim={obs_dim} must match env obs dim={env_obs_dim}."
            )
        if not hasattr(env.action_space, "n"):
            raise ValidationError(
                "gail_lite requires a discrete-action Gymnasium env."
            )
        if int(env.action_space.n) != int(n_actions):
            raise ValidationError(
                f"gail_lite demo n_actions={n_actions} must match env "
                f"n_actions={env.action_space.n}."
            )

        transitions = Transitions(
            obs=np.asarray(x, dtype=np.float32),
            acts=np.asarray(y_codes, dtype=np.int64),
            infos=np.array([{}] * n, dtype=object),
            next_obs=np.asarray(x, dtype=np.float32),
            dones=np.ones(n, dtype=bool),
        )

        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        gail = GAIL(
            demonstrations=transitions,
            demo_batch_size=min(32, max(1, n)),
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=2,
            venv=env,
            reward_net=reward_net,
            gen_algo=sb3.PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=random_state,
                n_steps=128,
                batch_size=64,
            ),
        )
        gail.train(total_timesteps=int(total_timesteps))

        policy_model = gail.gen_algo
        pred_raw = policy_model.predict(
            np.asarray(x, dtype=np.float32), deterministic=True
        )[0]
        pred = np.asarray(pred_raw).reshape(-1)
        train_score = float(np.mean(pred == y_codes))

        disclosures = [
            "GAIL-lite runs a small-budget adversarial imitation loop (honest lite).",
            "Requires buildml[rl-industry] and env-compatible demonstration rows.",
            "Honesty: teaching-depth GAIL: not robotics / AV / multi-agent sims.",
            "Offline RL disclosures: imitation from demos, not batch offline RL.",
            f"method=gail_lite; env_id={env_id!r}; total_timesteps={total_timesteps}.",
        ]
        warnings = [
            "GAIL-lite uses minimal timesteps; expect variance vs sklearn BC.",
        ]
        wrapper = TabularMlpPolicy(
            model=policy_model,
            obs_dim=obs_dim,
            n_actions=n_actions,
            method="gail_lite",
        )
        return wrapper, train_score, disclosures, warnings
    finally:
        env.close()
