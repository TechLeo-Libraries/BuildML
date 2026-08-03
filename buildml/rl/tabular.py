"""Tabular temporal-difference control (Q-learning family) — behind buildml[rl].

Foundational value-based RL: the agent learns an action-value table ``Q[s, a]``
by bootstrapping from its own estimates instead of learning a parameterised
policy directly (``gym_reinforce``) or a per-arm reward model
(``contextual_bandit``).

Algorithms
----------
``q_learning``
    Off-policy TD control. Bootstraps from ``max_a' Q[s', a']`` regardless of
    the action the behaviour policy actually takes.
``sarsa``
    On-policy TD control. Bootstraps from ``Q[s', a']`` where ``a'`` is the
    action the epsilon-greedy behaviour policy will actually take.
``expected_sarsa``
    On-policy TD control with the expectation over the behaviour policy,
    ``Σ_a' π(a'|s') Q[s', a']`` — lower variance than SARSA.
``double_q_learning``
    Two tables with cross-evaluated bootstrapping; removes the maximisation
    bias of vanilla Q-learning.

Continuous ``Box`` observations are handled by an explicit, inspectable
uniform-bin discretizer whose ranges come from the declared space bounds where
finite and from a seeded random-policy probe where the space is unbounded.

Honesty: small discrete-control teaching loops (FrozenLake / Taxi /
CliffWalking / discretised CartPole). Tabular methods do not scale to
high-dimensional observations — that is exactly what function approximation
(``gym_reinforce``) and deep value methods (``gym_sb3`` DQN) exist for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium

TabularAlgorithm = Literal[
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "double_q_learning",
]

TABULAR_ALGORITHMS: tuple[str, ...] = (
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "double_q_learning",
)

#: Refuse to allocate tables larger than this; suggest fewer bins instead.
MAX_TABULAR_STATES = 500_000

__all__ = [
    "MAX_TABULAR_STATES",
    "TABULAR_ALGORITHMS",
    "ObservationDiscretizer",
    "TabularAlgorithm",
    "TabularValuePolicy",
    "act_tabular_observation",
    "build_discretizer",
    "epsilon_greedy_probabilities",
    "evaluate_tabular_policy",
    "resolve_tabular_algorithm",
    "train_tabular_control",
]


def resolve_tabular_algorithm(algorithm: str | None) -> str:
    """Normalise a user-supplied tabular algorithm name to its canonical form.

    Called at the top of :func:`train_tabular_control` so that a bad algorithm
    name fails before an environment is created or a table is allocated. Case
    and hyphens are accepted (``"Double-Q-Learning"``) because these names are
    usually typed by hand rather than copied from the catalog.

    Parameters
    ----------
    algorithm : str or None
        Requested algorithm. ``None`` selects ``"q_learning"``, the default
        off-policy method and the one most readers will have been taught first.

    Returns
    -------
    str
        A member of :data:`TABULAR_ALGORITHMS`, safe to dispatch on.

    Raises
    ------
    ValidationError
        If the name is not a tabular TD-control algorithm. Deep or policy
        gradient names such as ``"ppo"`` land here; they belong to
        ``mode='gym_sb3'`` or ``mode='gym_reinforce'`` instead.

    See Also
    --------
    buildml.rl.catalog.list_rl_algorithms : Algorithms available per mode.
    """
    key = str(algorithm or "q_learning").lower().replace("-", "_")
    if key not in TABULAR_ALGORITHMS:
        raise ValidationError(
            f"algorithm='{key}' is not a tabular TD-control algorithm. "
            f"Choose from {list(TABULAR_ALGORITHMS)}."
        )
    return key


def epsilon_greedy_probabilities(
    q_row: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Compute the action distribution an epsilon-greedy policy would use.

    Expected SARSA needs the behaviour policy written out as probabilities so
    it can average ``Q[s', a']`` over them instead of sampling one action. The
    same distribution is what :meth:`TabularValuePolicy.action_probabilities_for_state`
    exposes for inspection after training.

    Parameters
    ----------
    q_row : np.ndarray
        Action values for one state, i.e. a row of the Q-table. Flattened
        before use, so a ``(1, n_actions)`` row is accepted.
    epsilon : float
        Exploration rate. ``0.0`` gives a purely greedy distribution and
        ``1.0`` a uniform one; values are clipped into ``[0, 1]`` rather than
        rejected, because callers pass decayed schedules.

    Returns
    -------
    np.ndarray
        Probabilities over actions, summing to 1.

    Raises
    ------
    ValidationError
        If ``q_row`` is empty, which would leave no action to assign mass to.

    Notes
    -----
    Ties among greedy actions split the greedy mass evenly. This matters at
    initialisation: an all-zero table makes every action greedy, and giving the
    whole ``1 - epsilon`` to action 0 would bias Expected SARSA's target from
    the first update onwards.
    """
    q = np.asarray(q_row, dtype=float).reshape(-1)
    n_actions = int(q.size)
    if n_actions == 0:
        raise ValidationError("Cannot build action probabilities for 0 actions.")
    eps = float(np.clip(epsilon, 0.0, 1.0))
    probs = np.full(n_actions, eps / n_actions, dtype=float)
    greedy = np.flatnonzero(q == q.max())
    probs[greedy] += (1.0 - eps) / float(greedy.size)
    return probs


@dataclass
class ObservationDiscretizer:
    """Map an environment observation to a single integer state index.

    ``kind='discrete'``
        Pass-through for Gymnasium ``Discrete`` observation spaces.
    ``kind='box'``
        Uniform per-dimension binning combined as a mixed-radix index.
    """

    kind: Literal["discrete", "box"]
    n_states: int
    obs_dim: int = 1
    n_bins: int = 0
    bin_edges: tuple[tuple[float, ...], ...] = ()
    low: tuple[float, ...] = ()
    high: tuple[float, ...] = ()
    bound_sources: tuple[str, ...] = ()

    def index(self, observation: Any) -> int:
        """Convert one raw environment observation into a Q-table row number.

        Every read and write of the Q-table goes through this method, so it is
        the single place where "what the environment emitted" becomes "which
        state the agent thinks it is in". Called once per environment step
        during :func:`train_tabular_control` and once per row during
        :func:`act_tabular_observation`.

        Parameters
        ----------
        observation : Any
            An observation from the environment: a scalar for ``Discrete``
            spaces, or an array of length :attr:`obs_dim` for ``Box`` spaces.

        Returns
        -------
        int
            A state index in ``[0, n_states)``.

        Raises
        ------
        ValidationError
            If a discrete observation is not scalar or falls outside
            ``[0, n_states)``, or if a box observation has the wrong length.
            Both indicate the discretizer was built for a different
            environment than the one being stepped.

        Notes
        -----
        Box observations are clipped into the modelled range instead of
        raising, because a value beyond the probed bounds is a normal event
        rather than a bug. Non-finite values are replaced by the midpoint of
        the range first; without that they would silently pile into the top
        bin and corrupt an otherwise usable state.
        """
        if self.kind == "discrete":
            flat = np.asarray(observation).reshape(-1)
            if flat.size != 1:
                raise ValidationError(
                    "Discrete observation spaces expect a scalar observation; "
                    f"got shape {np.shape(observation)!r}."
                )
            code = int(flat[0])
            if not 0 <= code < self.n_states:
                raise ValidationError(
                    f"Discrete observation {code} outside [0, {self.n_states})."
                )
            return code

        flat = np.asarray(observation, dtype=float).reshape(-1)
        if flat.size != self.obs_dim:
            raise ValidationError(
                f"Observation dim {flat.size} != discretizer obs_dim={self.obs_dim}."
            )
        low = np.asarray(self.low, dtype=float)
        high = np.asarray(self.high, dtype=float)
        # Non-finite observations would silently fall into the top bin; clamp
        # them to the modelled range instead.
        midpoint = 0.5 * (low + high)
        cleaned = np.where(np.isfinite(flat), flat, midpoint)
        cleaned = np.clip(cleaned, low, high)
        index = 0
        for dim, edges in enumerate(self.bin_edges):
            arr = np.asarray(edges, dtype=float)
            bucket = int(np.digitize(float(cleaned[dim]), arr))
            index = index * (int(arr.size) + 1) + bucket
        return int(index)

    def to_dict(self) -> dict[str, Any]:
        """Summarise the discretization as JSON-safe values for the RL plan.

        Stored under ``RlPlan.config['discretizer']`` by
        :meth:`buildml.Session.fit_rl` so that a saved bundle records how
        observations were bucketed, not just the resulting table.

        Returns
        -------
        dict of str to Any
            Space kind, state and dimension counts, bin count, per-dimension
            ranges, and where each range came from. Bin edges are omitted:
            they are derivable from the ranges and would dominate the plan.
        """
        return {
            "kind": self.kind,
            "n_states": self.n_states,
            "obs_dim": self.obs_dim,
            "n_bins": self.n_bins,
            "low": list(self.low),
            "high": list(self.high),
            "bound_sources": list(self.bound_sources),
        }


def _probe_observation_ranges(
    env: Any,
    *,
    obs_dim: int,
    episodes: int,
    max_steps: int,
    random_state: int | None,
) -> np.ndarray | None:
    """Collect observations under a random policy to bound unbounded dims."""
    try:
        env.action_space.seed(None if random_state is None else int(random_state))
    except Exception:  # noqa: BLE001 - seeding is best effort across env versions
        pass
    samples: list[np.ndarray] = []
    for episode in range(int(episodes)):
        seed = None if random_state is None else int(random_state) + 5_000 + episode
        reset_out = env.reset(seed=seed)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        samples.append(np.asarray(obs, dtype=float).reshape(-1))
        for _ in range(int(max_steps)):
            step_out = env.step(env.action_space.sample())
            if len(step_out) == 5:
                obs, _reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                obs, _reward, done, _info = step_out
            samples.append(np.asarray(obs, dtype=float).reshape(-1))
            if done:
                break
    if not samples:
        return None
    stacked = np.vstack([row for row in samples if row.size == obs_dim])
    if stacked.size == 0:
        return None
    return stacked


def build_discretizer(
    env: Any,
    *,
    n_bins: int = 8,
    probe_episodes: int = 20,
    probe_max_steps: int = 200,
    random_state: int | None = 0,
) -> tuple[ObservationDiscretizer, list[str]]:
    """Derive a state discretizer from an environment's observation space.

    Runs once at the start of :func:`train_tabular_control`, before the
    Q-table is allocated, because the table's row count is whatever this
    function decides. ``Discrete`` spaces pass through untouched; ``Box``
    spaces are cut into uniform bins per dimension and combined into a single
    mixed-radix index.

    Parameters
    ----------
    env : Any
        A constructed Gymnasium environment. It may be stepped here: unbounded
        dimensions are measured by rolling out a random policy, so pass the
        environment before training rather than one mid-episode.
    n_bins : int
        Bins per observation dimension for ``Box`` spaces. The table grows as
        ``n_bins ** obs_dim``, so raising this sharpens the agent's view of
        the state and multiplies the experience needed to fill the table.
        Ignored for ``Discrete`` spaces.
    probe_episodes : int
        Random-policy episodes used to measure dimensions the space declares
        as unbounded. More episodes give steadier ranges at a small one-off
        cost paid before learning starts.
    probe_max_steps : int
        Step cap per probe episode, so probing cannot hang on an environment
        that never terminates under random actions.
    random_state : int or None
        Seed for the probe. Fixing it makes the bin edges — and therefore the
        meaning of every state index — reproducible across runs.

    Returns
    -------
    tuple of (ObservationDiscretizer, list of str)
        The discretizer, and disclosure lines naming the bin count and where
        each dimension's range came from. The caller appends these to the
        fit's disclosures so the binning is never invisible.

    Raises
    ------
    ValidationError
        If the space is neither ``Discrete`` nor ``Box``-like, if it is
        ``MultiDiscrete``, if ``n_bins < 2``, or if the requested table would
        exceed :data:`MAX_TABULAR_STATES`.

    Notes
    -----
    **Scale:** declared bounds wider than ``1e6`` are treated as unbounded.
    Several Gymnasium spaces report the float32 maximum for dimensions that
    are conceptually infinite, and binning uniformly across that range would
    put every observation the agent ever sees into one middle bin.
    """
    obs_space = env.observation_space
    disclosures: list[str] = []

    if hasattr(obs_space, "n") and not getattr(obs_space, "shape", ()):
        n_states = int(obs_space.n)
        disclosures.append(
            f"Observation space is Discrete(n={n_states}); "
            "states are used directly with no discretization."
        )
        return (
            ObservationDiscretizer(
                kind="discrete",
                n_states=n_states,
                obs_dim=1,
            ),
            disclosures,
        )

    shape = getattr(obs_space, "shape", None)
    if not shape:
        raise ValidationError(
            "tabular_q requires a Discrete or Box-like observation space; "
            f"got {type(obs_space).__name__} without a usable shape."
        )
    if hasattr(obs_space, "nvec"):
        raise ValidationError(
            "tabular_q does not support MultiDiscrete observation spaces. "
            "Use a Discrete env (FrozenLake-v1 / Taxi-v3 / CliffWalking-v0) or "
            "a Box env, or switch to mode='gym_reinforce'."
        )

    bins = int(n_bins)
    if bins < 2:
        raise ValidationError("n_bins must be >= 2 for tabular_q discretization.")
    obs_dim = int(np.prod(shape))
    total_states = bins**obs_dim
    if total_states > MAX_TABULAR_STATES:
        raise ValidationError(
            f"tabular_q would allocate {total_states} states "
            f"(n_bins={bins} ** obs_dim={obs_dim}), above the "
            f"{MAX_TABULAR_STATES} guard. Lower n_bins, pick a Discrete env, or "
            "use mode='gym_reinforce' / mode='gym_sb3' for function approximation."
        )

    raw_low = np.asarray(getattr(obs_space, "low", np.full(obs_dim, -np.inf)), dtype=float)
    raw_high = np.asarray(getattr(obs_space, "high", np.full(obs_dim, np.inf)), dtype=float)
    raw_low = raw_low.reshape(-1)
    raw_high = raw_high.reshape(-1)
    if raw_low.size != obs_dim or raw_high.size != obs_dim:
        raw_low = np.full(obs_dim, -np.inf, dtype=float)
        raw_high = np.full(obs_dim, np.inf, dtype=float)

    declared_ok = np.isfinite(raw_low) & np.isfinite(raw_high) & (raw_high > raw_low)
    # Extremely wide declared bounds (float32 max on unbounded dims) make every
    # sample land in the middle bin; treat them as unbounded.
    declared_ok &= (raw_high - raw_low) < 1e6

    probe: np.ndarray | None = None
    if not bool(np.all(declared_ok)):
        probe = _probe_observation_ranges(
            env,
            obs_dim=obs_dim,
            episodes=probe_episodes,
            max_steps=probe_max_steps,
            random_state=random_state,
        )

    low = np.zeros(obs_dim, dtype=float)
    high = np.zeros(obs_dim, dtype=float)
    sources: list[str] = []
    for dim in range(obs_dim):
        if bool(declared_ok[dim]):
            low[dim] = float(raw_low[dim])
            high[dim] = float(raw_high[dim])
            sources.append("space_bounds")
            continue
        if probe is not None:
            lo = float(np.percentile(probe[:, dim], 1.0))
            hi = float(np.percentile(probe[:, dim], 99.0))
            if hi > lo:
                margin = 0.1 * (hi - lo)
                low[dim] = lo - margin
                high[dim] = hi + margin
                sources.append("random_policy_probe")
                continue
        low[dim] = -1.0
        high[dim] = 1.0
        sources.append("fallback_unit_range")

    bin_edges = tuple(
        tuple(float(edge) for edge in np.linspace(low[dim], high[dim], bins + 1)[1:-1])
        for dim in range(obs_dim)
    )
    disclosures.append(
        f"Box observations discretized into {bins} uniform bins per dimension "
        f"({obs_dim} dims → {total_states} tabular states)."
    )
    if "random_policy_probe" in sources:
        probed = [i for i, src in enumerate(sources) if src == "random_policy_probe"]
        disclosures.append(
            "Unbounded observation dimensions "
            f"{probed} were bounded by 1st/99th percentiles from a seeded "
            f"random-policy probe ({probe_episodes} episodes)."
        )
    if "fallback_unit_range" in sources:
        fallback = [i for i, src in enumerate(sources) if src == "fallback_unit_range"]
        disclosures.append(
            f"Observation dimensions {fallback} had no usable bounds or probe "
            "spread; they fall back to [-1, 1]."
        )
    return (
        ObservationDiscretizer(
            kind="box",
            n_states=int(total_states),
            obs_dim=obs_dim,
            n_bins=bins,
            bin_edges=bin_edges,
            low=tuple(float(v) for v in low.tolist()),
            high=tuple(float(v) for v in high.tolist()),
            bound_sources=tuple(sources),
        ),
        disclosures,
    )


@dataclass
class TabularValuePolicy:
    """Action-value table plus the discretizer that produced its state index."""

    n_actions: int
    n_states: int
    algorithm: str
    discretizer: ObservationDiscretizer
    gamma: float = 0.99
    learning_rate: float = 0.1
    env_id: str | None = None
    q_table: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    q_table_b: np.ndarray | None = None
    state_visits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        if int(self.n_actions) < 1:
            raise ValidationError("TabularValuePolicy requires n_actions >= 1.")
        if int(self.n_states) < 1:
            raise ValidationError("TabularValuePolicy requires n_states >= 1.")
        shape = (int(self.n_states), int(self.n_actions))
        if self.q_table.size == 0:
            self.q_table = np.zeros(shape, dtype=float)
        elif self.q_table.shape != shape:
            raise ValidationError(
                f"q_table shape {self.q_table.shape} != expected {shape}."
            )
        if self.algorithm == "double_q_learning" and self.q_table_b is None:
            self.q_table_b = np.zeros(shape, dtype=float)
        if self.q_table_b is not None and self.q_table_b.shape != shape:
            raise ValidationError(
                f"q_table_b shape {self.q_table_b.shape} != expected {shape}."
            )
        if self.state_visits.size == 0:
            self.state_visits = np.zeros(int(self.n_states), dtype=np.int64)

    @property
    def obs_dim(self) -> int:
        """Number of raw observation components this policy expects per step.

        Always ``1`` for ``Discrete`` environments. Read it to check an
        observation you are about to pass to :meth:`act` has the right width.
        """
        return int(self.discretizer.obs_dim)

    def state_index(self, observation: Any) -> int:
        """Look up the Q-table row for a raw observation.

        A convenience wrapper over the policy's own discretizer, so callers
        holding only the policy do not have to reach into it.

        Parameters
        ----------
        observation : Any
            An observation in the same format the environment emits.

        Returns
        -------
        int
            State index in ``[0, n_states)``.

        Raises
        ------
        ValidationError
            If the observation does not match the shape or range the
            discretizer was built for.
        """
        return self.discretizer.index(observation)

    def q_values_for_state(self, state: int) -> np.ndarray:
        """Return the action values this policy acts on for one state.

        Use this rather than indexing :attr:`q_table` directly: under Double
        Q-learning the acting values are the mean of both tables, and reading
        one table alone gives a biased half of the estimate.

        Parameters
        ----------
        state : int
            State index, typically from :meth:`state_index`.

        Returns
        -------
        np.ndarray
            One value per action, in the environment's action order. Higher
            means the agent expects more discounted future reward from taking
            that action in this state.
        """
        if self.q_table_b is None:
            return np.asarray(self.q_table[int(state)], dtype=float)
        return 0.5 * (
            np.asarray(self.q_table[int(state)], dtype=float)
            + np.asarray(self.q_table_b[int(state)], dtype=float)
        )

    def q_values(self, observation: Any) -> np.ndarray:
        """Return the action values for a raw observation.

        Discretizes and looks up in one call — the usual entry point when you
        want to see *why* the policy prefers an action rather than only which
        one it picks.

        Parameters
        ----------
        observation : Any
            An observation in the environment's own format.

        Returns
        -------
        np.ndarray
            One action value per action.

        Raises
        ------
        ValidationError
            If the observation does not match the discretizer.
        """
        return self.q_values_for_state(self.state_index(observation))

    def greedy_action_for_state(
        self,
        state: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Pick the highest-valued action for a state.

        This is the policy the agent is learning, as opposed to the
        exploratory one it behaves with while training. Evaluation and
        :meth:`buildml.Session.act_rl` use it by default.

        Parameters
        ----------
        state : int
            State index.
        rng : np.random.Generator or None
            Used only to break ties between equally-valued actions. Pass
            ``None`` to always take the lowest-numbered best action, which is
            reproducible but biased on an untrained table.

        Returns
        -------
        int
            Index of the chosen action.

        Notes
        -----
        Tie-breaking is not cosmetic. A freshly initialised table is all-zero,
        so every action ties; without ``rng`` the agent would take action 0
        forever and never generate the experience it needs to learn.
        """
        q = self.q_values_for_state(state)
        best = np.flatnonzero(q == q.max())
        if best.size == 1 or rng is None:
            return int(best[0])
        # Random tie-breaking matters: a freshly initialised table is all-zero,
        # so argmax would otherwise always lock onto action 0.
        return int(rng.choice(best))

    def epsilon_greedy_action_for_state(
        self,
        state: int,
        *,
        rng: np.random.Generator,
        epsilon: float,
    ) -> int:
        """Pick an action that explores with probability ``epsilon``.

        The behaviour policy used throughout :func:`train_tabular_control`.
        Exploration is what keeps the table honest: a state-action pair whose
        value is never sampled keeps whatever estimate it started with.

        Parameters
        ----------
        state : int
            State index.
        rng : np.random.Generator
            Draws both the explore/exploit coin and any tie-break. Required,
            so that a run seeded once is reproducible end to end.
        epsilon : float
            Probability of taking a uniformly random action instead of the
            greedy one. Higher explores more of the table and converges more
            slowly on the part that matters.

        Returns
        -------
        int
            Index of the chosen action.
        """
        if float(epsilon) > 0.0 and float(rng.random()) < float(epsilon):
            return int(rng.integers(0, self.n_actions))
        return self.greedy_action_for_state(state, rng=rng)

    def action_probabilities_for_state(
        self,
        state: int,
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return the action distribution this policy would use in a state.

        Useful for reporting how decisive the policy is somewhere, and for
        off-policy corrections that need the behaviour probabilities rather
        than a sampled action.

        Parameters
        ----------
        state : int
            State index.
        epsilon : float
            Exploration rate to describe. The default ``0.0`` gives the greedy
            distribution; pass the training epsilon to describe the behaviour
            policy instead.

        Returns
        -------
        np.ndarray
            Probabilities over actions, summing to 1.

        See Also
        --------
        epsilon_greedy_probabilities : The underlying calculation.
        """
        return epsilon_greedy_probabilities(
            self.q_values_for_state(state), epsilon=epsilon
        )

    def act(
        self,
        observation: Any,
        *,
        rng: np.random.Generator,
        deterministic: bool = True,
        epsilon: float = 0.0,
    ) -> int:
        """Choose an action for a raw observation.

        Discretizes the observation and applies either the greedy or the
        epsilon-greedy rule. This is the method to call when driving an
        environment yourself; :func:`act_tabular_observation` wraps it for the
        Session surface and also returns the values behind the choice.

        Parameters
        ----------
        observation : Any
            An observation in the environment's own format.
        rng : np.random.Generator
            Source of randomness for tie-breaks and exploration.
        deterministic : bool
            ``True`` acts greedily, which is what you want when measuring what
            the agent learned. ``False`` keeps exploring, which is what you
            want if the policy is still being improved online.
        epsilon : float
            Exploration rate, used only when ``deterministic`` is ``False``.

        Returns
        -------
        int
            Index of the chosen action.

        Raises
        ------
        ValidationError
            If the observation does not match the discretizer.
        """
        state = self.state_index(observation)
        if deterministic:
            return self.greedy_action_for_state(state, rng=rng)
        return self.epsilon_greedy_action_for_state(state, rng=rng, epsilon=epsilon)

    def greedy_policy_table(self) -> np.ndarray:
        """Return the action the policy would take in every state at once.

        The compact form of "what did the agent actually learn". On a gridworld
        this array reshapes to the grid and can be read as a map of arrows,
        which is usually more informative than any scalar metric.

        Returns
        -------
        np.ndarray
            Integer action index per state, of length ``n_states``.

        Notes
        -----
        States the agent never visited still appear here with an action, taken
        from their untouched initial values. Check
        :meth:`visited_state_fraction` before reading the whole table as
        something the agent has an opinion about.
        """
        if self.q_table_b is None:
            combined = self.q_table
        else:
            combined = 0.5 * (self.q_table + self.q_table_b)
        return np.argmax(combined, axis=1).astype(int)

    def state_value_table(self) -> np.ndarray:
        """Return the value of each state under the greedy policy.

        ``V(s) = max_a Q(s, a)``: how much discounted reward the agent expects
        from a state if it behaves greedily from there on. Plotted over a
        gridworld this shows the reward gradient the agent has discovered.

        Returns
        -------
        np.ndarray
            One value per state, of length ``n_states``.
        """
        if self.q_table_b is None:
            combined = self.q_table
        else:
            combined = 0.5 * (self.q_table + self.q_table_b)
        return np.max(combined, axis=1).astype(float)

    def visited_state_fraction(self) -> float:
        """Report how much of the state space training actually reached.

        Reported as ``state_coverage`` in the fit metrics and the honest test
        of whether a tabular run is trustworthy: the convergence argument for
        TD control assumes repeated visits to every state-action pair.

        Returns
        -------
        float
            Fraction of states visited at least once, in ``[0, 1]``. Low
            values on a ``Box`` environment usually mean ``n_bins`` is too
            high for the number of episodes run.
        """
        if self.state_visits.size == 0:
            return 0.0
        return float(np.count_nonzero(self.state_visits) / self.state_visits.size)


def _validate_hyperparameters(
    *,
    n_episodes: int,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
) -> None:
    if int(n_episodes) < 1:
        raise ValidationError("tabular_q requires n_episodes >= 1.")
    if int(max_steps) < 1:
        raise ValidationError("tabular_q requires max_steps >= 1.")
    if not 0.0 < float(learning_rate) <= 1.0:
        raise ValidationError(
            "tabular_q requires 0 < learning_rate <= 1 (TD step size alpha)."
        )
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValidationError("tabular_q requires 0 <= gamma <= 1.")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValidationError("tabular_q requires 0 <= epsilon <= 1.")
    if not 0.0 <= float(epsilon_min) <= 1.0:
        raise ValidationError("tabular_q requires 0 <= epsilon_min <= 1.")
    if float(epsilon_min) > float(epsilon):
        raise ValidationError("tabular_q requires epsilon_min <= epsilon.")
    if not 0.0 < float(epsilon_decay) <= 1.0:
        raise ValidationError("tabular_q requires 0 < epsilon_decay <= 1.")


def _td_update(
    policy: TabularValuePolicy,
    *,
    algorithm: str,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int,
    terminated: bool,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
) -> float:
    """Apply one TD-control update in place; return the signed TD error."""
    if algorithm == "double_q_learning":
        assert policy.q_table_b is not None  # guaranteed by __post_init__
        if bool(rng.random() < 0.5):
            updating, evaluating = policy.q_table, policy.q_table_b
        else:
            updating, evaluating = policy.q_table_b, policy.q_table
        if terminated:
            target = float(reward)
        else:
            # Select with the updating table, evaluate with the other one —
            # this decoupling is what removes maximisation bias.
            greedy = int(np.argmax(updating[next_state]))
            target = float(reward) + gamma * float(evaluating[next_state, greedy])
        td_error = target - float(updating[state, action])
        updating[state, action] = float(updating[state, action]) + learning_rate * td_error
        return float(td_error)

    q = policy.q_table
    if terminated:
        target = float(reward)
    elif algorithm == "q_learning":
        target = float(reward) + gamma * float(np.max(q[next_state]))
    elif algorithm == "sarsa":
        target = float(reward) + gamma * float(q[next_state, int(next_action)])
    elif algorithm == "expected_sarsa":
        probs = epsilon_greedy_probabilities(q[next_state], epsilon=epsilon)
        target = float(reward) + gamma * float(np.dot(probs, q[next_state]))
    else:  # pragma: no cover - guarded by resolve_tabular_algorithm
        raise ValidationError(f"Unknown tabular algorithm={algorithm!r}.")
    td_error = target - float(q[state, action])
    q[state, action] = float(q[state, action]) + learning_rate * td_error
    return float(td_error)


def train_tabular_control(
    *,
    env_id: str = "FrozenLake-v1",
    algorithm: str = "q_learning",
    n_episodes: int = 2_000,
    max_steps: int = 200,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    n_bins: int = 8,
    random_state: int | None = 0,
) -> tuple[TabularValuePolicy, dict[str, float], list[str], list[str]]:
    """Learn an action-value table by interacting with a Gymnasium environment.

    The engine behind ``Session.fit_rl(mode='tabular_q')``. Builds a
    discretizer from the environment, allocates ``Q[s, a]``, then runs
    ``n_episodes`` of epsilon-greedy interaction, applying one TD update per
    step. Nothing is fitted from a DataFrame here: unlike the bandit path, the
    data is generated by acting. The returned policy is what
    :func:`evaluate_tabular_policy` rolls out and
    :func:`act_tabular_observation` queries.

    Parameters
    ----------
    env_id : str
        Gymnasium id to train on. Must have a discrete action space. The
        default ``"FrozenLake-v1"`` is small enough to converge in seconds.
    algorithm : str
        One of :data:`TABULAR_ALGORITHMS`. ``q_learning`` and
        ``double_q_learning`` are off-policy and learn the optimal policy
        regardless of how much they explore; ``sarsa`` and ``expected_sarsa``
        are on-policy and learn a policy that accounts for its own
        exploration, which is the safer choice when mistakes are costly.
    n_episodes : int
        Episodes to run. Tabular methods need many visits per state-action
        pair, so this is the parameter to raise first when returns plateau
        below what the environment allows.
    max_steps : int
        Step cap per episode. Prevents a policy that has learned to stall from
        consuming the whole budget in one episode.
    learning_rate : float
        TD step size, ``alpha`` in the update rule: the fraction of the error
        folded into the estimate each step. Large values track change quickly
        but leave the estimates noisy; small values are steadier and slower.
    gamma : float
        Discount factor. Near ``1`` values distant reward almost as much as
        immediate reward, which is required for goal-reaching tasks; lower
        values make the agent short-sighted.
    epsilon : float
        Starting exploration rate. Starting near ``1.0`` is normal, since an
        untrained table has no useful preference to exploit.
    epsilon_min : float
        Floor for the decayed exploration rate. Keeping it above zero means
        the agent never fully stops checking its assumptions.
    epsilon_decay : float
        Per-episode multiplier applied to ``epsilon``. Values closer to ``1``
        explore for longer, which matters when reward is sparse and early
        greedy behaviour would lock in a bad path.
    n_bins : int
        Bins per dimension for continuous observations. Ignored for
        ``Discrete`` environments.
    random_state : int or None
        Seeds the discretizer probe, environment resets, and action sampling.
        Fix it for reproducible runs.

    Returns
    -------
    tuple
        ``(policy, metrics, disclosures, warnings)``. ``policy`` is a
        :class:`TabularValuePolicy`; ``metrics`` covers returns, table size,
        state coverage and mean absolute TD error; ``disclosures`` states what
        the method is and is not; ``warnings`` flags runs that did not improve,
        barely covered the table, or were too short to trust.

    Raises
    ------
    MissingExtraError
        If ``gymnasium`` is not installed. Install ``buildml[rl]``.
    ValidationError
        If a hyperparameter is out of range, the environment id cannot be
        made, the action space is continuous, or the observation space cannot
        be discretized within :data:`MAX_TABULAR_STATES`.

    Notes
    -----
    **Scale:** this is a teaching-scale method. The table has one row per
    discrete state, so cost grows with the size of the state space rather than
    the complexity of the task. For image or high-dimensional observations use
    ``mode='gym_sb3'`` with DQN, which replaces the table with a network but
    keeps this same update rule.

    Examples
    --------
    >>> policy, metrics, _disclosures, _warnings = train_tabular_control(
    ...     env_id="FrozenLake-v1", algorithm="q_learning", n_episodes=2000
    ... )
    >>> policy.greedy_policy_table().reshape(4, 4)  # doctest: +SKIP

    See Also
    --------
    evaluate_tabular_policy : Measure the learned policy under greedy actions.
    buildml.Session.fit_rl : The Session-level entry point.
    """
    algo = resolve_tabular_algorithm(algorithm)
    _validate_hyperparameters(
        n_episodes=n_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    gymnasium = require_gymnasium(feature="fit_rl(mode='tabular_q')")
    try:
        env = gymnasium.make(env_id)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Failed to create Gymnasium env_id={env_id!r}: {exc}"
        ) from exc

    warnings: list[str] = []
    disclosures = [
        "Tabular TD control learns an explicit Q[s, a] table by bootstrapping "
        "(no neural network, no function approximation).",
        f"algorithm={algo}; "
        + (
            "off-policy (bootstraps from max_a' Q)."
            if algo in {"q_learning", "double_q_learning"}
            else "on-policy (bootstraps from the behaviour policy)."
        ),
        "This path requires buildml[rl] (gymnasium). Core BC/bandit paths do not.",
        "Honesty: small discrete-control teaching loop — tabular methods do not "
        "scale to high-dimensional observations.",
        f"env_id={env_id!r}; n_episodes={n_episodes}; "
        f"learning_rate={learning_rate}; gamma={gamma}.",
    ]

    try:
        act_space = env.action_space
        if not hasattr(act_space, "n"):
            raise ValidationError(
                "tabular_q requires a discrete action space (action_space.n). "
                "Continuous control is out of scope for tabular methods."
            )
        n_actions = int(act_space.n)
        discretizer, disc_notes = build_discretizer(
            env,
            n_bins=n_bins,
            random_state=random_state,
        )
        disclosures.extend(disc_notes)
        policy = TabularValuePolicy(
            n_actions=n_actions,
            n_states=int(discretizer.n_states),
            algorithm=algo,
            discretizer=discretizer,
            gamma=float(gamma),
            learning_rate=float(learning_rate),
            env_id=env_id,
        )

        rng = np.random.default_rng(random_state)
        returns: list[float] = []
        episode_steps: list[int] = []
        abs_td_errors: list[float] = []
        current_epsilon = float(epsilon)
        total_steps = 0

        for episode in range(int(n_episodes)):
            current_epsilon = max(
                float(epsilon_min),
                float(epsilon) * float(epsilon_decay) ** episode,
            )
            seed = None if random_state is None else int(random_state) + episode
            reset_out = env.reset(seed=seed)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            state = discretizer.index(obs)
            action = policy.epsilon_greedy_action_for_state(
                state, rng=rng, epsilon=current_epsilon
            )
            episode_return = 0.0
            steps = 0
            for _ in range(int(max_steps)):
                step_out = env.step(int(action))
                if len(step_out) == 5:
                    next_obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    next_obs, reward, done, _info = step_out
                    terminated = bool(done)
                next_state = discretizer.index(next_obs)
                next_action = policy.epsilon_greedy_action_for_state(
                    next_state, rng=rng, epsilon=current_epsilon
                )
                td_error = _td_update(
                    policy,
                    algorithm=algo,
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    next_action=next_action,
                    terminated=bool(terminated),
                    learning_rate=float(learning_rate),
                    gamma=float(gamma),
                    epsilon=current_epsilon,
                    rng=rng,
                )
                policy.state_visits[state] += 1
                abs_td_errors.append(abs(td_error))
                episode_return += float(reward)
                steps += 1
                total_steps += 1
                state, action = next_state, next_action
                if done:
                    break
            returns.append(float(episode_return))
            episode_steps.append(int(steps))
    finally:
        env.close()

    window = min(20, len(returns))
    recent = returns[-window:] if window else []
    early = returns[:window] if window else []
    recent_td = abs_td_errors[-1_000:] if abs_td_errors else []
    metrics = {
        "n_episodes": float(len(returns)),
        "n_states": float(policy.n_states),
        "n_actions": float(policy.n_actions),
        "n_states_visited": float(int(np.count_nonzero(policy.state_visits))),
        "state_coverage": float(policy.visited_state_fraction()),
        "total_steps": float(total_steps),
        "mean_episode_steps": float(np.mean(episode_steps)) if episode_steps else float("nan"),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "last_return": float(returns[-1]) if returns else float("nan"),
        "mean_return_last_20": float(np.mean(recent)) if recent else float("nan"),
        "mean_return_first_20": float(np.mean(early)) if early else float("nan"),
        "final_epsilon": float(current_epsilon),
        "mean_abs_td_error": float(np.mean(recent_td)) if recent_td else float("nan"),
    }

    if metrics["state_coverage"] < 0.05 and discretizer.kind == "box":
        warnings.append(
            f"Only {metrics['n_states_visited']:.0f}/{policy.n_states} discretized "
            "states were ever visited; most of the table is untrained. Lower "
            "n_bins or train for more episodes."
        )
    if len(returns) >= 40 and metrics["mean_return_last_20"] <= metrics["mean_return_first_20"]:
        warnings.append(
            "Mean return did not improve between the first and last 20 episodes; "
            "try more episodes, a larger learning_rate, or slower epsilon_decay — "
            "this is an honest teaching loop, not a tuned agent."
        )
    if len(returns) < 200:
        warnings.append(
            f"n_episodes={len(returns)} is small for tabular TD control; "
            "convergence guarantees assume many visits per (state, action)."
        )

    return policy, metrics, disclosures, warnings


def evaluate_tabular_policy(
    policy: TabularValuePolicy,
    *,
    env_id: str | None = None,
    n_episodes: int = 20,
    max_steps: int = 200,
    random_state: int | None = 0,
    deterministic: bool = True,
    epsilon: float = 0.05,
) -> dict[str, float]:
    """Measure a fitted tabular policy by rolling it out in the environment.

    Backs ``Session.evaluate_rl`` for ``mode='tabular_q'``. Evaluation here is
    online: the score comes from fresh episodes rather than from held-out
    rows, so it answers "how well does this policy act" rather than "how well
    does it predict". Run it after :func:`train_tabular_control` and compare
    against the training returns, which are depressed by exploration.

    Parameters
    ----------
    policy : TabularValuePolicy
        A fitted policy, carrying both its table and its discretizer.
    env_id : str or None
        Environment to evaluate in. ``None`` reuses the id the policy trained
        on. Pass a different id only to test transfer; the state indices mean
        nothing in an environment with a different observation space.
    n_episodes : int
        Episodes to average over. Raise it for stochastic environments such as
        slippery FrozenLake, where single-episode returns say very little.
    max_steps : int
        Step cap per episode, so an unfinished policy cannot hang evaluation.
    random_state : int or None
        Seeds resets and tie-breaks. Offset internally from the training seeds
        so evaluation does not replay the exact episodes trained on.
    deterministic : bool
        ``True`` evaluates the greedy policy, which is the usual report.
        ``False`` evaluates the exploring behaviour policy instead.
    epsilon : float
        Exploration rate used only when ``deterministic`` is ``False``.

    Returns
    -------
    dict of str to float
        Episode-return statistics, mean episode length, and
        ``unseen_state_rate`` — the fraction of visited states the agent never
        saw in training.

    Raises
    ------
    MissingExtraError
        If ``gymnasium`` is not installed. Install ``buildml[rl]``.
    ValidationError
        If no environment id is available from either the argument or the
        policy.

    Notes
    -----
    Read ``unseen_state_rate`` before the returns. A high value means the
    policy is mostly acting on untrained entries, so the mean return describes
    initial values rather than anything that was learned.
    """
    gymnasium = require_gymnasium(feature="evaluate_rl(mode='tabular_q')")
    resolved_env = env_id or policy.env_id
    if not resolved_env:
        raise ValidationError("tabular_q evaluation requires an env_id.")
    env = gymnasium.make(resolved_env)
    rng = np.random.default_rng(random_state)
    returns: list[float] = []
    steps_per_episode: list[int] = []
    unseen_states = 0
    visited_states = 0
    try:
        for episode in range(int(n_episodes)):
            seed = (
                None
                if random_state is None
                else int(random_state) + 10_000 + episode
            )
            reset_out = env.reset(seed=seed)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            total = 0.0
            steps = 0
            for _ in range(int(max_steps)):
                state = policy.state_index(obs)
                visited_states += 1
                if int(policy.state_visits[state]) == 0:
                    unseen_states += 1
                if deterministic:
                    action = policy.greedy_action_for_state(state, rng=rng)
                else:
                    action = policy.epsilon_greedy_action_for_state(
                        state, rng=rng, epsilon=epsilon
                    )
                step_out = env.step(int(action))
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    obs, reward, done, _info = step_out
                total += float(reward)
                steps += 1
                if done:
                    break
            returns.append(total)
            steps_per_episode.append(steps)
    finally:
        env.close()
    return {
        "n_eval_episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "std_return": float(np.std(returns)) if returns else float("nan"),
        "min_return": float(np.min(returns)) if returns else float("nan"),
        "max_return": float(np.max(returns)) if returns else float("nan"),
        "mean_episode_steps": (
            float(np.mean(steps_per_episode)) if steps_per_episode else float("nan")
        ),
        "unseen_state_rate": (
            float(unseen_states / visited_states) if visited_states else float("nan")
        ),
    }


def act_tabular_observation(
    policy: TabularValuePolicy,
    observation: Any,
    *,
    random_state: int | None = 0,
    deterministic: bool = True,
    epsilon: float = 0.05,
) -> tuple[int, tuple[float, ...]]:
    """Score one observation, returning the action and the values behind it.

    Backs ``Session.act_rl`` for ``mode='tabular_q'``, which calls it once per
    supplied observation. Returning the values alongside the action is what
    lets the Session report *why* an action was chosen rather than only which
    one won.

    Parameters
    ----------
    policy : TabularValuePolicy
        A fitted policy.
    observation : Any
        A single observation in the environment's own format.
    random_state : int or None
        Seeds tie-breaking and exploration. Fixed by default so repeated calls
        on the same observation agree.
    deterministic : bool
        ``True`` returns the greedy action, which is the right choice for
        serving a trained policy. ``False`` explores.
    epsilon : float
        Exploration rate, used only when ``deterministic`` is ``False``.

    Returns
    -------
    tuple of (int, tuple of float)
        The chosen action index, and the action values for that state in
        action order. Near-equal values mean the policy is close to
        indifferent — often a sign the state was rarely visited.

    Raises
    ------
    ValidationError
        If the observation does not match the shape or range the policy's
        discretizer expects.

    See Also
    --------
    buildml.Session.act_rl : The Session-level entry point.
    """
    rng = np.random.default_rng(random_state)
    state = policy.state_index(observation)
    q_values = policy.q_values_for_state(state)
    if deterministic:
        action = policy.greedy_action_for_state(state, rng=rng)
    else:
        action = policy.epsilon_greedy_action_for_state(
            state, rng=rng, epsilon=epsilon
        )
    return int(action), tuple(float(v) for v in q_values.tolist())
