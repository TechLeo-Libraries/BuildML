"""Say what this installation can actually do, and resolve what was asked for.

Two jobs, both about the gap between the API surface and the current
environment. Several backends here are optional, so ``fit_rl`` accepting an
argument does not mean this machine can honour it.

**Reporting.** :func:`rl_capability_matrix` returns the whole picture as data:
which backends exist, which are installed right now, what each supports, what
the metrics mean, and what is explicitly out of scope. Because it is data rather
than prose it can drive a UI, a test, or a help message without any of them
drifting from the truth.

**Resolving.** The two ``resolve_*`` functions turn a partial request into a
concrete, valid triple, or refuse. Asking for ``algorithm='q_learning'`` implies
tabular control on the native backend, and that inference happens here: once,
so that every caller agrees. Invalid combinations fail at this boundary with a
message naming the alternatives, rather than deeper down where the cause is
harder to see.

The ``non_goals`` list in the matrix is deliberate. Knowing that batch offline RL
and robotics stacks are absent is more useful than discovering it after building
around the assumption that they are present.

See Also
--------
buildml.rl.extras : The dependency probes this builds on.
buildml.rl.fit : The main consumer of the resolvers.
"""

from __future__ import annotations

from typing import Any, Literal

from buildml.rl.extras import (
    gymnasium_available,
    imitation_available,
    rl_industry_available,
    stable_baselines3_available,
)

ImitationBackendName = Literal["sklearn", "industry"]
RlBackendName = Literal["sklearn", "native", "industry"]

SKLEARN_IMITATION_ESTIMATORS = (
    "logistic_regression",
    "hist_gradient_boosting",
    "ridge",
    "hist_gradient_boosting_regressor",
)
INDUSTRY_IMITATION_METHODS = (
    "bc_mlp",
    "gail_lite",
)

SKLEARN_RL_MODES = ("contextual_bandit",)
NATIVE_RL_MODES = ("gym_reinforce", "tabular_q")
INDUSTRY_RL_MODES = ("gym_sb3",)

BANDIT_ALGORITHMS = ("linucb", "epsilon_greedy", "softmax")
TABULAR_ALGORITHMS = (
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "double_q_learning",
)
POLICY_GRADIENT_ALGORITHMS = ("reinforce_linear_softmax",)
NATIVE_RL_ALGORITHMS = POLICY_GRADIENT_ALGORITHMS + TABULAR_ALGORITHMS
SB3_ALGORITHMS = ("ppo", "dqn", "a2c")

NATIVE_MODE_ALGORITHMS: dict[str, tuple[str, ...]] = {
    "gym_reinforce": POLICY_GRADIENT_ALGORITHMS,
    "tabular_q": TABULAR_ALGORITHMS,
}


def rl_capability_matrix() -> dict[str, Any]:
    """Report everything this installation can and cannot do, as data.

    Probes for optional dependencies each call, so the result reflects the
    environment now rather than at import time. Use it to decide what to offer a
    user, or to check that an intended path is actually available before
    building around it.

    Returns
    -------
    dict
        ``imitation_backends`` and ``rl_backends``, each mapping a backend name
        to its availability, the extra that provides it, the methods or modes
        and algorithms it supports, its modality, and a note on scope.
        ``evaluation`` lists the metrics each path produces plus the holdout
        rule and the offline-RL disclosure. ``default_*_when_installed`` say
        which path is chosen when nothing is specified. ``install_hints`` give
        the pip commands. ``non_goals`` lists what is deliberately absent, and
        the trailing ``*_present`` flags report individual packages.

    Notes
    -----
    **``available`` is computed per call, not cached.** Installing an extra mid
    session is reflected immediately.

    **Read ``non_goals`` before planning around this package.** Batch offline
    RL, robotics stacks, multi-agent simulation, and Ray RLlib are not here and
    are not coming. Knowing that now is cheaper than finding out later.

    Examples
    --------
    >>> matrix = rl_capability_matrix()
    >>> matrix["rl_backends"]["sklearn"]["available"]
    True
    >>> matrix["rl_backends"]["sklearn"]["algorithms"]
    ['linucb', 'epsilon_greedy', 'softmax']

    See Also
    --------
    list_rl_algorithms : Just the algorithm names.
    list_imitation_methods : Just the imitation method names.
    """
    return {
        "imitation_backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": list(SKLEARN_IMITATION_ESTIMATORS),
                "modality": "tabular",
                "notes": (
                    "Behavioral cloning from demonstration tables on train only: "
                    "always available sklearn fallback."
                ),
            },
            "industry": {
                "available": rl_industry_available(),
                "extra": "rl-industry",
                "methods": list(INDUSTRY_IMITATION_METHODS),
                "modality": "tabular (+ env for gail_lite)",
                "notes": (
                    "MLP BC via imitation+SB3 (bc_mlp) and small-budget GAIL-lite "
                    "when env-compatible demos are provided "
                    "(buildml[rl-industry])."
                ),
            },
        },
        "rl_backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "modes": list(SKLEARN_RL_MODES),
                "algorithms": list(BANDIT_ALGORITHMS),
                "modality": "tabular",
                "notes": (
                    "Contextual bandits (LinUCB / epsilon-greedy / softmax) on "
                    "logged train rows; offline DM/IPS evaluation disclosed."
                ),
            },
            "native": {
                "available": gymnasium_available(),
                "extra": "rl",
                "modes": list(NATIVE_RL_MODES),
                "algorithms": list(NATIVE_RL_ALGORITHMS),
                "algorithms_by_mode": {
                    mode: list(algos)
                    for mode, algos in NATIVE_MODE_ALGORITHMS.items()
                },
                "modality": "gymnasium",
                "notes": (
                    "REINFORCE-lite linear softmax (policy gradient) and tabular "
                    "TD control: Q-learning / SARSA / Expected SARSA / Double "
                    "Q-learning: on small discrete Gymnasium envs (buildml[rl]). "
                    "Continuous Box observations are uniformly discretized for "
                    "the tabular path."
                ),
            },
            "industry": {
                "available": rl_industry_available(),
                "extra": "rl-industry",
                "modes": list(INDUSTRY_RL_MODES),
                "algorithms": list(SB3_ALGORITHMS),
                "modality": "gymnasium",
                "notes": (
                    "Stable-Baselines3 PPO/DQN/A2C on honest small Gymnasium sims "
                    "(buildml[rl-industry]). Not robotics/AV/multi-agent."
                ),
            },
        },
        "evaluation": {
            "imitation_metrics_classification": ["accuracy", "macro_f1", "weighted_f1"],
            "imitation_metrics_regression": ["rmse", "mae", "r2"],
            "bandit_offline_metrics": [
                "direct_method",
                "ips",
                "action_match_rate",
                "mean_logged_reward_on_match",
            ],
            "gym_metrics": ["mean_return", "std_return", "n_eval_episodes"],
            "tabular_metrics": [
                "mean_return",
                "std_return",
                "n_eval_episodes",
                "mean_episode_steps",
                "unseen_state_rate",
            ],
            "holdout_rule": (
                "BC/bandit fit on train only; bandit holdout is offline DM/IPS."
            ),
            "offline_rl_disclosure": (
                "BuildML does not ship batch offline RL (CQL/IQL/Decision Transformer). "
                "Bandit IPS/DM and imitation-from-demos are disclosed offline paths. "
                "tabular_q is *online* off-policy TD control inside an env loop: "
                "off-policy is not the same thing as batch offline RL."
            ),
        },
        "default_imitation_backend_when_installed": _default_imitation_backend(),
        "default_rl_backend_when_installed": _default_rl_backend(),
        "default_rl_mode_when_installed": _default_rl_mode(),
        "install_hints": {
            "rl": (
                "pip install 'buildml[rl]'  "
                "# Gymnasium REINFORCE-lite + tabular Q-learning/SARSA loops"
            ),
            "rl-industry": (
                "pip install 'buildml[rl-industry]'  "
                "# SB3 PPO/DQN/A2C + imitation BC/GAIL-lite"
            ),
        },
        "non_goals": [
            "MuJoCo / robotics product stacks",
            "Autonomous-vehicle / multi-agent world sims",
            "Ray RLlib integration (prefer clean SB3 adapter)",
            "Batch offline RL without explicit offline-RL disclosure",
        ],
        "gymnasium_present": gymnasium_available(),
        "stable_baselines3_present": stable_baselines3_available(),
        "imitation_present": imitation_available(),
        "rl_industry_extra_present": rl_industry_available(),
    }


def _default_imitation_backend() -> str:
    if rl_industry_available():
        return "industry"
    return "sklearn"


def _default_rl_backend() -> str:
    if rl_industry_available():
        return "industry"
    if gymnasium_available():
        return "native"
    return "sklearn"


def _default_rl_mode() -> str:
    if rl_industry_available():
        return "gym_sb3"
    if gymnasium_available():
        return "gym_reinforce"
    return "contextual_bandit"


def list_imitation_methods(
    *,
    backend: ImitationBackendName | None = None,
) -> list[str]:
    """List the imitation methods you can actually use right now.

    Filters the capability matrix down to installed backends, so nothing in the
    result will fail with a missing-dependency error.

    Parameters
    ----------
    backend:
        Restrict to one backend, or ``None`` for everything installed.

    Returns
    -------
    list of str
        Method names, in backend order and de-duplicated. Empty when the named
        backend is unknown or not installed: the same answer for both, since
        from a caller's position they are equivalent.

    Examples
    --------
    >>> list_imitation_methods(backend="sklearn")
    ['logistic_regression', 'hist_gradient_boosting', 'ridge', 'hist_gradient_boosting_regressor']

    See Also
    --------
    rl_capability_matrix : The full picture, including what is not installed.
    """
    matrix = rl_capability_matrix()
    if backend is not None:
        entry = matrix["imitation_backends"].get(backend)
        if entry is None or not entry.get("available"):
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = []
    for entry in matrix["imitation_backends"].values():
        if not entry.get("available"):
            continue
        for name in entry.get("methods") or []:
            if name not in methods:
                methods.append(name)
    return methods


def list_rl_algorithms(
    *,
    backend: RlBackendName | None = None,
    mode: str | None = None,
) -> list[str]:
    """List the RL algorithms you can actually use right now.

    Filters the capability matrix down to installed backends, optionally
    narrowing further to one mode: useful because the native backend serves two
    modes with entirely different algorithm sets.

    Parameters
    ----------
    backend:
        Restrict to one backend, or ``None`` for everything installed.
    mode:
        Restrict further to one mode. Only meaningful together with
        ``backend``.

    Returns
    -------
    list of str
        Algorithm names, de-duplicated. Empty when the backend is unknown, not
        installed, or does not serve the requested mode.

    Examples
    --------
    >>> list_rl_algorithms(backend="sklearn")
    ['linucb', 'epsilon_greedy', 'softmax']
    >>> list_rl_algorithms(backend="sklearn", mode="gym_reinforce")
    []

    See Also
    --------
    rl_capability_matrix : The full picture, including what is not installed.
    """
    matrix = rl_capability_matrix()
    if backend is not None:
        entry = matrix["rl_backends"].get(backend)
        if entry is None or not entry.get("available"):
            return []
        if mode is not None:
            if mode not in (entry.get("modes") or ()):
                return []
            by_mode = entry.get("algorithms_by_mode") or {}
            if mode in by_mode:
                return list(by_mode[mode])
        return list(entry.get("algorithms") or [])
    algos: list[str] = []
    for entry in matrix["rl_backends"].values():
        if not entry.get("available"):
            continue
        for name in entry.get("algorithms") or []:
            if name not in algos:
                algos.append(name)
    return algos


def imitation_backend_available(name: ImitationBackendName) -> bool:
    """Say whether an imitation backend can be used right now.

    Checks the capability matrix rather than importing anything, so it is cheap
    enough to call before offering a choice to a user.

    Parameters
    ----------
    name:
        ``'sklearn'`` or ``'industry'``.

    Returns
    -------
    bool
        ``True`` when the backend exists and its dependencies are installed.
        An unknown name returns ``False`` rather than raising: unusable and
        non-existent amount to the same thing for a caller.

    See Also
    --------
    rl_backend_available : The RL counterpart.
    """
    entry = rl_capability_matrix()["imitation_backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def rl_backend_available(name: RlBackendName) -> bool:
    """Say whether an RL backend can be used right now.

    Checks the capability matrix rather than importing anything.

    Parameters
    ----------
    name:
        ``'sklearn'``, ``'native'``, or ``'industry'``.

    Returns
    -------
    bool
        ``True`` when the backend exists and its dependencies are installed.
        ``'sklearn'`` is always available; the other two need ``buildml[rl]`` or
        ``buildml[rl-industry]``.

    See Also
    --------
    imitation_backend_available : The imitation counterpart.
    """
    entry = rl_capability_matrix()["rl_backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_imitation_backend_method(
    *,
    backend: ImitationBackendName | None,
    estimator: str | None,
    method: str | None,
    task: str,
) -> tuple[ImitationBackendName, str]:
    """Turn a partial imitation request into a concrete backend and method.

    Callers may specify a backend, an estimator, a method, all of them, or none.
    This works out what was meant, fills in defaults that suit the task, and
    refuses combinations that cannot work: at the boundary, where the message
    can name the alternatives.

    Parameters
    ----------
    backend:
        The requested backend, or ``None`` to infer. Naming an industry method
        implies ``'industry'``; anything else implies ``'sklearn'``.
    estimator:
        The scikit-learn estimator. Defaults to ``'logistic_regression'`` for
        classification and ``'ridge'`` for regression.
    method:
        The industry method. Defaults to ``'bc_mlp'``.
    task:
        ``'classification'`` or ``'regression'``, already inferred from the
        action column.

    Returns
    -------
    ImitationBackendName
        The resolved backend.
    str
        The resolved estimator or method key, lowercased with hyphens
        normalised to underscores.

    Raises
    ------
    ValidationError
        If the estimator or method is not valid for the backend, if an industry
        method is requested for a regression task, or if the backend name is
        unknown.
    MissingExtraError
        If ``'industry'`` is resolved without ``buildml[rl-industry]``.

    Notes
    -----
    **Industry methods are classification-only.** Both ``'bc_mlp'`` and
    ``'gail_lite'`` assume a discrete action space, so a continuous action
    column is rejected here rather than producing a policy that quietly treats
    distinct action values as unrelated categories.

    See Also
    --------
    resolve_rl_backend_mode_algorithm : The RL counterpart.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    if backend is None:
        meth_key = None if method is None else str(method).lower().replace("-", "_")
        if meth_key in INDUSTRY_IMITATION_METHODS:
            backend = "industry"  # type: ignore[assignment]
        else:
            backend = "sklearn"  # type: ignore[assignment]

    if backend == "sklearn":
        est = estimator or (
            "logistic_regression" if task == "classification" else "ridge"
        )
        est_key = str(est).lower().replace("-", "_")
        if est_key not in SKLEARN_IMITATION_ESTIMATORS:
            raise ValidationError(
                f"estimator='{est_key}' is not valid for backend='sklearn'. "
                f"Choose from {list(SKLEARN_IMITATION_ESTIMATORS)}."
            )
        return "sklearn", est_key

    if backend == "industry":
        meth = method or "bc_mlp"
        meth_key = str(meth).lower().replace("-", "_")
        if meth_key not in INDUSTRY_IMITATION_METHODS:
            raise ValidationError(
                f"method='{meth_key}' is not valid for backend='industry'. "
                f"Choose from {list(INDUSTRY_IMITATION_METHODS)}."
            )
        if task != "classification":
            raise ValidationError(
                "Industry imitation methods (bc_mlp, gail_lite) require "
                "task='classification' (discrete actions)."
            )
        if not imitation_backend_available("industry"):
            raise MissingExtraError("rl-industry", "backend='industry' imitation")
        return "industry", meth_key

    raise ValidationError(f"Unknown imitation backend={backend!r}.")


def resolve_rl_backend_mode_algorithm(
    *,
    backend: RlBackendName | None,
    mode: str | None,
    algorithm: str | None,
) -> tuple[RlBackendName, str, str]:
    """Turn a partial RL request into a concrete backend, mode, and algorithm.

    The three settings constrain one another, so specifying any of them usually
    determines the rest. Asking for ``algorithm='q_learning'`` can only mean
    tabular control on the native backend, and this is where that inference
    happens: once, so every caller resolves it the same way.

    Parameters
    ----------
    backend:
        The requested backend, or ``None`` to infer from the mode, then the
        algorithm, then what is installed.
    mode:
        The requested mode, or ``None`` to infer from the backend. Under
        ``'native'``, a tabular algorithm selects ``'tabular_q'`` and anything
        else selects ``'gym_reinforce'``.
    algorithm:
        The requested algorithm, or ``None`` for the mode's default.

    Returns
    -------
    RlBackendName
        The resolved backend.
    str
        The resolved mode.
    str
        The resolved algorithm key, lowercased with hyphens normalised to
        underscores.

    Raises
    ------
    ValidationError
        If the backend and mode are incompatible, or the algorithm does not
        belong to the mode. Requesting a tabular algorithm under
        ``'gym_reinforce'`` gets a message pointing at ``'tabular_q'`` rather
        than a bare rejection.
    MissingExtraError
        If the resolved backend's dependencies are not installed.

    Notes
    -----
    **``'linucb'`` under ``'tabular_q'`` is treated as unset**, not as an error.
    It is ``fit_rl``'s shared default, so a caller who passed only
    ``mode='tabular_q'`` never chose it: rejecting them for a default they did
    not set would be unhelpful. They get ``'q_learning'``.

    See Also
    --------
    resolve_imitation_backend_method : The imitation counterpart.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: RlBackendName
    algo_key = str(algorithm or "linucb").lower().replace("-", "_")
    if backend is None:
        if mode in NATIVE_RL_MODES:
            resolved_backend = "native"
        elif mode == "gym_sb3":
            resolved_backend = "industry"
        elif mode == "contextual_bandit" or algo_key in BANDIT_ALGORITHMS:
            resolved_backend = "sklearn"
        elif algo_key in NATIVE_RL_ALGORITHMS:
            resolved_backend = "native"
        elif algo_key in SB3_ALGORITHMS:
            resolved_backend = "industry"
        else:
            resolved_backend = _default_rl_backend()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    resolved_mode: str
    if mode is None:
        if resolved_backend == "sklearn":
            resolved_mode = "contextual_bandit"
        elif resolved_backend == "native":
            # An explicit tabular algorithm selects the tabular mode; otherwise
            # the historical REINFORCE default stands.
            resolved_mode = (
                "tabular_q" if algo_key in TABULAR_ALGORITHMS else "gym_reinforce"
            )
        else:
            resolved_mode = "gym_sb3"
    else:
        resolved_mode = str(mode)

    if resolved_backend == "sklearn" and resolved_mode != "contextual_bandit":
        raise ValidationError(
            "backend='sklearn' supports mode='contextual_bandit' only."
        )
    if resolved_backend == "native" and resolved_mode not in NATIVE_RL_MODES:
        raise ValidationError(
            f"backend='native' supports mode in {list(NATIVE_RL_MODES)} only."
        )
    if resolved_backend == "industry" and resolved_mode != "gym_sb3":
        raise ValidationError(
            "backend='industry' supports mode='gym_sb3' only."
        )

    if not rl_backend_available(resolved_backend):
        extra = rl_capability_matrix()["rl_backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "rl-industry"), f"backend='{resolved_backend}'")

    if resolved_mode == "contextual_bandit":
        algo = algorithm or "linucb"
        algo_key = str(algo).lower().replace("-", "_")
        if algo_key not in BANDIT_ALGORITHMS:
            raise ValidationError(
                f"algorithm='{algo_key}' invalid for contextual_bandit. "
                f"Choose from {list(BANDIT_ALGORITHMS)}."
            )
        return resolved_backend, resolved_mode, algo_key

    if resolved_mode == "gym_reinforce":
        if algo_key in TABULAR_ALGORITHMS:
            raise ValidationError(
                f"algorithm='{algo_key}' is tabular TD control; "
                "use mode='tabular_q' (or drop mode=) instead of 'gym_reinforce'."
            )
        return resolved_backend, resolved_mode, "reinforce_linear_softmax"

    if resolved_mode == "tabular_q":
        # 'linucb' is the shared fit_rl default, so treat it as "unset" here
        # rather than rejecting a caller who only passed mode='tabular_q'.
        if algorithm is None or algo_key == "linucb":
            return resolved_backend, resolved_mode, "q_learning"
        if algo_key not in TABULAR_ALGORITHMS:
            raise ValidationError(
                f"algorithm='{algo_key}' invalid for tabular_q. "
                f"Choose from {list(TABULAR_ALGORITHMS)}."
            )
        return resolved_backend, resolved_mode, algo_key

    algo = algorithm or "ppo"
    algo_key = str(algo).lower().replace("-", "_")
    if algo_key not in SB3_ALGORITHMS:
        raise ValidationError(
            f"algorithm='{algo_key}' invalid for gym_sb3. "
            f"Choose from {list(SB3_ALGORITHMS)}."
        )
    return resolved_backend, resolved_mode, algo_key
