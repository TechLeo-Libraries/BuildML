"""RL / imitation catalog and honest capability matrix."""

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
NATIVE_RL_MODES = ("gym_reinforce",)
INDUSTRY_RL_MODES = ("gym_sb3",)

BANDIT_ALGORITHMS = ("linucb", "epsilon_greedy", "softmax")
SB3_ALGORITHMS = ("ppo", "dqn", "a2c")


def rl_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for imitation + RL backends and methods."""
    return {
        "imitation_backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": list(SKLEARN_IMITATION_ESTIMATORS),
                "modality": "tabular",
                "notes": (
                    "Behavioral cloning from demonstration tables on train only — "
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
                "algorithms": ("reinforce_linear_softmax",),
                "modality": "gymnasium",
                "notes": (
                    "REINFORCE-lite linear softmax on small discrete Gymnasium "
                    "envs (buildml[rl])."
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
            "holdout_rule": (
                "BC/bandit fit on train only; bandit holdout is offline DM/IPS."
            ),
            "offline_rl_disclosure": (
                "BuildML does not ship batch offline RL (CQL/IQL/Decision Transformer). "
                "Bandit IPS/DM and imitation-from-demos are disclosed offline paths."
            ),
        },
        "default_imitation_backend_when_installed": _default_imitation_backend(),
        "default_rl_backend_when_installed": _default_rl_backend(),
        "default_rl_mode_when_installed": _default_rl_mode(),
        "install_hints": {
            "rl": (
                "pip install 'buildml[rl]'  "
                "# Gymnasium REINFORCE-lite teaching loop"
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
    matrix = rl_capability_matrix()
    if backend is not None:
        entry = matrix["rl_backends"].get(backend)
        if entry is None or not entry.get("available"):
            return []
        if mode is not None and mode not in (entry.get("modes") or ()):
            return []
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
    entry = rl_capability_matrix()["imitation_backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def rl_backend_available(name: RlBackendName) -> bool:
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
    """Validate imitation backend/method pairing and apply honest defaults."""
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
    """Validate RL backend/mode/algorithm and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: RlBackendName
    algo_key = str(algorithm or "linucb").lower().replace("-", "_")
    if backend is None:
        if mode == "gym_reinforce":
            resolved_backend = "native"
        elif mode == "gym_sb3":
            resolved_backend = "industry"
        elif mode == "contextual_bandit" or algo_key in BANDIT_ALGORITHMS:
            resolved_backend = "sklearn"
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
            resolved_mode = "gym_reinforce"
        else:
            resolved_mode = "gym_sb3"
    else:
        resolved_mode = str(mode)

    if resolved_backend == "sklearn" and resolved_mode != "contextual_bandit":
        raise ValidationError(
            "backend='sklearn' supports mode='contextual_bandit' only."
        )
    if resolved_backend == "native" and resolved_mode != "gym_reinforce":
        raise ValidationError(
            "backend='native' supports mode='gym_reinforce' only."
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
        return resolved_backend, resolved_mode, "reinforce_linear_softmax"

    algo = algorithm or "ppo"
    algo_key = str(algo).lower().replace("-", "_")
    if algo_key not in SB3_ALGORITHMS:
        raise ValidationError(
            f"algorithm='{algo_key}' invalid for gym_sb3. "
            f"Choose from {list(SB3_ALGORITHMS)}."
        )
    return resolved_backend, resolved_mode, algo_key
