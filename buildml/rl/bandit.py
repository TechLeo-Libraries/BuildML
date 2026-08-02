"""Contextual bandit policies (LinUCB / epsilon-greedy / softmax) — core sklearn/numpy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from buildml.core.errors import ValidationError
from buildml.rl.features import softmax


@dataclass
class LinUCBPolicy:
    """Disjoint LinUCB (one linear model per arm)."""

    n_arms: int
    dim: int
    alpha: float = 1.0
    A: list[np.ndarray] = field(default_factory=list)
    b: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.A:
            self.A = [np.eye(self.dim, dtype=float) for _ in range(self.n_arms)]
        if not self.b:
            self.b = [np.zeros(self.dim, dtype=float) for _ in range(self.n_arms)]

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        out = np.zeros(self.n_arms, dtype=float)
        for a in range(self.n_arms):
            a_inv = np.linalg.inv(self.A[a])
            theta = a_inv @ self.b[a]
            exploit = float(theta @ x)
            explore = float(self.alpha * np.sqrt(max(x @ a_inv @ x, 0.0)))
            out[a] = exploit + explore
        return out

    def select(self, x: np.ndarray, *, rng: np.random.Generator | None = None) -> int:
        del rng  # deterministic UCB
        return int(np.argmax(self.scores(x)))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        a = int(arm)
        self.A[a] = self.A[a] + np.outer(x, x)
        self.b[a] = self.b[a] + float(reward) * x

    def fit_logged(
        self,
        x: np.ndarray,
        arms: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        for i in range(x.shape[0]):
            self.update(x[i], int(arms[i]), float(rewards[i]))


@dataclass
class RewardModelBandit:
    """Per-arm Ridge reward models + epsilon-greedy or softmax selection."""

    n_arms: int
    dim: int
    algorithm: str = "epsilon_greedy"
    epsilon: float = 0.1
    temperature: float = 1.0
    random_state: int | None = 0
    models: list[Any] = field(default_factory=list)
    _fitted: bool = False

    def fit_logged(
        self,
        x: np.ndarray,
        arms: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        self.models = []
        for a in range(self.n_arms):
            mask = arms == a
            model = Ridge(alpha=1.0, random_state=self.random_state)
            if int(mask.sum()) == 0:
                # No logged pulls — constant-zero predictor via empty fit fallback.
                model.coef_ = np.zeros(self.dim, dtype=float)
                model.intercept_ = 0.0
                model.n_features_in_ = self.dim
            elif int(mask.sum()) == 1:
                # Ridge needs >=1 sample; fit a constant on the single reward.
                model.fit(x[mask], rewards[mask])
            else:
                model.fit(x[mask], rewards[mask])
            self.models.append(model)
        self._fitted = True

    def predicted_rewards(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValidationError("Bandit reward models are not fitted.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        preds = np.column_stack(
            [np.asarray(m.predict(x), dtype=float) for m in self.models]
        )
        return preds

    def scores_row(self, x_row: np.ndarray) -> np.ndarray:
        return self.predicted_rewards(x_row).reshape(-1)

    def select(self, x: np.ndarray, *, rng: np.random.Generator | None = None) -> int:
        scores = self.scores_row(x)
        gen = rng if rng is not None else np.random.default_rng(self.random_state)
        if self.algorithm == "epsilon_greedy":
            if float(gen.random()) < float(self.epsilon):
                return int(gen.integers(0, self.n_arms))
            return int(np.argmax(scores))
        if self.algorithm == "softmax":
            probs = softmax(scores, temperature=self.temperature)
            return int(gen.choice(self.n_arms, p=probs))
        raise ValidationError(
            f"Unknown reward-model bandit algorithm={self.algorithm!r}."
        )


def fit_propensity_model(
    x: np.ndarray,
    arms: np.ndarray,
    *,
    random_state: int | None = 0,
) -> LogisticRegression:
    """Fit a multinomial propensity model π(a|x) on logged train data."""
    model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=random_state,
    )
    try:
        model.fit(x, arms)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Failed to fit bandit propensity model for IPS: {exc}"
        ) from exc
    return model


def offline_bandit_metrics(
    *,
    x: np.ndarray,
    logged_arms: np.ndarray,
    logged_rewards: np.ndarray,
    policy_arms: np.ndarray,
    predicted_rewards: np.ndarray | None,
    propensity: np.ndarray | None,
) -> dict[str, float]:
    """Direct method + IPS + match-rate offline estimators (disclosed as offline)."""
    n = int(x.shape[0])
    if n == 0:
        return {
            "n_rows": 0.0,
            "direct_method": float("nan"),
            "ips": float("nan"),
            "action_match_rate": float("nan"),
            "mean_logged_reward_on_match": float("nan"),
        }
    match = policy_arms == logged_arms
    match_rate = float(np.mean(match))
    mean_logged_on_match = (
        float(np.mean(logged_rewards[match])) if match.any() else float("nan")
    )
    dm = float("nan")
    if predicted_rewards is not None:
        # Direct method: E[r̂(x, π(x))]
        row_idx = np.arange(n)
        dm = float(np.mean(predicted_rewards[row_idx, policy_arms]))
    ips = float("nan")
    if propensity is not None:
        # IPS: (1/n) Σ r_i * 1[π(x_i)=a_i] / π_b(a_i|x_i)
        pi = np.clip(propensity[np.arange(n), logged_arms], 1e-6, 1.0)
        ips = float(np.mean(logged_rewards * match.astype(float) / pi))
    return {
        "n_rows": float(n),
        "direct_method": dm,
        "ips": ips,
        "action_match_rate": match_rate,
        "mean_logged_reward_on_match": mean_logged_on_match,
        "mean_logged_reward": float(np.mean(logged_rewards)),
    }
