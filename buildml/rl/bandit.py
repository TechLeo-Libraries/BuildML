"""Choose among a fixed set of actions, learning from the rewards you observe.

A contextual bandit faces the simplest interesting version of the
decision-making problem. Each round it sees a context, picks one of several
arms, and observes a reward for *that arm only*. It never learns what the other
arms would have paid. Actions do not affect what happens next, so there is no
sequence to reason about: just the same one-step choice, repeatedly.

The whole difficulty is the exploration–exploitation trade-off. Exploit and you
keep pulling the arm that has looked best so far, which may only look best
because you have barely tried the others. Explore and you spend reward
discovering things. The three policies here resolve that tension differently.

**LinUCB** is the principled one. It maintains, per arm, both an estimate of
reward and a measure of how uncertain that estimate is for this particular
context, then picks the arm with the highest optimistic bound. Arms it knows
little about get a large bonus and are tried; as evidence accumulates the bonus
shrinks. Exploration is directed at genuine ignorance rather than spent at
random.

**Epsilon-greedy** takes the best-predicted arm most of the time and a uniformly
random arm otherwise. It explores arms it already understands as often as ones
it does not, which is wasteful: but it is trivial to reason about and to
explain to a stakeholder.

**Softmax** samples in proportion to predicted reward, so a clearly bad arm is
rarely tried and near-ties are explored roughly equally. It sits between the
other two.

Learning here is offline, from a fixed log. A policy fitted this way is
*estimated* to be better; only running it establishes that it is.

See Also
--------
buildml.rl.evaluate : Why offline estimates need care.
buildml.rl.fit.fit_rl : The user-facing entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from buildml.core.errors import ValidationError
from buildml.rl.features import softmax


@dataclass
class LinUCBPolicy:
    """Pick arms optimistically, favouring the ones you know least about.

    Keeps one independent linear reward model per arm: "disjoint" LinUCB, so
    called because arms share no parameters and what is learned about one says
    nothing about another. Each arm is scored as its predicted reward plus a
    bonus proportional to how uncertain that prediction is for the context in
    front of it, and the highest total wins.

    That bonus is what makes the algorithm work. An arm tried rarely, or tried
    only in very different contexts, carries large uncertainty and so gets
    chosen even when its point estimate is mediocre. As evidence accumulates the
    bonus decays and the policy settles onto what is genuinely best.

    Attributes
    ----------
    n_arms:
        How many actions are available.
    dim:
        The context width.
    alpha:
        How much uncertainty is worth. At 0.0 the policy is purely greedy and
        never explores; larger values buy more exploration at the cost of
        near-term reward. 1.0 is a reasonable default.
    A:
        Per-arm ``dim × dim`` matrices accumulating ``x xᵀ`` over the rows where
        that arm was pulled. Their inverses give the uncertainty estimate.
        Initialised to the identity, which acts as a mild prior and keeps them
        invertible before any data arrives.
    b:
        Per-arm ``dim`` vectors accumulating ``reward · x``. With ``A`` these
        give the ridge solution ``θ = A⁻¹b``.

    Notes
    -----
    **Reward is assumed linear in the context.** Where it is not, LinUCB will
    happily converge on the wrong arm. Try ``'epsilon_greedy'``, whose ridge
    models are no more expressive but whose exploration does not depend on the
    linear-uncertainty geometry being right.

    Cost grows with ``n_arms``: scoring inverts one ``dim × dim`` matrix per arm
    per row. Fine for a handful of arms; not intended for thousands.

    See Also
    --------
    RewardModelBandit : The epsilon-greedy and softmax alternatives.
    """

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
        """Score every arm for one context, optimism included.

        Each arm's score is its predicted reward plus a bonus that grows with
        how little the arm has been seen in contexts like this one.

        Parameters
        ----------
        x:
            A single context vector of length ``dim``.

        Returns
        -------
        numpy.ndarray
            One upper confidence bound per arm.

        Notes
        -----
        **These are not predicted rewards.** Each is a predicted reward plus an
        uncertainty bonus, so a high score can mean "probably good" or "no idea,
        worth finding out". The two are indistinguishable from the score alone,
        which is by design: the point is to act on both.
        """
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
        """Choose the arm with the highest upper confidence bound.

        Takes the best of :meth:`scores`, which means it will sometimes pick an
        arm it is merely uncertain about rather than one it believes is best.

        Parameters
        ----------
        x:
            A single context vector.
        rng:
            Accepted for interface compatibility with
            :meth:`RewardModelBandit.select` and ignored.

        Returns
        -------
        int
            The chosen arm index.

        Notes
        -----
        **LinUCB is deterministic.** Its exploration comes from the optimism in
        the bound, not from randomness, so the same context always yields the
        same arm: until an update changes the estimates.
        """
        del rng  # deterministic UCB
        return int(np.argmax(self.scores(x)))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        """Fold one observed ``(context, arm, reward)`` into the arm's model.

        Accumulates the outer product of the context into that arm's ``A`` and
        the reward-weighted context into its ``b``, which together sharpen both
        the reward estimate and the uncertainty around it.

        Parameters
        ----------
        x:
            The context the decision was made in.
        arm:
            Which arm was pulled.
        reward:
            What it paid.

        Notes
        -----
        Only the pulled arm is touched; the others learn nothing, which is
        exactly the bandit constraint. The update is incremental and needs no
        history, so a live policy can learn one interaction at a time.
        """
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
        """Learn from a whole log by replaying it one row at a time.

        The offline form of :meth:`update`: each logged decision is folded into
        the arm that was actually pulled.

        Parameters
        ----------
        x:
            Contexts, one row each.
        arms:
            The arm pulled in each row.
        rewards:
            The reward observed in each row.

        Notes
        -----
        **The order of the rows does not matter here.** LinUCB's updates are
        additive, so replaying a log gives the same state whatever sequence it
        arrives in. That also means this is not a simulation of how the policy
        would have behaved online: it is a fit to the log as a whole.
        """
        for i in range(x.shape[0]):
            self.update(x[i], int(arms[i]), float(rewards[i]))


@dataclass
class RewardModelBandit:
    """Predict each arm's reward, then explore by chance rather than by design.

    Splits the problem in two: a ridge regression per arm estimates what that
    arm pays in a given context, and a selection rule decides how much to trust
    those estimates. The separation makes the policy easy to inspect: you can
    look at predicted rewards directly, without an uncertainty bonus mixed in.

    ``'epsilon_greedy'`` takes the best-predicted arm with probability
    ``1 - epsilon`` and a uniformly random one otherwise. ``'softmax'`` samples
    in proportion to predicted reward, so poor arms are rarely tried and
    near-ties are explored roughly evenly.

    Attributes
    ----------
    n_arms:
        How many actions are available.
    dim:
        The context width.
    algorithm:
        ``'epsilon_greedy'`` or ``'softmax'``.
    epsilon:
        Random-action probability, for epsilon-greedy.
    temperature:
        Sampling sharpness, for softmax. Low concentrates on the best arm; high
        flattens toward uniform.
    random_state:
        Seed used when no generator is supplied to :meth:`select`.
    models:
        The per-arm ridge models, in arm order.

    Notes
    -----
    **An arm with no logged pulls gets a constant-zero model.** There is nothing
    to fit, and refusing would make the policy unusable on any log where one
    action was never tried. The consequence is that such an arm is predicted to
    pay zero: which may be optimistic or pessimistic depending on the reward
    scale, and either way is a guess. Exploration is the only thing that will
    ever correct it.

    **Unlike LinUCB, exploration here is undirected.** A random action is as
    likely to be one already well understood as one never tried, which is why
    LinUCB usually needs fewer interactions to find the best arm.

    See Also
    --------
    LinUCBPolicy : Uncertainty-directed exploration.
    """

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
        """Fit one reward model per arm from the rows where that arm was pulled.

        The log is partitioned by arm and a ridge regression fitted within each
        part, so every model learns only from decisions that actually chose its
        arm.

        Parameters
        ----------
        x:
            Contexts, one row each.
        arms:
            The arm pulled in each row.
        rewards:
            The reward observed in each row.

        Notes
        -----
        Each model sees only its own arm's rows, so an arm pulled rarely gets a
        correspondingly weak model: and one never pulled gets a constant-zero
        stand-in rather than an error. The fit succeeds either way; the
        resulting predictions for those arms are guesses, and only exploration
        will improve them.
        """
        self.models = []
        for a in range(self.n_arms):
            mask = arms == a
            model = Ridge(alpha=1.0, random_state=self.random_state)
            if int(mask.sum()) == 0:
                # No logged pulls: constant-zero predictor via empty fit fallback.
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
        """Predict what every arm would pay, for every context row.

        Runs all the per-arm models over the same contexts and stacks the
        answers, giving the full grid of what-if rewards rather than just the
        arm that was chosen.

        Parameters
        ----------
        x:
            A ``(n_rows, dim)`` context matrix, or a single 1-D context, which
            is treated as one row.

        Returns
        -------
        numpy.ndarray
            A ``(n_rows, n_arms)`` matrix of predicted rewards.

        Raises
        ------
        ValidationError
            If the models have not been fitted.

        Notes
        -----
        These are the counterfactual estimates the direct method relies on :
        what each arm *would* have paid, including the arms the log never tried
        in this context. That is also where they are least reliable.

        See Also
        --------
        buildml.rl.evaluate : How these feed the direct-method estimate.
        """
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
        """Predict what every arm would pay for one context.

        The single-row convenience form of :meth:`predicted_rewards`, flattened
        to a 1-D array so it can be indexed by arm.

        Parameters
        ----------
        x_row:
            A single context vector.

        Returns
        -------
        numpy.ndarray
            One predicted reward per arm.

        See Also
        --------
        predicted_rewards : The batch form.
        """
        return self.predicted_rewards(x_row).reshape(-1)

    def select(self, x: np.ndarray, *, rng: np.random.Generator | None = None) -> int:
        """Choose an arm, exploring according to the configured rule.

        Scores every arm, then applies epsilon-greedy or softmax selection :
        so the arm returned is not always the best-predicted one.

        Parameters
        ----------
        x:
            A single context vector.
        rng:
            The generator to draw from. Pass one when selecting for many rows,
            so successive calls advance the same stream; without it a fresh
            generator is built from ``random_state`` and every call makes the
            same draw.

        Returns
        -------
        int
            The chosen arm index.

        Raises
        ------
        ValidationError
            If ``algorithm`` is neither ``'epsilon_greedy'`` nor ``'softmax'``.

        Notes
        -----
        **This deliberately does not always pick the best arm.** For greedy
        selection, take ``argmax`` of :meth:`scores_row` instead: that is what
        :func:`~buildml.rl.act.act_rl` does when ``deterministic=True``.
        """
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
    """Model how the *logging* policy chose its actions.

    Inverse propensity scoring needs to know how likely each logged action was,
    given its context. That is rarely recorded, so it is estimated after the
    fact by fitting a classifier that predicts the logged action from the
    context.

    Parameters
    ----------
    x:
        Contexts from the training log.
    arms:
        The arm pulled in each row: the *label* this model learns to predict.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A fitted model whose ``predict_proba`` gives ``π_b(a | x)``.

    Raises
    ------
    ValidationError
        If the fit fails: most often because some arm appears in too few rows
        to support a class. Callers treat this as non-fatal: the bandit still
        fits and IPS is reported as ``NaN``.

    Notes
    -----
    **An estimated propensity is not a recorded one.** If the logging policy
    conditioned on something absent from ``x``: a human's judgement, a feature
    that was dropped: the estimate is systematically wrong and IPS inherits
    that error. This is the confounding assumption behind every
    propensity-weighted estimate, and no diagnostic here can check it.

    See Also
    --------
    offline_bandit_metrics : Where the propensities are used.
    """
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
    """Estimate what a new policy would have earned on a log it did not generate.

    Computes both offline estimators plus the diagnostics needed to judge them,
    and returns everything together so that no single number is read in
    isolation.

    Parameters
    ----------
    x:
        Holdout contexts.
    logged_arms:
        The arm the logging policy actually pulled in each row.
    logged_rewards:
        The reward that followed.
    policy_arms:
        The arm the new policy would pull in each row.
    predicted_rewards:
        A ``(n_rows, n_arms)`` matrix of counterfactual reward estimates, or
        ``None`` to skip the direct method.
    propensity:
        A ``(n_rows, n_arms)`` matrix of logging-policy action probabilities, or
        ``None`` to skip IPS.

    Returns
    -------
    dict
        ``n_rows``; ``direct_method``, the mean predicted reward for the new
        policy's choices; ``ips``, the propensity-weighted mean of rewards on
        agreeing rows; ``action_match_rate``, the fraction of rows where the two
        policies agree; ``mean_logged_reward_on_match``, the raw mean reward on
        those agreeing rows; and ``mean_logged_reward``, the baseline over all
        rows. Unavailable estimators are ``NaN`` rather than absent, so callers
        can rely on the keys.

    Notes
    -----
    **Read ``action_match_rate`` before either estimate.** At 0.95 the new policy
    barely differs from the log and both estimates are close to the observed
    reward: reliable, but also uninteresting. At 0.05 IPS rests on one row in
    twenty and the direct method extrapolates almost everywhere; neither
    supports a decision.

    **``mean_logged_reward`` is the baseline to beat.** An estimate is only
    meaningful relative to what the current policy already earns.

    **Propensities are clipped at 1e-6.** Without a floor, an action the model
    thinks was near-impossible produces an astronomical weight that swamps every
    other row. Clipping bounds that at a million, which is still enough for one
    row to dominate: a reason to distrust IPS when the match rate is low.

    See Also
    --------
    buildml.rl.evaluate : The trade-off between the two estimators.
    """
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
