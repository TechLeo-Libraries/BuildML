# ruff: noqa: E501
"""Imitation learning + reinforcement learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

RL_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="imitation-behavioral-cloning",
            title="Behavioral cloning from demonstration tables",
            summary="BC fits a supervised state→action policy on Session train demonstrations only.",
            definition=(
                "Behavioral cloning treats expert demonstrations as supervised "
                "examples: features are states, the target is the demonstrated "
                "action. fit_imitation trains on the train partition only."
            ),
            intuition=(
                "Watch an expert's (state, action) pairs and imitate the mapping "
                "with a classifier or regressor."
            ),
            formal_idea="π_θ ≈ argmin_θ Σ_{(s,a)∈D_train} ℓ(π_θ(s), a).",
            why_it_matters=(
                "Train-only cloning preserves holdout honesty for imitation metrics.",
                "Simple, Session-shaped entry point before online RL.",
            ),
            how_buildml_uses=(
                "Session.fit_imitation → predict_imitation_action / evaluate_imitation.",
            ),
            interpretation_rules=(
                "accuracy/macro_f1 (discrete) or rmse/mae/r2 (continuous) vs demos.",
                "train_score is in-sample — prefer holdout evaluate_imitation.",
            ),
            assumptions=("Non-null state features and actions on train; split present.",),
            failure_modes=(
                "Covariate shift vs expert; compounding errors (DAgger not default).",
            ),
            anti_patterns=(
                "Fitting BC on the full frame before split.",
                "Calling BC a robotics / MuJoCo platform.",
            ),
            worked_example_pattern=(
                "fit_imitation() → evaluate_imitation(partition='validation').",
            ),
            related_concepts=(
                "imitation-bundle-boundary",
                "rl-contextual-bandit",
                "leakage-boundary",
            ),
        ),
        _note(
            key="imitation-bundle-boundary",
            title="Imitation bundle vs Session checkpoint",
            summary="buildml.imitation_bundle.v1 stores ImitationPlan; checkpoints do not embed the policy.",
            definition=(
                "An imitation bundle directory holds meta.json + imitation_plan.joblib "
                "under schema buildml.imitation_bundle.v1."
            ),
            intuition="Save the cloned policy separately from workflow resume state.",
            formal_idea="ImitationPlan is not embedded in a Session checkpoint payload.",
            why_it_matters=("Avoid silent gaps when reloading workflows.",),
            how_buildml_uses=("save_imitation_bundle / load_imitation_bundle.",),
            interpretation_rules=(
                "Reload policy via load_imitation_bundle after checkpoint_load.",
            ),
            assumptions=("Bundle format matches buildml.imitation_bundle.v1.",),
            failure_modes=("Mixing imitation bundles with RL / CBR / RAG bundles.",),
            anti_patterns=("Expecting checkpoint_load to restore the BC policy.",),
            worked_example_pattern=(
                "save_imitation_bundle(path) → load_imitation_bundle(path).",
            ),
            related_concepts=("imitation-behavioral-cloning", "rl-bundle-boundary"),
        ),
        _note(
            key="rl-contextual-bandit",
            title="Contextual bandits on logged tables",
            summary="LinUCB / epsilon-greedy / softmax policies learn from train (context, action, reward) only.",
            definition=(
                "A contextual bandit observes context x, chooses arm a, and receives "
                "reward r. BuildML fits offline from logged train rows and evaluates "
                "with disclosed offline estimators (direct method, IPS)."
            ),
            intuition=(
                "Given features about a user/context, pick an action that maximizes "
                "expected reward — learned from historical logs."
            ),
            formal_idea=(
                "LinUCB: a_hat = argmax_a [theta_a·x + alpha*sqrt(x⊤A_a^{-1}x)]. "
                "Offline: DM = E[r_hat(x,π(x))]; IPS = E[r 1[π=a]/π_b(a|x)]."
            ),
            why_it_matters=(
                "Train-only updates preserve holdout honesty.",
                "Offline metrics must not be confused with online A/B lifts.",
            ),
            how_buildml_uses=(
                "Session.fit_rl(mode='contextual_bandit') → act_rl / evaluate_rl.",
            ),
            interpretation_rules=(
                "Read evaluate_rl.offline=True and DM/IPS disclosures.",
                "IPS needs a propensity model; treat confounding cautiously.",
            ),
            assumptions=("Logged actions + numeric rewards on train; discrete arms.",),
            failure_modes=(
                "Confounded logs; support mismatch; empty arms in train.",
            ),
            anti_patterns=(
                "Reporting IPS as online A/B.",
                "Updating the bandit from validation/test.",
            ),
            worked_example_pattern=(
                "fit_rl(algorithm='linucb', reward_column='reward') → "
                "evaluate_rl(partition='validation').",
            ),
            related_concepts=(
                "rl-offline-metrics",
                "rl-gym-reinforce",
                "rl-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="rl-offline-metrics",
            title="Offline bandit evaluation (DM / IPS)",
            summary="Holdout bandit scores are offline estimators with explicit disclosures — not live A/B.",
            definition=(
                "Direct method uses predicted rewards under the learned policy. "
                "IPS reweights logged rewards by inverse propensity when the "
                "policy action matches the logged action."
            ),
            intuition=(
                "Estimate how a new policy would have done on historical logs "
                "without deploying it."
            ),
            formal_idea=(
                "DM = (1/n)Σ r_hat(x_i, π(x_i)); "
                "IPS = (1/n)Σ r_i 1[π(x_i)=a_i] / π_b(a_i|x_i)."
            ),
            why_it_matters=("Prevents overclaiming online gains from log replay.",),
            how_buildml_uses=("evaluate_rl for contextual_bandit sets offline=True.",),
            interpretation_rules=(
                "Prefer reporting both DM and IPS with caveats.",
                "action_match_rate shows overlap with the logging policy.",
            ),
            assumptions=("Propensity model fitted on train; positivity roughly holds.",),
            failure_modes=(
                "Extreme propensities; distribution shift; unobserved confounders.",
            ),
            anti_patterns=("Calling offline IPS 'production lift'.",),
            worked_example_pattern=(
                "Inspect evaluate_rl().metrics['ips'] and disclosures.",
            ),
            related_concepts=("rl-contextual-bandit",),
        ),
        _note(
            key="rl-gym-reinforce",
            title="Gymnasium REINFORCE-lite (optional buildml[rl])",
            summary="Linear softmax REINFORCE on small discrete Gymnasium envs; core never requires gymnasium.",
            definition=(
                "fit_rl(mode='gym_reinforce') trains a linear softmax policy with "
                "REINFORCE returns-to-go inside a Gymnasium env loop. Requires "
                "optional extra buildml[rl]."
            ),
            intuition=(
                "Play episodes in a small env, reinforce actions that led to "
                "higher returns."
            ),
            formal_idea="π(a|s)=softmax(Ws); ∇J ≈ Σ_t G_t ∇log π(a_t|s_t).",
            why_it_matters=(
                "Optional depth without weighing the core install.",
                "Honest small-env teaching path — not MuJoCo/robotics.",
            ),
            how_buildml_uses=(
                "Session.fit_rl(mode='gym_reinforce', env_id='CartPole-v1') → "
                "evaluate_rl / act_rl(observations=...).",
            ),
            interpretation_rules=(
                "mean_return from env rollouts; offline=False for this mode.",
            ),
            assumptions=(
                "Discrete action space; Box-like observations; gymnasium installed.",
            ),
            failure_modes=(
                "Under-trained CartPole; continuous-action envs unsupported.",
            ),
            anti_patterns=(
                "Claiming BuildML is a MuJoCo / robotics platform.",
                "Importing gymnasium into core paths.",
            ),
            worked_example_pattern=(
                "pip install 'buildml[rl]'; fit_rl(mode='gym_reinforce', n_episodes=300).",
            ),
            related_concepts=(
                "rl-contextual-bandit",
                "rl-tabular-q-learning",
                "rl-bundle-boundary",
            ),
        ),
        _note(
            key="rl-tabular-q-learning",
            title="Tabular Q-learning (off-policy TD control)",
            summary="Learn Q[s,a] by bootstrapping from max_a' Q[s',a'] — the foundation DQN scales up with a neural network.",
            definition=(
                "Q-learning is off-policy temporal-difference control. It stores "
                "one action-value per (state, action) pair and moves each entry "
                "toward the Bellman optimality target r + γ max_a' Q(s', a'), "
                "regardless of which action the exploring behaviour policy took. "
                "fit_rl(mode='tabular_q', algorithm='q_learning') runs this loop "
                "on a discrete-action Gymnasium env behind buildml[rl]."
            ),
            intuition=(
                "Keep a lookup table of 'how good is this action in this "
                "situation'. After every step, nudge the entry you just used "
                "toward the reward you got plus the best value you believe is "
                "available next."
            ),
            formal_idea=(
                "Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]; "
                "terminal transitions use the target r with no bootstrap. "
                "Double Q-learning keeps Q_A / Q_B and evaluates "
                "argmax_a Q_A(s',a) with Q_B to cancel maximisation bias."
            ),
            why_it_matters=(
                "It is the reference point for value-based RL: DQN is Q-learning "
                "with a neural network, a replay buffer, and a target network.",
                "The Q-table is fully inspectable — greedy_policy_table() and "
                "state_value_table() show exactly what was learned.",
                "Converges to the optimal policy under standard conditions "
                "(every state-action visited infinitely often, decaying α).",
            ),
            how_buildml_uses=(
                "Session.fit_rl(mode='tabular_q', algorithm='q_learning') → "
                "evaluate_rl / act_rl(observations=...).",
                "Continuous Box observations are uniformly discretized first; "
                "Discrete spaces (FrozenLake / Taxi / CliffWalking) index directly.",
            ),
            interpretation_rules=(
                "mean_return from env rollouts; offline=False for this mode.",
                "state_coverage shows how much of the table was ever updated.",
                "unseen_state_rate at eval time shows how often the greedy policy "
                "acted from an untrained Q-row.",
            ),
            assumptions=(
                "Discrete action space; discrete or discretizable observations.",
                "gymnasium installed (buildml[rl]).",
                "Enough episodes for repeated (state, action) visits.",
            ),
            failure_modes=(
                "State-space blow-up: n_bins ** obs_dim grows exponentially.",
                "Maximisation bias with noisy rewards (use double_q_learning).",
                "Aliasing: two genuinely different states share a discretized bin.",
            ),
            anti_patterns=(
                "Using tabular Q-learning on high-dimensional observations "
                "instead of gym_reinforce / gym_sb3 function approximation.",
                "Calling off-policy TD control 'offline RL' — tabular_q is an "
                "online env loop, not batch RL from a fixed dataset.",
            ),
            worked_example_pattern=(
                "pip install 'buildml[rl]'; fit_rl(mode='tabular_q', "
                "algorithm='q_learning', env_id='FrozenLake-v1', n_episodes=3000).",
            ),
            related_concepts=(
                "rl-sarsa-on-policy",
                "rl-state-discretization",
                "rl-gym-reinforce",
                "rl-sb3-industry",
                "rl-bundle-boundary",
            ),
        ),
        _note(
            key="rl-sarsa-on-policy",
            title="SARSA and Expected SARSA (on-policy TD control)",
            summary="On-policy TD control bootstraps from the action the behaviour policy will actually take, so it learns a safer exploring policy.",
            definition=(
                "SARSA updates Q(s,a) toward r + γ Q(s', a') where a' is the "
                "action the epsilon-greedy behaviour policy actually selects. "
                "Expected SARSA replaces that sample with its expectation under "
                "the behaviour policy, removing the variance of the a' draw."
            ),
            intuition=(
                "Q-learning learns the value of behaving optimally afterwards. "
                "SARSA learns the value of continuing to behave the way you "
                "actually behave — including your exploration mistakes."
            ),
            formal_idea=(
                "SARSA: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') − Q(s,a)]. "
                "Expected SARSA: Q(s,a) ← Q(s,a) + α[r + γ Σ_a' π(a'|s') Q(s',a') − Q(s,a)]."
            ),
            why_it_matters=(
                "On-policy control avoids the classic CliffWalking failure where "
                "the Q-learning optimal path runs along a cliff the exploring "
                "agent keeps falling off.",
                "The on-policy vs off-policy distinction is the single most "
                "load-bearing idea when reading RL papers.",
            ),
            how_buildml_uses=(
                "Session.fit_rl(mode='tabular_q', algorithm='sarsa') or "
                "algorithm='expected_sarsa'.",
            ),
            interpretation_rules=(
                "Compare mean_return against algorithm='q_learning' on the same "
                "env and seed; differences are about the exploration policy.",
                "Expected SARSA usually shows lower mean_abs_td_error than SARSA.",
            ),
            assumptions=(
                "Same discrete-action / discretizable-state assumptions as Q-learning.",
                "The epsilon schedule is part of the learned objective for on-policy control.",
            ),
            failure_modes=(
                "Fixed high epsilon keeps the learned values pessimistic forever.",
                "Comparing on-policy and off-policy returns without matching seeds.",
            ),
            anti_patterns=(
                "Reporting SARSA values as optimal Q* values.",
                "Assuming SARSA and Q-learning must converge to the same policy.",
            ),
            worked_example_pattern=(
                "fit_rl(mode='tabular_q', algorithm='sarsa', "
                "env_id='CliffWalking-v0', n_episodes=2000).",
            ),
            related_concepts=(
                "rl-tabular-q-learning",
                "rl-state-discretization",
                "rl-gym-reinforce",
            ),
        ),
        _note(
            key="rl-state-discretization",
            title="Discretizing continuous observations for tabular RL",
            summary="Tabular methods need finite states; BuildML bins Box observations uniformly and discloses where each bound came from.",
            definition=(
                "A discretizer maps a continuous observation vector to one "
                "integer state index by binning each dimension and combining the "
                "per-dimension buckets as a mixed-radix number. BuildML uses the "
                "declared space bounds where they are finite and a seeded "
                "random-policy probe (1st/99th percentile) where they are not."
            ),
            intuition=(
                "Turn a continuous dial into a small number of labelled notches "
                "so a lookup table can have one row per notch combination."
            ),
            formal_idea=(
                "state = Σ_d bucket_d · n_bins^(D−1−d), so the table has "
                "n_bins^D rows; BuildML refuses allocations above a hard state cap."
            ),
            why_it_matters=(
                "It makes the curse of dimensionality concrete: doubling bins or "
                "adding a dimension multiplies the table size.",
                "It is the honest reason function approximation (REINFORCE, DQN) "
                "exists at all.",
            ),
            how_buildml_uses=(
                "fit_rl(mode='tabular_q', n_bins=...) builds the discretizer and "
                "records it in RlPlan.config['discretizer'].",
            ),
            interpretation_rules=(
                "bound_sources tells you which dims used declared space bounds, "
                "which were probed, and which fell back to [-1, 1].",
                "Low state_coverage means most bins are unreachable — reduce n_bins.",
            ),
            assumptions=(
                "Box or Discrete observation space; MultiDiscrete is refused.",
                "Observations stay roughly inside the modelled range at eval time.",
            ),
            failure_modes=(
                "Too few bins → state aliasing; too many → no repeated visits.",
                "Distribution shift pushing observations outside the probed range "
                "(values are clipped into the edge bins).",
            ),
            anti_patterns=(
                "Cranking n_bins up until the state cap error appears instead of "
                "switching to function approximation.",
                "Comparing tabular returns across different n_bins as if the "
                "state space were the same.",
            ),
            worked_example_pattern=(
                "fit_rl(mode='tabular_q', env_id='CartPole-v1', n_bins=6, "
                "n_episodes=4000); inspect rl_plan.config['discretizer'].",
            ),
            related_concepts=(
                "rl-tabular-q-learning",
                "rl-sarsa-on-policy",
                "rl-gym-reinforce",
            ),
        ),
        _note(
            key="rl-sb3-industry",
            title="Stable-Baselines3 industry path (buildml[rl-industry])",
            summary="PPO/DQN/A2C on small Gymnasium envs via SB3; defaults when rl-industry is installed.",
            definition=(
                "fit_rl(backend='industry', mode='gym_sb3') trains Stable-Baselines3 "
                "policies on honest small discrete-action envs. Requires "
                "buildml[rl-industry] (SB3 + imitation + gymnasium + torch)."
            ),
            intuition=(
                "Use industry-grade policy-gradient / value-based RL on CartPole-class "
                "teaching envs without claiming a robotics product."
            ),
            formal_idea=(
                "π_θ learned by PPO/DQN/A2C env interaction; evaluate via mean_return. "
                "DQN is tabular Q-learning scaled up: the Q-table becomes a network, "
                "plus a replay buffer and a target network for stability."
            ),
            why_it_matters=(
                "Industry depth without Ray RLlib complexity or MuJoCo scope creep.",
                "Offline RL (batch RL) remains explicitly out of scope.",
            ),
            how_buildml_uses=(
                "Session.fit_rl(backend='industry', mode='gym_sb3', algorithm='ppo') → "
                "evaluate_rl / act_rl(observations=...).",
            ),
            interpretation_rules=(
                "mean_return from env rollouts; offline=False.",
                "Read rl_capability_matrix() for backend defaults and non-goals.",
            ),
            assumptions=(
                "Discrete action space; Box-like observations; rl-industry installed.",
            ),
            failure_modes=(
                "Under-trained CartPole; continuous-action envs unsupported.",
            ),
            anti_patterns=(
                "Claiming BuildML is a MuJoCo / AV / multi-agent platform.",
                "Confusing bandit offline IPS with SB3 online returns.",
            ),
            worked_example_pattern=(
                "pip install 'buildml[rl-industry]'; "
                "fit_rl(mode='gym_sb3', algorithm='ppo', total_timesteps=25000).",
            ),
            related_concepts=(
                "rl-gym-reinforce",
                "rl-tabular-q-learning",
                "rl-offline-metrics",
                "rl-bundle-boundary",
            ),
        ),
        _note(
            key="rl-bundle-boundary",
            title="RL bundle vs Session checkpoint",
            summary="buildml.rl_bundle.v1 stores RlPlan; checkpoints do not embed the policy.",
            definition=(
                "An RL bundle directory holds meta.json + rl_plan.joblib under "
                "schema buildml.rl_bundle.v1 (bandit or gym policy)."
            ),
            intuition="Save the RL policy separately from workflow resume state.",
            formal_idea="RlPlan is not embedded in a Session checkpoint payload.",
            why_it_matters=("Avoid silent gaps when reloading workflows.",),
            how_buildml_uses=("save_rl_bundle / load_rl_bundle.",),
            interpretation_rules=(
                "Reload policy via load_rl_bundle after checkpoint_load.",
            ),
            assumptions=("Bundle format matches buildml.rl_bundle.v1.",),
            failure_modes=("Mixing RL bundles with imitation / CBR / RAG bundles.",),
            anti_patterns=("Expecting checkpoint_load to restore the RL policy.",),
            worked_example_pattern=("save_rl_bundle(path) → load_rl_bundle(path).",),
            related_concepts=(
                "rl-contextual-bandit",
                "rl-gym-reinforce",
                "imitation-bundle-boundary",
            ),
        ),
    )
}
