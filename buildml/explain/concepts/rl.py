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
            related_concepts=("rl-contextual-bandit", "rl-bundle-boundary"),
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
            formal_idea="π_θ learned by PPO/DQN/A2C env interaction; evaluate via mean_return.",
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
            related_concepts=("rl-gym-reinforce", "rl-offline-metrics", "rl-bundle-boundary"),
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
