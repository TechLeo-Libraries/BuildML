# ruff: noqa: E501
"""Beginner layers for imitation learning and reinforcement learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

RL_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "imitation-behavioral-cloning",
        plain=(
            "Behavioral cloning is the simplest way to learn a decision policy: treat recorded expert "
            "decisions as ordinary training data. The situation becomes the features, the action the expert "
            "took becomes the label, and you fit a normal supervised model."
        ),
        analogy=(
            "Learning to drive by watching thousands of hours of a good driver and copying what they do in "
            "each situation. You never work out *why*: you just reproduce the behaviour."
        ),
        steps=(
            "Assemble a table of demonstrations: one row per decision, with the situation described by features and the chosen action as the target.",
            "Split as usual, respecting any episode or session grouping so one episode does not straddle the boundary.",
            "Fit an ordinary classifier or regressor on the training demonstrations.",
            "Predict actions for new situations.",
            "Evaluate on held-out demonstrations: you are measuring agreement with the expert, not real-world outcome quality.",
        ),
        use=(
            "When you have a substantial log of good decisions and no safe way to experiment.",
            "As a starting policy that a reinforcement-learning method can later improve on.",
        ),
        avoid=(
            "Do not use it when your demonstrations are of mediocre decisions; the policy will faithfully clone the mediocrity.",
            "Do not deploy it into situations unlike anything in the demonstrations: a cloned policy has no idea what to do off-distribution.",
        ),
        myths=(
            (
                "Behavioral cloning is reinforcement learning.",
                "It is supervised learning with actions as labels. No rewards, no exploration, no long-term planning.",
            ),
            (
                "A policy matching the expert 95% of the time performs 95% as well.",
                "Small errors compound. One wrong action moves you to a state the expert never visited, where the policy is guessing, and the trajectory drifts away.",
            ),
        ),
        example=(
            "session.set_roles({'action_taken': 'target', 'episode_id': 'group'})",
            "session.group_split(group_column='episode_id', test_size=0.2, random_state=0)",
            "session.rl.fit_imitation(estimator=HistGradientBoostingClassifier())",
            "session.rl.evaluate_imitation(partition='test')",
        ),
        check=(
            "Were your demonstrations produced by someone genuinely good at the task?",
            "What does your policy do when it reaches a state absent from the logs?",
        ),
        tools=("fit_imitation", "predict_imitation_action", "evaluate_imitation", "group_split"),
        terms=("imitation learning", "policy", "agent", "supervised"),
        difficulty=CORE,
    ),
    _layer(
        "imitation-bundle-boundary",
        plain=(
            "The cloned policy saves as an imitation bundle: the fitted model, the action vocabulary, and "
            "the feature contract. Session checkpoints hold your demonstration data, not the policy."
        ),
        analogy=(
            "The trainee's learned habits are not the same artifact as the video library they learned from. "
            "You can keep one without the other."
        ),
        steps=(
            "Fit an imitation policy so a plan exists.",
            "Call `session.rl.save_imitation_bundle(path)`.",
            "Reload with `session.rl.load_imitation_bundle(path)` wherever decisions are served.",
            "Call `session.rl.predict_imitation` with the current situation's features.",
            "Keep checkpoints separately for the demonstration dataset.",
        ),
        use=(
            "When the policy runs in a decision service outside your notebook.",
            "When the action vocabulary must be pinned so downstream systems can map outputs reliably.",
        ),
        avoid=(
            "Do not deploy the policy without also recording the demonstration period it came from; behaviour policies age quickly.",
            "Do not expect the bundle to contain the demonstrations themselves.",
        ),
        myths=(
            (
                "A policy bundle behaves like a model bundle.",
                "Structurally it is similar; contractually it carries the action vocabulary and situation contract, which a generic model bundle does not enforce.",
            ),
            (
                "Once saved, the policy stays valid.",
                "The world the demonstrations came from changes. A cloned policy encodes the past and needs periodic re-cloning.",
            ),
        ),
        example=(
            "session.rl.save_imitation_bundle('artifacts/routing-policy')",
            "service = Session.ingest(state_frame).rl.load_imitation_bundle('artifacts/routing-policy')",
            "action = service.rl.predict_imitation()",
        ),
        check=(
            "When were your demonstrations recorded, and is that behaviour still current?",
            "Which artifact holds the logs, and which holds the policy?",
        ),
        tools=("save_imitation_bundle", "load_imitation_bundle", "predict_imitation_action", "checkpoint_save"),
        terms=("bundle", "checkpoint", "policy", "imitation learning"),
        difficulty=CORE,
    ),
    _layer(
        "rl-contextual-bandit",
        plain=(
            "A contextual bandit is the simplest genuine decision-learning setting. You see a situation, "
            "pick one of a few actions, and observe a reward for the action you picked only: never for the "
            "ones you did not. There is no long-term state to plan through."
        ),
        analogy=(
            "Choosing which of three headlines to show a visitor. You learn whether the one you chose was "
            "clicked. You never find out what would have happened with the other two."
        ),
        steps=(
            "Assemble logged rows of (context features, action taken, reward observed).",
            "Choose a policy family: LinUCB adds an optimism bonus for uncertain actions; epsilon-greedy explores at random; softmax explores in proportion to estimated value.",
            "Fit on training rows only.",
            "Ask the policy for an action given a new context.",
            "Evaluate offline, with the strong caveats that come with counterfactual estimation.",
        ),
        use=(
            "Content selection, offer targeting, treatment assignment: anywhere you choose among a small set of options repeatedly.",
            "When you can log the action and reward together, which is the minimum requirement.",
        ),
        avoid=(
            "Do not use it when today's action changes tomorrow's situation; that is full reinforcement learning and bandits will get it wrong.",
            "Do not use it when your logs came from a deterministic policy: with no variation, there is nothing to learn about the untried actions.",
        ),
        myths=(
            (
                "A bandit is just a classifier over actions.",
                "A classifier needs to know the right answer for every row. A bandit only ever sees the reward for the action that was actually taken, which is a fundamentally harder problem.",
            ),
            (
                "Always taking the currently-best action is optimal.",
                "Pure exploitation locks in whatever your early data suggested. Without exploration you never discover that a rarely-tried action is better.",
            ),
        ),
        example=(
            "session.rl.fit(",
            "    method='linucb', context_columns=['segment', 'recency'],",
            "    action_column='offer_shown', reward_column='converted', alpha=1.0,",
            ")",
            "action = session.rl.act(context={'segment': 'A', 'recency': 3})",
        ),
        check=(
            "Did your logging policy try every action at least sometimes?",
            "Does the action you take now change the situation you will face later?",
        ),
        tools=("fit_rl", "act_rl", "evaluate_rl"),
        terms=("bandit", "reinforcement learning", "policy", "reward", "exploration"),
        difficulty=CORE,
    ),
    _layer(
        "rl-offline-metrics",
        plain=(
            "Evaluating a new policy from old logs is genuinely hard, because you only recorded what "
            "happened under the old policy. Offline estimators such as the direct method and inverse "
            "propensity scoring try to answer 'what would the new policy have earned?': with real "
            "limitations that BuildML states explicitly."
        ),
        analogy=(
            "Asking what your investment would be worth if you had bought different shares. You can "
            "estimate it, and the estimate is only as good as your assumptions about what else would have "
            "changed."
        ),
        steps=(
            "The direct method fits a reward model and asks it what the new policy's actions would have earned.",
            "Inverse propensity scoring re-weights logged rewards by how likely the new policy was to take the action that was actually taken.",
            "Both need the logging policy to have had some chance of taking the new policy's actions.",
            "Read the disclosures: variance is often enormous when the two policies disagree a lot.",
            "Treat the result as a screen before an online test, never as a substitute for one.",
        ),
        use=(
            "To rule out obviously bad policies before spending real traffic on them.",
            "When live experimentation is slow, expensive, or ethically constrained.",
        ),
        avoid=(
            "Do not report an offline estimate as expected live performance; the confidence intervals are usually far wider than they look.",
            "Do not use it when the new policy would take actions the logging policy essentially never took: there is no evidence to reweight.",
        ),
        myths=(
            (
                "Offline evaluation replaces A/B testing.",
                "It ranks candidates cheaply. It cannot capture feedback loops, novelty effects, or how users respond to a changed system.",
            ),
            (
                "IPS is unbiased, so it is reliable.",
                "It is unbiased and can have enormous variance. An unbiased estimator with a huge spread is not a usable number.",
            ),
        ),
        example=(
            "report = session.rl.evaluate(partition='test', estimator='ips')",
            "print(report.estimated_value, report.effective_sample_size)",
            "print(report.disclosures)",
        ),
        check=(
            "How often does your new policy pick an action the logs rarely contain?",
            "What is the effective sample size behind your estimate?",
        ),
        tools=("evaluate_rl", "fit_rl", "act_rl"),
        terms=("off-policy evaluation", "policy", "reward", "propensity", "bandit"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-gym-reinforce",
        plain=(
            "Full reinforcement learning needs an environment the agent can interact with, not just a table "
            "of logs. With the optional extra, BuildML can train a simple policy-gradient agent on small "
            "Gymnasium environments: a teaching surface for how the loop works, not a research platform."
        ),
        analogy=(
            "A driving simulator. Safe to crash in, cheap to repeat, and clearly not the same as the road."
        ),
        steps=(
            "Install `pip install buildml[rl]`: the core never requires Gymnasium.",
            "Choose a small discrete environment such as CartPole.",
            "The agent runs an episode, collecting states, actions, and rewards.",
            "REINFORCE increases the probability of actions that preceded good returns and decreases the rest.",
            "Repeat over many episodes; the reward curve should trend upward, noisily.",
        ),
        use=(
            "To learn how the reinforcement-learning loop actually works, hands on.",
            "For small toy control problems where a simple policy gradient is sufficient.",
        ),
        avoid=(
            "Do not use it for anything real-world; REINFORCE is high-variance and sample-inefficient.",
            "Do not use it when you have logged data but no environment: that is offline reinforcement learning, which is a different problem entirely.",
        ),
        myths=(
            (
                "Reinforcement learning works on my historical table.",
                "It needs an environment to try actions in and observe consequences. A static table supports bandits and offline evaluation, not interactive learning.",
            ),
            (
                "A rising reward curve means the agent learned the task.",
                "Reinforcement-learning curves are extremely noisy and seed-dependent. A single good run proves very little.",
            ),
        ),
        example=(
            "# pip install \"buildml[rl]\"",
            "session.rl.fit(",
            "    method='reinforce', env_id='CartPole-v1',",
            "    n_episodes=500, random_state=0,",
            ")",
            "session.rl.evaluate(n_eval_episodes=20)",
        ),
        check=(
            "Do you have an environment, or only logs?",
            "Does your reward curve hold up across several seeds?",
        ),
        tools=("fit_rl", "act_rl", "evaluate_rl"),
        terms=(
            "reinforcement learning",
            "agent",
            "policy",
            "policy gradient",
            "reward",
            "episode",
            "extra",
        ),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-sb3-industry",
        plain=(
            "Stable-Baselines3 provides well-tested implementations of the standard reinforcement-learning "
            "algorithms: PPO, DQN, A2C. When the industry extra is installed, BuildML routes to them by "
            "default because a correct implementation matters enormously in this field."
        ),
        analogy=(
            "Using a surveyed, published recipe rather than reconstructing one from memory. Reinforcement "
            "learning is notoriously sensitive to implementation details that never make it into papers."
        ),
        steps=(
            "Install `pip install buildml[rl-industry]`.",
            "Choose an algorithm: PPO as a robust default, DQN for discrete actions, A2C when you want something lighter.",
            "Set the training budget in timesteps rather than episodes.",
            "Train, then evaluate over many episodes because single-episode results are meaningless.",
            "Run several seeds; reinforcement-learning results vary enormously between them.",
        ),
        use=(
            "When you have a genuine environment and want results rather than a lesson in implementation.",
            "When you need something that has been validated against published benchmarks.",
        ),
        avoid=(
            "Do not use it on a static dataset; you still need an environment to step through.",
            "Do not report a single-seed result: the seed-to-seed variation frequently exceeds the difference between algorithms.",
        ),
        myths=(
            (
                "The algorithm choice is what determines success.",
                "Reward shaping, observation design, and hyperparameters usually matter more than PPO versus A2C.",
            ),
            (
                "Reinforcement learning is the natural fit for business decision problems.",
                "Most business decision problems are bandits or plain supervised learning wearing a costume. Full reinforcement learning needs sequential state that your actions genuinely change.",
            ),
        ),
        example=(
            "# pip install \"buildml[rl-industry]\"",
            "session.rl.fit(",
            "    method='ppo', env_id='CartPole-v1',",
            "    total_timesteps=50_000, random_state=0,",
            ")",
            "session.rl.evaluate(n_eval_episodes=50)",
        ),
        check=(
            "Do your actions change the state you will see next?",
            "How much do your results move across three different seeds?",
        ),
        tools=("fit_rl", "act_rl", "evaluate_rl", "save_rl_bundle"),
        terms=("reinforcement learning", "policy", "agent", "reward", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-bundle-boundary",
        plain=(
            "The learned policy: bandit weights or a trained agent: saves as an RL bundle. Session "
            "checkpoints hold your logged data and workflow state, never the policy."
        ),
        analogy=(
            "The trained pilot and the flight logs are different records. The logs explain how the training "
            "went; only the pilot can fly the next leg."
        ),
        steps=(
            "Fit a bandit or an agent so a plan exists.",
            "Call `session.rl.save_bundle(path)` to store the policy and its action space.",
            "Reload with `session.rl.load_bundle(path)` in the serving path.",
            "Call `session.rl.act` with a context to get an action.",
            "Keep checkpoints separately for the interaction logs.",
        ),
        use=(
            "When decisions are served continuously and the policy must survive restarts.",
            "When you need to compare a deployed policy against a newly trained one on the same logs.",
        ),
        avoid=(
            "Do not deploy an exploring policy without recording the exploration parameters; you need them to interpret the resulting logs later.",
            "Do not expect a checkpoint to restore the RL plan.",
        ),
        myths=(
            (
                "The bundle records the rewards it earned.",
                "It records the fitted policy. Reward history belongs to your logging system, and you will need it for the next round of offline evaluation.",
            ),
            (
                "A saved policy is frozen and therefore safe.",
                "A frozen exploring policy keeps exploring in production. Freezing the weights is not the same as freezing the behaviour.",
            ),
        ),
        example=(
            "session.rl.save_bundle('artifacts/offer-policy')",
            "service = Session().rl.load_bundle('artifacts/offer-policy')",
            "action = service.rl.act(context={'segment': 'A', 'recency': 3})",
        ),
        check=(
            "Is your deployed policy still exploring, and at what rate?",
            "Are you logging the action and the propensity so the next offline evaluation is possible?",
        ),
        tools=("save_rl_bundle", "load_rl_bundle", "act_rl", "checkpoint_save"),
        terms=("bundle", "checkpoint", "policy", "exploration"),
        difficulty=CORE,
    ),
    _layer(
        "rl-tabular-q-learning",
        plain=(
            "Q-learning keeps a table with one number for every combination of situation and action: how "
            "good is doing this here? After each step the agent nudges the entry it just used toward the "
            "reward it received plus the best value it believes is available next."
        ),
        analogy=(
            "A notebook of every junction you have driven through, with a score beside each turn. Every "
            "trip you update a few scores based on how the rest of the journey went."
        ),
        steps=(
            "Start with a table of zeros: the agent believes nothing about anything.",
            "Act mostly greedily but sometimes at random, so unexplored actions get tried.",
            "After each step, compute the target: the reward received plus the discounted best value of the next situation.",
            "Move the used entry a small step (`alpha`) toward that target.",
            "Repeat over many episodes; the table converges toward the values of acting optimally.",
        ),
        use=(
            "When the situations and actions are few enough to enumerate: grid worlds, small simulators, discretized problems.",
            "When you want a policy you can read: the whole thing is a table you can print.",
        ),
        avoid=(
            "Do not use it when the number of situations explodes; a table with a million rows will never be visited enough to learn.",
            "Do not use it when you cannot let the agent explore, which includes almost every real production setting.",
        ),
        myths=(
            (
                "Q-learning learns the policy it is following.",
                "It learns the value of acting optimally afterwards, no matter how badly it explored. That is what 'off-policy' means, and it is the difference from SARSA.",
            ),
            (
                "It is a toy compared with deep reinforcement learning.",
                "DQN is Q-learning with a neural network replacing the table, plus a replay buffer and a target network. Understanding this makes the rest legible.",
            ),
        ),
        example=(
            "# pip install \"buildml[rl]\"",
            "session.rl.fit(mode='tabular_q', algorithm='q_learning', env_id='FrozenLake-v1')",
            "print(session.rl.plan.greedy_policy_table())",
            "session.rl.evaluate(n_episodes=100)",
        ),
        check=(
            "What fraction of your table was ever updated? Low coverage means most of it is still zeros.",
            "How often does the greedy policy hit a situation it never saw in training?",
        ),
        tools=("fit_rl", "evaluate_rl", "act_rl"),
        terms=("reinforcement learning", "policy", "reward", "exploration", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-sarsa-on-policy",
        plain=(
            "SARSA is Q-learning's cautious sibling. Instead of assuming it will behave perfectly from the "
            "next step onward, it updates using the action it is actually going to take: exploration "
            "mistakes included. The result is a policy that accounts for its own fallibility."
        ),
        analogy=(
            "Planning a cycling route knowing you will occasionally wobble. The theoretically fastest line "
            "runs along the cliff edge; the route you should actually take does not."
        ),
        steps=(
            "Take an action and observe the reward and the next situation.",
            "Choose the next action using the same exploring policy you are following.",
            "Update using *that* action's value rather than the best available one.",
            "Expected SARSA improves on this by averaging over what the policy might do, removing the randomness of the single draw.",
            "As exploration decays, the learned values approach the optimal ones.",
        ),
        use=(
            "When exploration mistakes are costly and the policy should route around its own errors.",
            "When you want to understand the on-policy versus off-policy distinction concretely: this is the cleanest demonstration.",
        ),
        avoid=(
            "Do not use it with a high fixed exploration rate; the learned values stay permanently pessimistic.",
            "Do not compare its returns against Q-learning without matching seeds and environments: the difference is subtle and easily swamped by noise.",
        ),
        myths=(
            (
                "SARSA is just a worse Q-learning.",
                "On the classic cliff-walking problem it earns more reward, because it learns a route that survives the exploration it is actually doing.",
            ),
            (
                "The exploration rate is a training detail.",
                "For on-policy control it is part of what is being learned. Change epsilon and you change the objective, not just the search.",
            ),
        ),
        example=(
            "session.rl.fit(mode='tabular_q', algorithm='sarsa', env_id='CliffWalking-v0')",
            "session.rl.fit(mode='tabular_q', algorithm='expected_sarsa', env_id='CliffWalking-v0')",
            "# compare mean_return against algorithm='q_learning' with the same seed",
        ),
        check=(
            "Is your exploration rate decaying, or fixed forever?",
            "Does the on-policy result differ from off-policy on your environment, and does that difference make sense?",
        ),
        tools=("fit_rl", "evaluate_rl", "act_rl"),
        terms=("reinforcement learning", "policy", "exploration", "reward"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-state-discretization",
        plain=(
            "A lookup table needs a finite list of situations, but many environments report continuous "
            "measurements: position, angle, velocity. Discretization chops each measurement into a few "
            "buckets so every combination becomes one table row."
        ),
        analogy=(
            "Replacing a continuous dial with a switch that has five labelled notches. You lose precision "
            "and gain something you can actually list."
        ),
        steps=(
            "Each observation dimension is divided into `n_bins` equal buckets.",
            "Bucket numbers across dimensions combine into one integer state index.",
            "Where the environment declares finite bounds, BuildML uses them.",
            "Where it does not, a seeded random-policy probe estimates a sensible range and records that it did so.",
            "The table has `n_bins` to the power of the number of dimensions: BuildML refuses allocations above a hard cap.",
        ),
        use=(
            "When you want tabular methods on an environment with continuous measurements.",
            "As a teaching device for the curse of dimensionality: the table size is right there in front of you.",
        ),
        avoid=(
            "Do not add bins to improve accuracy without checking coverage; more bins means fewer visits each and nothing gets learned.",
            "Do not use it with many observation dimensions; this is exactly the situation function approximation exists to solve.",
        ),
        myths=(
            (
                "More bins always means a better policy.",
                "Table size grows exponentially. Doubling bins on a four-dimensional observation multiplies the table by sixteen, and each row gets visited a sixteenth as often.",
            ),
            (
                "Observations outside the modelled range are handled gracefully.",
                "They are clipped into the edge buckets. The agent cannot tell 'slightly past the edge' from 'far past the edge', which is a real failure mode under distribution shift.",
            ),
        ),
        example=(
            "session.rl.fit(mode='tabular_q', env_id='CartPole-v1', n_bins=6)",
            "disc = session.rl.plan.config['discretizer']",
            "print(disc['bound_sources'], disc['n_states'])",
        ),
        check=(
            "How many table rows did your bin choice create, and how many episodes will you run?",
            "Which dimensions used probed bounds rather than declared ones?",
        ),
        tools=("fit_rl", "evaluate_rl"),
        terms=("reinforcement learning", "discretization", "curse of dimensionality", "policy"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-monte-carlo-returns",
        plain=(
            "Monte Carlo methods wait until an episode ends, then credit each action with the "
            "actual return that followed. BuildML's gym_reinforce path uses these full-episode "
            "returns: unbiased but noisy compared with bootstrapped TD or actor-critic updates."
        ),
        analogy=(
            "Judge each chess move only after you see whether the game was won or lost: no "
            "guessing mid-game from a value table."
        ),
        steps=(
            "Install buildml[rl] so Gymnasium env loops are available.",
            "session.rl.fit(mode='gym_reinforce', env_id='CartPole-v1', n_episodes=...).",
            "Plot mean_return over episodes: expect high variance.",
            "Compare sample efficiency against tabular_q or gym_sb3 on the same env and seed.",
        ),
        use=(
            "Teaching policy gradients without introducing a value baseline first.",
            "Small discrete envs where full episodes are cheap to roll out.",
        ),
        avoid=(
            "Expecting sample efficiency matching PPO or DQN on the same timestep budget.",
            "Calling REINFORCE actor-critic: it has no critic network by default.",
        ),
        myths=(
            ("REINFORCE is actor-critic.", "It is Monte Carlo policy gradient without a critic."),
            ("Monte Carlo removes variance.", "It is unbiased but often high-variance versus TD."),
        ),
        example=(
            "session.rl.fit(mode='gym_reinforce', env_id='CartPole-v1', n_episodes=300)",
            "session.rl.evaluate()",
        ),
        check=(
            "Is mean_return trending up over episodes?",
            "Did you match seeds when comparing to tabular_q or gym_sb3?",
        ),
        tools=("fit_rl", "evaluate_rl", "rl_capability_matrix"),
        terms=("reinforcement learning", "policy gradient", "agent"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-n-step-td",
        plain=(
            "n-step TD blends several immediate rewards with a bootstrap value from your table. "
            "Q-learning and SARSA in tabular_q are the one-step case; increasing n moves targets "
            "toward full Monte Carlo returns while keeping some bootstrapping bias."
        ),
        analogy=(
            "Look a few steps ahead with real outcomes, then estimate the rest from your current "
            "scoreboard instead of waiting for the entire game to finish."
        ),
        steps=(
            "session.rl.fit(mode='tabular_q', algorithm='q_learning' or 'sarsa').",
            "Inspect state_coverage and mean_abs_td_error in fit results.",
            "Compare mean_return against the other algorithm on CliffWalking with the same seed.",
            "Use rl-n-step-td as the bridge when reading DQN target-network papers.",
        ),
        use=(
            "Understanding bootstrapping before reading DQN or PPO implementations.",
            "Teaching on-policy vs off-policy control with inspectable Q tables.",
        ),
        avoid=(
            "Calling tabular_q offline RL: it still interacts with the env online.",
            "Cranking n_bins until the state cap error instead of switching to function approximation.",
        ),
        myths=(
            ("Q-learning and SARSA must converge to the same policy.", "On-policy vs off-policy targets differ."),
            ("One-step TD is always lower variance than Monte Carlo.", "Bias-variance trade-off depends on n and noise."),
        ),
        example=(
            "session.rl.fit(mode='tabular_q', algorithm='sarsa', env_id='CliffWalking-v0')",
            "session.rl.evaluate()",
        ),
        check=(
            "Does state_coverage show enough repeated (state, action) visits?",
            "Did SARSA and Q-learning use the same epsilon schedule for a fair compare?",
        ),
        tools=("fit_rl", "evaluate_rl"),
        terms=("reinforcement learning", "policy", "reward"),
        difficulty=ADVANCED,
    ),
    _layer(
        "rl-actor-critic",
        plain=(
            "Actor-critic learns a policy (actor) and a value estimate (critic) together. The critic "
            "smooths learning versus raw Monte Carlo returns: SB3 PPO and A2C expose this industry "
            "path when buildml[rl-industry] is installed."
        ),
        analogy=(
            "A player (actor) chooses moves while a coach (critic) estimates how promising the "
            "position looks: smoother feedback than waiting for the final score alone."
        ),
        steps=(
            "pip install 'buildml[rl-industry]'.",
            "session.rl.fit(mode='gym_sb3', algorithm='ppo' or 'a2c', total_timesteps=...).",
            "session.rl.evaluate and read mean_return with offline=False disclosures.",
            "Contrast against gym_reinforce (MC) and tabular_q on the same env.",
        ),
        use=(
            "Industry teaching depth on CartPole-class discrete envs.",
            "Connecting REINFORCE and DQN ideas via a shared actor-critic framing.",
        ),
        avoid=(
            "Claiming SB3 CartPole training equals robotics or MuJoCo deployment.",
            "Confusing actor-critic online returns with bandit offline IPS metrics.",
        ),
        myths=(
            ("Actor-critic removes exploration noise.", "It reduces variance; exploration policy remains."),
            ("PPO is unrelated to REINFORCE.", "Both are policy-gradient families with different variance control."),
        ),
        example=(
            "session.rl.fit(mode='gym_sb3', algorithm='ppo', total_timesteps=25000)",
            "session.rl.evaluate()",
        ),
        check=(
            "Did you read session.rl.capability_matrix() for backend defaults?",
            "Does mean_return beat REINFORCE with matched seeds and timesteps?",
        ),
        tools=("fit_rl", "evaluate_rl", "rl_capability_matrix"),
        terms=("reinforcement learning", "policy gradient", "extra"),
        difficulty=ADVANCED,
    ),
)

__all__ = ["RL_BEGINNER"]
