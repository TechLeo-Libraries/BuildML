# ruff: noqa: E501
"""Beginner layers for causal inference."""

from __future__ import annotations

from buildml.explain.beginner._builder import (
    ADVANCED,
    CORE,
    FOUNDATION,
    BeginnerLayer,
    _index,
    _layer,
)

CAUSAL_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "causal-assumptions",
        plain=(
            "Causal questions ask what *would happen if* you intervened, which is a much stronger claim "
            "than noticing that two things move together. You cannot get there from data alone: you need "
            "assumptions. BuildML makes you write them down before it will estimate anything."
        ),
        analogy=(
            "You cannot work out whether the medicine helped just by looking at who recovered, because "
            "healthier people may have taken it. You have to state what else you measured and why you "
            "believe that is enough."
        ),
        steps=(
            "Name the treatment: the thing that was done, such as a discount sent or a drug given.",
            "Name the outcome: the thing you think it affected.",
            "List the confounders: everything that plausibly influenced both the treatment and the outcome.",
            "State the estimand: the specific quantity you want, such as the average treatment effect.",
            "Acknowledge unconfoundedness (you measured all the common causes) and positivity (every kind of row could have received either treatment).",
        ),
        use=(
            "Any time someone asks 'did this work?' rather than 'can we predict this?'.",
            "Before running a causal estimator at all: BuildML refuses without declared assumptions on purpose.",
        ),
        avoid=(
            "Do not use this path for pure prediction; a classifier does not need confounders declared and will be simpler.",
            "Do not declare assumptions you do not believe just to make the API run. The estimate inherits every assumption you sign.",
        ),
        myths=(
            (
                "With enough data, causality falls out of the numbers.",
                "No amount of observational data distinguishes 'the treatment worked' from 'the people who got it were already going to do better'. Only assumptions or randomization can.",
            ),
            (
                "Adding every available column as a confounder is the safe choice.",
                "Some columns are colliders or mediators, and adjusting for them actively introduces bias. Confounder choice is a modelling judgement, not a data-cleaning step.",
            ),
        ),
        example=(
            "session.declare_causal_assumptions(",
            "    treatment='received_discount',",
            "    outcome='renewed',",
            "    confounders=['tenure_months', 'plan_tier', 'prior_usage'],",
            "    estimand='ate',",
            "    unconfoundedness_ack=True, positivity_ack=True,",
            ")",
        ),
        check=(
            "Can you name a plausible cause of both treatment and outcome that you did not measure?",
            "Was any of your confounders recorded *after* the treatment?",
        ),
        tools=("declare_causal_assumptions", "fit_causal", "estimate_causal", "refute_causal"),
        terms=("causal inference", "treatment", "confounder", "unconfoundedness", "positivity", "ATE"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "causal-ate-backdoor",
        plain=(
            "The average treatment effect is the difference between two imaginary worlds: one where "
            "everybody got the treatment and one where nobody did. Backdoor adjustment estimates it by "
            "comparing treated and untreated rows that look alike on your declared confounders."
        ),
        analogy=(
            "Comparing exam results between students who did and did not attend a revision class: but only "
            "comparing students with the same prior grades, same subject, and same attendance record."
        ),
        steps=(
            "Declare your treatment, outcome, and the confounder set that closes the backdoor paths.",
            "BuildML groups or models rows so that treated and untreated cases become comparable on those confounders.",
            "It computes the average difference in outcome across that adjusted comparison.",
            "The result is an ATE with a confidence interval, plus the disclosures attached to it.",
            "Run refutation checks to see how fragile the estimate is.",
        ),
        use=(
            "When you have observational data, a clearly defined treatment, and a defensible confounder set.",
            "When randomization was impossible or already happened and you are analyzing the results.",
        ),
        avoid=(
            "Do not use it when the important confounders were not measured: the estimate will be confidently wrong.",
            "Do not use it for instrumental-variable, front-door, or causal-discovery questions; this surface only does backdoor adjustment and says so.",
        ),
        myths=(
            (
                "A causal estimate is more reliable than a predictive one because it is 'deeper'.",
                "It rests on untestable assumptions. A predictive model's honesty can be checked on holdout data; a causal claim's central assumption cannot be checked at all.",
            ),
            (
                "A narrow confidence interval means the estimate is trustworthy.",
                "The interval reflects sampling noise only. It says nothing about bias from an unmeasured confounder, which is usually the bigger problem.",
            ),
        ),
        example=(
            "session.fit_causal(method='aipw', random_state=0)",
            "estimate = session.estimate_causal()",
            "print(estimate.ate, estimate.confidence_interval)",
            "print(estimate.disclosures)",
        ),
        check=(
            "Which backdoor paths do your confounders close, and which stay open?",
            "How large would an unmeasured confounder need to be to erase your effect?",
        ),
        tools=("fit_causal", "estimate_causal", "refute_causal", "declare_causal_assumptions"),
        terms=("ATE", "confounder", "causal inference", "treatment", "confidence interval"),
        difficulty=CORE,
    ),
    _layer(
        "causal-t-learner",
        plain=(
            "The T-learner is the most intuitive causal estimator. Train one ordinary model on the treated "
            "rows and a second on the untreated rows. For every row, ask both models what they predict, and "
            "the average gap between their answers is your estimated effect."
        ),
        analogy=(
            "Two forecasters, one who only studied sunny regions and one who only studied rainy ones. For "
            "any given place you ask both what they expect, and the difference is your estimate of what "
            "the climate does."
        ),
        steps=(
            "Split rows into the treated arm and the control arm.",
            "Fit an outcome model separately on each arm using the confounders as features.",
            "For every row in the data, predict the outcome under both models.",
            "Take the average of the per-row differences: that is the estimated ATE.",
            "Check that both arms had enough rows; a thin arm makes its model unreliable.",
        ),
        use=(
            "When both treatment arms are reasonably large and the outcome is easy to model well.",
            "As a first, easily explained estimator before trying anything doubly robust.",
        ),
        avoid=(
            "Do not use it when one arm is tiny: the model for that arm will be extrapolating far beyond its data.",
            "Do not use it when the two arms barely overlap in feature space; both models will be predicting for rows unlike anything they saw.",
        ),
        myths=(
            (
                "Two models are always better than one.",
                "Splitting the data halves each model's training set, and any bias in either arm's model flows straight into the effect estimate.",
            ),
            (
                "A good predictive fit means a good causal estimate.",
                "Prediction accuracy in each arm does not guarantee unbiased differences, especially where the arms do not overlap.",
            ),
        ),
        example=(
            "session.fit_causal(method='t_learner', random_state=0)",
            "estimate = session.estimate_causal()",
            "print(estimate.ate, estimate.per_arm_n)",
        ),
        check=(
            "How many rows are in your smaller treatment arm?",
            "Do the two arms cover the same ranges of your confounders?",
        ),
        tools=("fit_causal", "estimate_causal", "declare_causal_assumptions"),
        terms=("ATE", "treatment", "confounder", "causal inference"),
        difficulty=CORE,
    ),
    _layer(
        "causal-ipw",
        plain=(
            "Inverse propensity weighting takes a different route. Instead of modelling the outcome, it "
            "models who was likely to get the treatment, then re-weights rows so the treated and untreated "
            "groups become statistically comparable. Rare-but-treated rows count for more."
        ),
        analogy=(
            "A survey that under-sampled young people. You give each young respondent extra weight so the "
            "totals reflect the real population rather than who happened to answer."
        ),
        steps=(
            "Fit a propensity model: given the confounders, how likely was this row to receive treatment?",
            "Weight each treated row by 1 divided by its propensity, and each control row by 1 divided by one minus its propensity.",
            "Compute the weighted average outcome in each group.",
            "The difference is the ATE.",
            "Clip extreme propensities, because a propensity of 0.001 produces a weight of 1000 and one row dominates everything.",
        ),
        use=(
            "When the treatment assignment mechanism is easier to model than the outcome.",
            "When you want an estimator whose assumptions are concentrated in one clearly inspectable model.",
        ),
        avoid=(
            "Do not use it when propensities pile up near 0 or 1: that is a positivity violation and no amount of clipping fixes it honestly.",
            "Do not use it alone when you can use a doubly robust method instead; AIPW gives you two chances to be right.",
        ),
        myths=(
            (
                "Clipping extreme weights makes the estimate safe.",
                "Clipping trades variance for bias. It stabilizes the number without restoring the comparability that was missing.",
            ),
            (
                "A well-fitting propensity model means good balance.",
                "A propensity model that predicts treatment perfectly is a disaster: it means the groups do not overlap at all.",
            ),
        ),
        example=(
            "session.fit_causal(method='ipw', propensity_clip=(0.01, 0.99), random_state=0)",
            "estimate = session.estimate_causal()",
            "print(estimate.ate, estimate.effective_sample_size)",
        ),
        check=(
            "What is the distribution of your propensity scores: is anything below 0.05 or above 0.95?",
            "What is your effective sample size after weighting, compared with your row count?",
        ),
        tools=("fit_causal", "estimate_causal", "refute_causal"),
        terms=("IPW", "propensity", "positivity", "ATE", "confounder"),
        difficulty=ADVANCED,
    ),
    _layer(
        "causal-aipw",
        plain=(
            "AIPW combines the two previous approaches: it models the outcome *and* the treatment, then "
            "corrects each with the other. The payoff is called double robustness: if either of the two "
            "models is right, the effect estimate stays honest even if the other is wrong."
        ),
        analogy=(
            "Two independent safety checks on the same aircraft. You are not relying on both being perfect; "
            "you are relying on not both failing in the same way."
        ),
        steps=(
            "Fit an outcome model on the confounders, as in the T-learner.",
            "Fit a propensity model, as in IPW.",
            "Combine them: start from the outcome model's prediction, then add a propensity-weighted correction for its residuals.",
            "Average across rows to get the ATE.",
            "Read the disclosures: double robustness protects against misspecification, not against unmeasured confounding.",
        ),
        use=(
            "As the default choice when you are unsure which of the two nuisance models you trust more.",
            "When the estimate will be scrutinized and you want the strongest defensible observational method available here.",
        ),
        avoid=(
            "Do not use it as a substitute for good confounder selection; if a common cause is missing, both models are wrong in the same direction and double robustness does nothing.",
            "Do not use it on very small samples, where fitting two nuisance models leaves too little signal for either.",
        ),
        myths=(
            (
                "Doubly robust means twice as reliable.",
                "It means consistent if *either* model is correctly specified. If both are wrong, it can be worse than either alone.",
            ),
            (
                "Double robustness protects against unmeasured confounding.",
                "It does not. Nothing statistical does. That is what refutation checks and sensitivity analysis are for.",
            ),
        ),
        example=(
            "session.fit_causal(method='aipw', random_state=0)",
            "estimate = session.estimate_causal()",
            "refutation = session.refute_causal(method='placebo_treatment')",
            "print(estimate.ate, refutation.passed)",
        ),
        check=(
            "Which of your two nuisance models do you actually believe, and why?",
            "Does the estimate survive a placebo-treatment refutation?",
        ),
        tools=("fit_causal", "estimate_causal", "refute_causal"),
        terms=("doubly robust", "IPW", "propensity", "ATE", "confounder"),
        difficulty=ADVANCED,
    ),
    _layer(
        "causal-eda-boundary",
        plain=(
            "BuildML's exploratory tools: correlations, mutual information, feature importance, the "
            "Teaching Studio: deliberately refuse to make causal claims. They describe association. "
            "Causal statements only come from the declared-assumption path, and that separation is enforced."
        ),
        analogy=(
            "A thermometer tells you the room is warm. It does not tell you the heater caused it. Reading "
            "causation off a thermometer is a category error, however good the thermometer is."
        ),
        steps=(
            "During exploration, read correlations and importances as prompts for questions.",
            "Notice that the reports say 'associated with', never 'causes'. That wording is deliberate.",
            "When someone asks a causal question, move to the causal path and declare assumptions.",
            "Keep the two clearly separated in your write-up so readers know which claim they are reading.",
            "If assumptions cannot be defended, the honest answer is that the question cannot be answered from this data.",
        ),
        use=(
            "EDA to generate hypotheses, spot data problems, and understand your dataset.",
            "The causal path when a decision depends on what would change if you intervened.",
        ),
        avoid=(
            "Do not present a feature importance chart as evidence that changing that feature changes the outcome.",
            "Do not skip EDA before causal work: you still need to understand the data, you just cannot conclude from it.",
        ),
        myths=(
            (
                "A very strong correlation is basically causal.",
                "Strength is not evidence of direction or mechanism. A strong correlation with an unmeasured common cause is exactly as strong as a real effect.",
            ),
            (
                "Feature importance shows which levers to pull.",
                "It shows what the model relied on. A model can rely heavily on a downstream consequence of the outcome, which is the opposite of a lever.",
            ),
        ),
        example=(
            "report = session.eda()             # association only, by design",
            "# for a causal question, switch surfaces:",
            "session.declare_causal_assumptions(treatment='...', outcome='...', confounders=[...])",
            "session.fit_causal(method='aipw')",
        ),
        check=(
            "Is the question you are answering predictive or interventional?",
            "Which sentence in your report would change if the arrow ran the other way?",
        ),
        tools=("eda", "feature_importance", "declare_causal_assumptions", "fit_causal"),
        terms=("causal inference", "correlation", "feature importance", "confounder"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "causal-bundle-boundary",
        plain=(
            "A causal bundle stores the declared assumptions alongside the fitted nuisance models and the "
            "estimates. The assumptions are the important part: an effect number without the assumptions "
            "that produced it is not interpretable."
        ),
        analogy=(
            "A lab result is meaningless without the protocol. Storing the number and throwing away the "
            "method leaves you with a figure nobody can defend."
        ),
        steps=(
            "Declare assumptions and fit a causal plan.",
            "Call `save_causal_bundle(path)`: assumptions, nuisance models, and estimates travel together.",
            "Reload with `load_causal_bundle(path)`.",
            "Re-read the assumptions before quoting the estimate anywhere.",
            "Keep checkpoints separately for the data state.",
        ),
        use=(
            "When a causal estimate will be reviewed, audited, or cited later.",
            "When the same analysis will be rerun on new data and must use identical assumptions.",
        ),
        avoid=(
            "Do not extract the ATE into a slide without carrying the assumptions with it.",
            "Do not expect a Session checkpoint to contain the causal plan; it does not.",
        ),
        myths=(
            (
                "The effect size is the deliverable.",
                "The deliverable is the effect size *plus* the assumptions and refutations that make it credible. The number alone is unfalsifiable.",
            ),
            (
                "Assumptions can be reconstructed from the code later.",
                "Confounder choices reflect domain reasoning that lives in someone's head. Bundling them is how that reasoning survives.",
            ),
        ),
        example=(
            "session.save_causal_bundle('artifacts/discount-effect')",
            "review = Session.ingest(frame).load_causal_bundle('artifacts/discount-effect')",
            "print(review.causal_plan.assumptions)",
        ),
        check=(
            "Could a reviewer reconstruct your confounder set from the artifact alone?",
            "Are your refutation results stored with the estimate?",
        ),
        tools=("save_causal_bundle", "load_causal_bundle", "declare_causal_assumptions", "checkpoint_save"),
        terms=("bundle", "checkpoint", "confounder", "causal inference"),
        difficulty=CORE,
    ),
    _layer(
        "causal-dowhy",
        plain=(
            "DoWhy is an established causal library. With the optional extra installed, BuildML can build a "
            "causal graph from your declared confounders, use DoWhy to identify the estimand formally, and "
            "then run DoWhy's refutation tests against your estimate."
        ),
        analogy=(
            "Having a specialist review your reasoning with their own checklist. They may spot that your "
            "argument does not actually establish what you claimed, before you present it."
        ),
        steps=(
            "Install the extra: `pip install buildml[causal-industry]`.",
            "Declare assumptions as usual: they become the graph.",
            "Pass `backend='dowhy'` to `fit_causal`.",
            "DoWhy performs identification: does your graph actually let this estimand be estimated?",
            "Run refuters: placebo treatment, random common cause, data subset: and read whether the estimate holds up.",
        ),
        use=(
            "When the estimate matters enough to justify formal identification and systematic refutation.",
            "When reviewers expect a recognized causal framework rather than a bespoke implementation.",
        ),
        avoid=(
            "Do not use it as a shortcut past thinking about confounders; the graph is built from what you declared.",
            "Do not install the extra for a one-off exploratory estimate.",
        ),
        myths=(
            (
                "Refutation tests prove the estimate is correct.",
                "They try to break it. Surviving them is reassuring; it is not proof. A placebo test cannot detect an unmeasured confounder.",
            ),
            (
                "Identification means the answer is right.",
                "Identification means the quantity is *estimable given your graph*. If the graph is wrong, identification succeeds and the answer is still wrong.",
            ),
        ),
        example=(
            "# pip install \"buildml[causal-industry]\"",
            "session.fit_causal(backend='dowhy', method='backdoor.propensity_score_matching')",
            "print(session.refute_causal(method='random_common_cause'))",
            "print(session.refute_causal(method='placebo_treatment'))",
        ),
        check=(
            "Did identification succeed, and under which adjustment set?",
            "Which refuters did you run, and did any of them move the estimate materially?",
        ),
        tools=("fit_causal", "refute_causal", "estimate_causal", "declare_causal_assumptions"),
        terms=("causal inference", "confounder", "ATE", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "causal-econml",
        plain=(
            "EconML brings machine-learning-powered causal estimators. Where the basic methods give you one "
            "average effect, these can estimate how the effect *varies* across rows: who benefits most: "
            "and can turn that into a targeting policy."
        ),
        analogy=(
            "Instead of 'the medicine helps by 3 points on average', you learn that it helps older patients "
            "a lot and younger ones barely at all. That changes who you give it to."
        ),
        steps=(
            "Install `buildml[causal-industry]` and declare assumptions as usual.",
            "Pass `backend='econml'` with a method such as double machine learning or causal forest.",
            "The estimator learns a per-row treatment effect rather than a single average.",
            "Inspect how the effect varies across your confounders.",
            "Optionally learn a policy tree: an interpretable rule for who should be treated.",
        ),
        use=(
            "When the decision is 'who should we target' rather than 'does this work on average'.",
            "When you suspect strong heterogeneity and averaging would hide it.",
        ),
        avoid=(
            "Do not chase heterogeneity on small samples: subgroup effect estimates are noisy long before the average is.",
            "Do not use it before you trust the average effect; heterogeneity built on a biased average is heterogeneity in the bias.",
        ),
        myths=(
            (
                "Heterogeneous effects are just subgroup analyses done properly.",
                "They are estimated jointly with regularization, which controls the multiple-comparisons problem that makes naive subgroup hunting so unreliable.",
            ),
            (
                "A policy tree tells you the optimal policy.",
                "It tells you the best rule under your assumptions and your sample. It inherits every limitation of the underlying causal estimate.",
            ),
        ),
        example=(
            "# pip install \"buildml[causal-industry]\"",
            "session.fit_causal(backend='econml', method='causal_forest', random_state=0)",
            "estimate = session.estimate_causal()",
            "print(estimate.ate, estimate.cate_summary)",
        ),
        check=(
            "How many rows sit in your smallest interesting subgroup?",
            "Does the average effect itself survive refutation before you split it?",
        ),
        tools=("fit_causal", "estimate_causal", "refute_causal", "fit_decision_policy"),
        terms=("causal inference", "ATE", "treatment", "policy", "extra"),
        difficulty=ADVANCED,
    ),
)

__all__ = ["CAUSAL_BEGINNER"]
