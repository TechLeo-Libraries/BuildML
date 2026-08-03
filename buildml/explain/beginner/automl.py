# ruff: noqa: E501
"""Beginner layers for automated model and preprocessing search."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

AUTOML_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "automl-beyond-hpo",
        plain=(
            "Hyperparameter search tunes the dials on one model you already chose. AutoML goes a level up: "
            "it tries different *kinds* of model and different preprocessing recipes, and picks the whole "
            "combination. It answers 'what should I build?', not just 'what settings?'."
        ),
        analogy=(
            "Tuning is adjusting the seat and mirrors of the car you already own. AutoML is test-driving "
            "several cars, with several tyre choices, before deciding which to buy."
        ),
        steps=(
            "Set up your roles and split first — AutoML runs inside the training partition.",
            "Choose a time or trial budget, because the search space is large and you are buying compute.",
            "Let AutoML evaluate candidate model families, each with its own settings and preprocessing choices.",
            "Read the leaderboard: what won, by how much, and how close the runners-up were.",
            "Confirm the frozen winner once on test.",
        ),
        use=(
            "At the start of a project when you have no strong prior about which model family suits the data.",
            "As a strong baseline to check whether your hand-built pipeline is actually earning its keep.",
        ),
        avoid=(
            "Do not use it when you already know the model family and only need settings — `grid_search` or `optuna_search` is cheaper and clearer.",
            "Do not use it as a substitute for understanding the data; it will happily optimize a leaked feature to a spectacular score.",
        ),
        myths=(
            (
                "AutoML removes the need to understand the problem.",
                "It searches within the frame you gave it. Wrong target, wrong split boundary, or a leaked column will be amplified, not caught.",
            ),
            (
                "AutoML always beats a hand-built model.",
                "It usually beats a careless one. A domain expert with good features frequently beats an unaided search.",
            ),
        ),
        example=(
            "session.run_automl(",
            "    time_budget=300, selection='cv', cv=5, random_state=0,",
            ")",
            "print(session.automl_plan.leaderboard[:5])",
            "session.evaluate_automl(partition='test')",
        ),
        check=(
            "How much better is the winner than the second place, relative to fold noise?",
            "Would your top feature survive a leakage review?",
        ),
        tools=("run_automl", "evaluate_automl", "grid_search", "optuna_search"),
        terms=("AutoML", "hyperparameter", "grid search", "cross-validation"),
        difficulty=CORE,
    ),
    _layer(
        "automl-recipe-strategy-search",
        plain=(
            "A preprocessing recipe is a bundle of choices: which imputation, which scaler, which encoder, "
            "which feature selection. AutoML can treat those choices as part of the search, refitting the "
            "whole recipe inside every fold so the comparison stays honest."
        ),
        analogy=(
            "Testing not just which cake recipe wins, but which oven temperature and tin each recipe needs. "
            "And baking a fresh cake each round rather than reusing yesterday's batter."
        ),
        steps=(
            "Start from unpoisoned data — no Session-wide impute, encode, or scale already applied.",
            "Define which preprocessing strategies are candidates.",
            "For each fold, AutoML fits the recipe on that fold's training rows only.",
            "The candidate's score reflects the model *and* its preparation together.",
            "The winning recipe is stored on the plan so scoring reproduces it exactly.",
        ),
        use=(
            "When you genuinely do not know whether median or mean imputation, one-hot or target encoding, suits your data.",
            "When preprocessing choices interact strongly with the model family — which they usually do.",
        ),
        avoid=(
            "Do not run recipe search after you already applied Session-global preprocessing; the fold-local fits would sit on top of leaked transforms.",
            "Do not expand the recipe space without expanding the budget — every extra option multiplies the search.",
        ),
        myths=(
            (
                "Preprocessing is neutral, so it can be done once up front.",
                "Imputers, encoders, and scalers all learn from data. Doing them once on everything leaks; doing them per fold is what makes the comparison meaningful.",
            ),
            (
                "The best recipe for one model is the best for all.",
                "Scaling transforms a kNN's fate and leaves a tree indifferent. That is precisely why the search is joint.",
            ),
        ),
        example=(
            "recipe = PreprocessRecipe(",
            "    impute=['median', 'most_frequent'],",
            "    encode=['onehot', 'target'],",
            "    scale=['standard', None],",
            ")",
            "session.run_automl(recipe=recipe, selection='cv', cv=5, random_state=0)",
        ),
        check=(
            "Is your data unpoisoned — no Session-global transforms applied before the search?",
            "How many recipe combinations does your budget actually allow?",
        ),
        tools=("run_automl", "evaluate_automl", "impute", "encode", "scale"),
        terms=("AutoML", "pipeline", "cross-validation", "leakage"),
        difficulty=ADVANCED,
    ),
    _layer(
        "automl-selection-honesty",
        plain=(
            "AutoML tries many candidates, and trying many things is exactly how you accidentally find "
            "something that only looks good. Selection mode controls how the winner is chosen — inside "
            "cross-validation, nested cross-validation, or a validation partition — while your test "
            "partition stays sealed."
        ),
        analogy=(
            "Roll enough dice and one will come up six three times in a row. That die is not lucky; you just "
            "rolled a lot. Honest selection is about not mistaking the winner of a big search for a great model."
        ),
        steps=(
            "Pick a selection mode: `cv` ranks by cross-validated score inside train, `validation` ranks on the validation partition, `nested` scores the whole search procedure.",
            "Run the search. Test is never touched during any of these modes.",
            "Read the leaderboard gap and the fold spread together.",
            "Freeze the winner.",
            "Score test once for the number you report.",
        ),
        use=(
            "Any time the search space is more than a handful of candidates.",
            "When you need to report a number that stands for the *method*, not just the lucky winner — that is what `nested` gives you.",
        ),
        avoid=(
            "Do not use `validation` mode and then also tune the threshold on the same validation rows without acknowledging the double use.",
            "Do not run AutoML repeatedly, peeking at test between runs; each peek burns the partition.",
        ),
        myths=(
            (
                "The winner's cross-validation score is its expected performance.",
                "It is the maximum of many noisy scores, so it is biased upward. Nested cross-validation or a clean holdout gives the honest figure.",
            ),
            (
                "A bigger search is always better.",
                "A bigger search increases both the chance of finding something genuinely good and the winner's optimism. You need the honest estimate to tell them apart.",
            ),
        ),
        example=(
            "session.run_automl(selection='nested', inner_cv=3, outer_cv=5, random_state=0)",
            "print(session.automl_plan.selection_disclosures)",
            "session.evaluate_automl(partition='test')   # once, after freezing",
        ),
        check=(
            "Which partition ranked your candidates, and which one produced your reported number?",
            "How large is the winner's lead compared with the spread across folds?",
        ),
        tools=("run_automl", "evaluate_automl", "nested_cv_score", "evaluate"),
        terms=("AutoML", "cross-validation", "nested cross-validation", "validation", "test"),
        difficulty=ADVANCED,
    ),
    _layer(
        "automl-bundle-boundary",
        plain=(
            "The result of an AutoML run — the winning model, its recipe, the leaderboard, and the honesty "
            "disclosures — is saved as its own bundle. A Session checkpoint does not contain it."
        ),
        analogy=(
            "The tournament results sheet is a separate document from the venue booking. Keeping the venue "
            "details does not tell you who won."
        ),
        steps=(
            "Run AutoML so a plan exists on the Session.",
            "Call `save_automl_bundle(path)` to persist the winner, the recipe, and the disclosures.",
            "Reload with `load_automl_bundle(path)`.",
            "Evaluate or predict with the restored plan.",
            "Save a checkpoint too if you also want the data state that produced it.",
        ),
        use=(
            "When the AutoML winner is going into production and the recipe must travel with it.",
            "For audit: the disclosures record how the winner was selected, which is part of the claim.",
        ),
        avoid=(
            "Do not rebuild the winner by hand from the leaderboard; the recipe and its fitted parameters are what make it reproducible.",
            "Do not assume loading the bundle restores your dataset — it does not.",
        ),
        myths=(
            (
                "The leaderboard is enough to recreate the model.",
                "The leaderboard names the configuration. The fitted parameters, recipe plans, and preprocessing state live in the bundle.",
            ),
            (
                "Checkpoints and bundles overlap, so one is redundant.",
                "They answer different questions: 'what was my data workflow' versus 'what model did I end up with'.",
            ),
        ),
        example=(
            "session.save_automl_bundle('artifacts/automl')",
            "restored = Session.ingest(frame).load_automl_bundle('artifacts/automl')",
            "restored.evaluate_automl(partition='test')",
        ),
        check=(
            "Does your saved bundle include the preprocessing recipe the winner needs?",
            "Can you show a reviewer how the winner was selected?",
        ),
        tools=("save_automl_bundle", "load_automl_bundle", "run_automl", "checkpoint_save"),
        terms=("bundle", "AutoML", "checkpoint", "disclosure"),
        difficulty=CORE,
    ),
    _layer(
        "automl-industry-backends",
        plain=(
            "BuildML's own AutoML searches fold-local recipes. If you install the optional industry extras "
            "you can instead delegate the search to FLAML or AutoGluon, which bring their own tuned search "
            "strategies. BuildML still hands them training rows only and records what it could not control."
        ),
        analogy=(
            "You can plan the trip yourself or hand it to a travel agent. The agent is often faster and "
            "knows tricks you do not — but you still choose the destination and you should still read the itinerary."
        ),
        steps=(
            "Install the relevant extra, for example `pip install buildml[automl-industry]`.",
            "Pass `backend='flaml'` or `backend='autogluon'` to `run_automl`.",
            "BuildML passes training-partition data only and applies its own split boundary.",
            "The adapter's internal search runs under its own rules; BuildML records the limits it cannot enforce as disclosures.",
            "Evaluate and bundle exactly as you would with the native backend.",
        ),
        use=(
            "When you want a strong result quickly and the extra dependency is acceptable.",
            "When you are benchmarking BuildML's native search against an established tool.",
        ),
        avoid=(
            "Do not use an industry backend when you need full visibility into every fold decision; the native backend is the transparent one.",
            "Do not install it 'just in case' — extras add weight and version constraints.",
        ),
        myths=(
            (
                "An industry backend is always better.",
                "It is usually stronger per unit of compute on generic tabular tasks. On small or unusual data, the native recipe search is frequently competitive and always more inspectable.",
            ),
            (
                "Delegating the search delegates the leakage responsibility.",
                "BuildML controls what data goes in. What the adapter does internally is disclosed, not eliminated — you still own the split design.",
            ),
        ),
        example=(
            "# pip install \"buildml[automl-industry]\"",
            "session.run_automl(backend='flaml', time_budget=300, random_state=0)",
            "print(session.automl_plan.disclosures)",
        ),
        check=(
            "Is the extra installed, and does `automl_capability_matrix()` confirm the backend is available?",
            "Which disclosures did the adapter produce, and do any of them affect your claim?",
        ),
        tools=("run_automl", "evaluate_automl", "save_automl_bundle"),
        terms=("AutoML", "extra", "disclosure", "gradient boosting"),
        difficulty=ADVANCED,
    ),
)

__all__ = ["AUTOML_BEGINNER"]
