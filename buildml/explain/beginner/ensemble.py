# ruff: noqa: E501
"""Beginner layers for voting, stacking, and blending ensembles."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

ENSEMBLE_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "ensemble-voting-vs-single-tree",
        plain=(
            "A voting ensemble asks several *different kinds* of model the same question and combines their "
            "answers. That is not the same thing as a random forest, which is many copies of one kind of "
            "model and which BuildML treats as a single estimator."
        ),
        analogy=(
            "A panel of a doctor, an engineer, and an accountant will disagree in different ways than a "
            "panel of ten accountants. Diversity of viewpoint is what makes the average worth more than "
            "any member."
        ),
        steps=(
            "Pick two or more models that make different kinds of mistakes: a linear model, a tree ensemble, a nearest-neighbour model.",
            "Choose how to combine them: hard voting takes a majority of the predicted classes, soft voting averages the predicted probabilities.",
            "Fit them all on the same training rows through `fit_voting`.",
            "Evaluate the ensemble against each individual member on validation.",
            "Keep it only if the combination genuinely beats the best member by more than fold-level noise.",
        ),
        use=(
            "When several models score similarly but you can see they fail on different rows.",
            "When you want a small, cheap robustness gain without building a stacking layer.",
        ),
        avoid=(
            "Do not combine models that are all variations of the same algorithm; their errors correlate and averaging buys almost nothing.",
            "Do not use it when one member is far better than the rest: averaging drags the good model toward the bad ones.",
        ),
        myths=(
            (
                "Passing a RandomForest to `fit` creates an ensemble in BuildML.",
                "A random forest is internally an ensemble, but to BuildML it is one estimator with one plan. Native voting is a separate surface with its own bundle.",
            ),
            (
                "More members always means a better ensemble.",
                "Adding a weak, correlated member usually hurts. Diversity matters far more than count.",
            ),
        ),
        example=(
            "session.fit_voting(",
            "    estimators={'logreg': LogisticRegression(max_iter=1000),",
            "                'forest': RandomForestClassifier(random_state=0)},",
            "    voting='soft',",
            ")",
            "session.evaluate_ensemble(partition='validation')",
        ),
        check=(
            "Do your members disagree on different rows, or make the same mistakes together?",
            "Does the ensemble beat your best single model by more than the cross-validation spread?",
        ),
        tools=("fit_voting", "evaluate_ensemble", "compare_models"),
        terms=("ensemble", "voting", "random forest", "predict_proba"),
        difficulty=CORE,
    ),
    _layer(
        "ensemble-stacking-oof",
        plain=(
            "Stacking trains a small extra model whose job is to learn how much to trust each base model. "
            "The catch is what you feed that extra model: it must see predictions made for rows the base "
            "model did not train on, which is why stacking uses cross-validation folds inside the training "
            "partition."
        ),
        analogy=(
            "A manager learning which specialist to believe. To learn that, they need to see each "
            "specialist's calls on cases the specialist had not already been told the answer to."
        ),
        steps=(
            "Choose your base models and one simple meta-model: logistic or linear regression is usually enough.",
            "BuildML splits the training rows into folds and, for each fold, trains the base models on the other folds and predicts the held-out one.",
            "Those out-of-fold predictions become the meta-model's features.",
            "The meta-model learns the combination weights from those honest predictions.",
            "Base models are then refitted on all training rows for actual scoring; validation and test never enter any of this.",
        ),
        use=(
            "When you have several genuinely different strong models and simple averaging leaves value on the table.",
            "In competition-style settings where a small consistent gain is worth the extra complexity and compute.",
        ),
        avoid=(
            "Do not stack when your dataset is small: folds get thin, the meta-features get noisy, and the meta-model overfits them.",
            "Do not stack when the added serving complexity outweighs a fractional metric gain.",
        ),
        myths=(
            (
                "You can train the meta-model on the base models' training predictions.",
                "Those predictions are contaminated by memorization, so the meta-model learns to trust whichever base model overfits hardest. Out-of-fold is not optional.",
            ),
            (
                "A complex meta-model extracts more.",
                "The meta-model has very few features and should stay simple. A deep meta-model is usually just another way to overfit the folds.",
            ),
        ),
        example=(
            "session.fit_stacking(",
            "    estimators={'logreg': LogisticRegression(max_iter=1000),",
            "                'gbdt': HistGradientBoostingClassifier(random_state=0)},",
            "    final_estimator=LogisticRegression(max_iter=1000),",
            "    cv=5,",
            ")",
            "session.evaluate_ensemble(partition='validation')",
        ),
        check=(
            "How many rows does each inner fold leave for the meta-features?",
            "Does stacking beat plain soft voting on validation?",
        ),
        tools=("fit_stacking", "evaluate_ensemble", "cv_score"),
        terms=("stacking", "out-of-fold", "cross-validation", "ensemble"),
        difficulty=ADVANCED,
    ),
    _layer(
        "ensemble-blending-holdout",
        plain=(
            "Blending is stacking's simpler cousin. Instead of cross-validation folds, it carves one slice "
            "out of the training rows, trains the base models on the rest, and fits the combiner on that "
            "single slice. Cheaper to run, noisier to trust."
        ),
        analogy=(
            "Judging your specialists on one afternoon's cases instead of a full rotation. Much quicker, and "
            "you had better hope that afternoon was representative."
        ),
        steps=(
            "BuildML carves an inner holdout out of the Session training partition: not from validation or test.",
            "Base models train on the remaining training rows.",
            "They predict the inner holdout, and those predictions train the combiner.",
            "The result is one blended plan you can evaluate on validation as usual.",
            "Compare it against stacking; if the gap is large, your inner holdout was probably too small.",
        ),
        use=(
            "When cross-validated stacking is too slow: many base models, large data, or expensive fits.",
            "As a quick check on whether combining is worth pursuing at all before investing in stacking.",
        ),
        avoid=(
            "Do not blend on small datasets; a single thin slice gives the combiner almost nothing to learn from.",
            "Do not carve the blending holdout from validation or test: the whole point is that it stays inside train.",
        ),
        myths=(
            (
                "Blending and stacking give the same answer more cheaply.",
                "Blending uses far less data for the combiner, so its weights are noisier and can differ meaningfully from stacked weights.",
            ),
            (
                "The blend holdout is a validation set.",
                "It is an inner slice of training rows. Your real validation partition stays untouched so it can still judge the finished ensemble.",
            ),
        ),
        example=(
            "session.fit_blending(",
            "    estimators={'logreg': LogisticRegression(max_iter=1000),",
            "                'gbdt': HistGradientBoostingClassifier(random_state=0)},",
            "    holdout_size=0.25, random_state=0,",
            ")",
            "session.evaluate_ensemble(partition='validation')",
        ),
        check=(
            "How many rows ended up in the inner blending holdout?",
            "Do the blend weights change a lot if you change the blending seed?",
        ),
        tools=("fit_blending", "fit_stacking", "evaluate_ensemble"),
        terms=("blending", "stacking", "holdout", "ensemble"),
        difficulty=ADVANCED,
    ),
    _layer(
        "ensemble-bundle-boundary",
        plain=(
            "A fitted ensemble is saved as its own artifact holding the ensemble plan and its fit result. "
            "It is not a Session checkpoint and not a single-model pipeline bundle: three different things "
            "with three different contracts."
        ),
        analogy=(
            "A recipe for a dish, a photograph of your kitchen, and a single ingredient's label. All useful, "
            "none a substitute for the others."
        ),
        steps=(
            "Fit a voting, stacking, or blending ensemble so an ensemble plan exists.",
            "Call `save_ensemble_bundle(path)` to persist the members, the combination rule, and the fit result.",
            "Reload with `load_ensemble_bundle(path)` on a Session whose features match.",
            "Evaluate or predict with the restored plan.",
            "Keep checkpoints separately if you also want the data workflow state back.",
        ),
        use=(
            "When the ensemble is the model you intend to ship or hand over.",
            "When you want the member list and combination rule preserved exactly rather than rebuilt from a script.",
        ),
        avoid=(
            "Do not expect a pipeline bundle to contain the ensemble; a pipeline stores preprocess plans plus one active estimator.",
            "Do not load an ensemble bundle from an untrusted source: it deserializes fitted estimators.",
        ),
        myths=(
            (
                "Saving the ensemble also saves my preprocessing.",
                "Preprocess plans belong to the pipeline surface. Save both if scoring needs both.",
            ),
            (
                "One artifact per project is simpler.",
                "One artifact per meaning is simpler to reason about. Mixing them is what causes 'why did my load succeed but predict wrong' bugs.",
            ),
        ),
        example=(
            "session.save_ensemble_bundle('artifacts/ensemble')",
            "restored = Session.ingest(frame).load_ensemble_bundle('artifacts/ensemble')",
            "restored.evaluate_ensemble(partition='test')",
        ),
        check=(
            "Which artifact holds your preprocessing, and which holds your ensemble?",
            "Can a colleague reproduce your scoring path from the files you committed?",
        ),
        tools=("save_ensemble_bundle", "load_ensemble_bundle", "save_pipeline", "checkpoint_save"),
        terms=("bundle", "ensemble", "checkpoint", "pipeline"),
        difficulty=CORE,
    ),
)

__all__ = ["ENSEMBLE_BEGINNER"]
