# ruff: noqa: E501
"""Native ensemble learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

ENSEMBLE_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="ensemble-voting-vs-single-tree",
            title="Voting ensembles vs a single bagged tree model",
            summary="Native voting combines heterogeneous base learners; a RandomForest passed to Session.fit is still one estimator.",
            definition=(
                "A voting ensemble aggregates predictions from multiple named base estimators "
                "(hard vote or soft probability average). Tree ensembles like RandomForest "
                "internally bag trees but expose a single sklearn estimator interface."
            ),
            intuition=(
                "Asking three different specialists and taking a majority is voting. "
                "Growing many trees inside one forest is still one specialist with a committee inside."
            ),
            formal_idea=(
                "ŷ = majority({f_i(x)}) or argmax_c Σ_i w_i P_i(y=c|x) for soft voting. "
                "RandomForest is one f with internal bootstrap aggregation."
            ),
            why_it_matters=(
                "Operators confuse 'use an ensemble model' with native multi-estimator voting/stacking.",
                "Teaching cards must disclose strategy, bases, and voting mode.",
            ),
            how_buildml_uses=(
                "Session.fit_voting builds VotingClassifier/VotingRegressor.",
                "Session.fit still accepts RandomForest as a single estimator.",
                "Catalog concepts keep the distinction explicit.",
            ),
            interpretation_rules=(
                "Read estimator_names and voting mode beside metrics.",
                "Soft voting needs predict_proba on every base.",
            ),
            assumptions=(
                "Bases are trained on the same train feature contract.",
                "Soft voting probabilities are calibrated enough to average meaningfully.",
            ),
            failure_modes=(
                "Calling fit_voting with one estimator.",
                "Soft voting with a base that lacks predict_proba.",
            ),
            anti_patterns=(
                "Treating RandomForest via Session.fit as the native ensemble product.",
            ),
            worked_example_pattern=(
                "split → prep → fit_voting({'lr': ..., 'rf': ...}) → evaluate_ensemble.",
            ),
            related_concepts=("ensemble-stacking-oof", "leakage-boundary", "baselines"),
        ),
        _note(
            key="ensemble-stacking-oof",
            title="Stacking uses out-of-fold meta features inside train",
            summary="Stacking CV folds stay inside the Session train partition; test never enters meta-learner fitting.",
            definition=(
                "Stacking fits base learners with cross-validated out-of-fold predictions "
                "to train a meta-learner, then typically refits bases on full train for "
                "inference. All of that happens on the training partition only."
            ),
            intuition=(
                "The meta-learner must see honest base mistakes. Out-of-fold predictions "
                "inside train provide that without peeking at the exam (test) set."
            ),
            formal_idea=(
                "For fold k, bases fit on train\\k predict on fold k. Meta features M_train "
                "train g. At deploy, bases refit on full train; ŷ = g(M(x))."
            ),
            why_it_matters=(
                "Using Session test to build meta features is label leakage.",
                "Stacking without OOF often overfits the meta-learner.",
            ),
            how_buildml_uses=(
                "Session.fit_stacking wraps sklearn StackingClassifier/Regressor.",
                "fit_estimator only materializes train rows; Session test stays out.",
                "Disclosures record cv and passthrough.",
            ),
            interpretation_rules=(
                "Report stacking cv beside test metrics.",
                "Passthrough concatenates raw features with meta features: watch dimensionality.",
            ),
            assumptions=(
                "Train is large enough for the chosen cv.",
                "Bases and meta-learner match the task type.",
            ),
            failure_modes=(
                "cv=1 or fitting meta features on the evaluation partition.",
                "Tiny train with aggressive stacking cv.",
            ),
            anti_patterns=(
                "Scoring stacking meta-features on Session test during fit.",
            ),
            worked_example_pattern=(
                "split → prep → fit_stacking(..., cv=5) → evaluate_ensemble(partition='test').",
            ),
            related_concepts=(
                "ensemble-blending-holdout",
                "cross-validation",
                "leakage-boundary",
            ),
        ),
        _note(
            key="ensemble-blending-holdout",
            title="Blending holdout is carved from train",
            summary="Holdout blending fits the meta-learner on an inner train holdout: never Session validation/test.",
            definition=(
                "Blending reserves a fraction of the training partition to generate base "
                "predictions for the meta-learner. Session validation/test partitions remain "
                "untouched during blend fit. Bases may then be refit on full train for deploy."
            ),
            intuition=(
                "Keep a practice quiz inside the study guide for the coach (meta-learner). "
                "Do not use the final exam pages to train the coach."
            ),
            formal_idea=(
                "Split train → (base_fit, blend_holdout). Fit f_i on base_fit; "
                "fit g on {f_i(x) : x ∈ blend_holdout}. Optionally refit f_i on full train."
            ),
            why_it_matters=(
                "Using Session test as the blend holdout leaks evaluation labels into g.",
                "Blending is weaker than stacking OOF but honest when scoped to train.",
            ),
            how_buildml_uses=(
                "Session.fit_blending uses HoldoutBlendClassifier/Regressor.",
                "Disclosures record holdout_fraction, blend_method, and full-train refit.",
                "Prefer fit_stacking when CV OOF meta features are desired.",
            ),
            interpretation_rules=(
                "Read holdout_fraction beside n_train_rows.",
                "Small blend holdouts make unstable meta-learners.",
            ),
            assumptions=(
                "holdout_fraction ∈ [0.05, 0.5).",
                "Train is large enough after the carve.",
            ),
            failure_modes=(
                "Carving blend holdout from concatenated train+test.",
                "Interpreting blend holdout metrics as Session test metrics.",
            ),
            anti_patterns=(
                "Calling Session validation the 'blend set' while also reporting it as final eval.",
            ),
            worked_example_pattern=(
                "split → prep → fit_blending(..., holdout_fraction=0.2) → evaluate_ensemble.",
            ),
            related_concepts=(
                "ensemble-stacking-oof",
                "evaluation-partitions",
                "leakage-boundary",
            ),
        ),
        _note(
            key="ensemble-bundle-boundary",
            title="Ensemble bundle boundary",
            summary="Ensemble bundles store EnsemblePlan + FitResult; they are not Session checkpoints or pipeline bundles.",
            definition=(
                "buildml.ensemble_bundle.v1 persists strategy disclosures and the fitted "
                "sklearn-compatible ensemble with its feature contract. Session checkpoints "
                "carry data/splits/history; pipeline bundles add preprocess plans + model card."
            ),
            intuition=(
                "Save the committee's playbook separately from the kitchen (preprocess) and "
                "the lab notebook (checkpoint)."
            ),
            formal_idea=(
                "EnsembleBundle ⊇ {EnsemblePlan, FitResult}; "
                "Checkpoint ⊉ ensemble weights; PipelineBundle may wrap the same estimator "
                "with plans."
            ),
            why_it_matters=(
                "Operators expecting preprocess restore from an ensemble bundle will miss plans.",
                "Complementary artifacts prevent silent deploy gaps.",
            ),
            how_buildml_uses=(
                "save_ensemble_bundle / load_ensemble_bundle.",
                "save_pipeline still works after fit_* because fit_result is set.",
                "CHECKPOINT_BOUNDARY prose is embedded in meta.json.",
            ),
            interpretation_rules=(
                "Reload tabular workflow via checkpoint_load; reload ensemble via load_ensemble_bundle.",
                "Use save_pipeline when impute/encode/scale must travel.",
            ),
            assumptions=(
                "Feature columns at load match the saved contract.",
            ),
            failure_modes=(
                "Loading an ensemble bundle into a frame with different feature columns.",
            ),
            anti_patterns=(
                "Treating ensemble_bundle.v1 as a Session checkpoint.",
            ),
            worked_example_pattern=(
                "fit_stacking → save_ensemble_bundle → new Session → load_ensemble_bundle → evaluate.",
            ),
            related_concepts=("unsupervised-bundle-boundary", "leakage-boundary"),
        ),
    )
}
