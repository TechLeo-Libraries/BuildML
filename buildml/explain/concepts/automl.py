# ruff: noqa: E501
"""AutoML concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

AUTOML_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="automl-beyond-hpo",
            title="AutoML vs single-estimator hyperparameter search",
            summary="AutoML jointly searches model families and preprocess strategies; grid/optuna_search tune one fixed estimator.",
            definition=(
                "Single-estimator HPO (grid_search / randomized_search / optuna_search / "
                "evolutionary_search) "
                "sweeps parameters (and optional recipe knobs) for one chosen model. "
                "BuildML AutoML additionally ranks across a finite catalog of estimator "
                "families and discrete fold-local preprocess strategies under a trial budget."
            ),
            intuition=(
                "Tuning the dials on one radio is HPO. Trying several radios and several "
                "antenna setups under a time budget is AutoML — still not inventing new radios (NAS)."
            ),
            formal_idea=(
                "Select (f, r, θ) ∈ F × R × Θ_f maximizing selection score S on train-only "
                "evidence (CV / nested / validation), then refit on full train."
            ),
            why_it_matters=(
                "Operators confuse 'I ran Optuna' with 'I ran AutoML'.",
                "Teaching cards must disclose catalogs, budget, and non-NAS scope.",
            ),
            how_buildml_uses=(
                "Session.run_automl searches families + recipe strategies.",
                "Session.grid_search / optuna_search remain the single-estimator path.",
                "Disclosures state finite catalogs and no causal claims.",
            ),
            interpretation_rules=(
                "Read best_family and best_recipe_strategy beside the score.",
                "Treat trial budgets as hard exploration caps.",
            ),
            assumptions=(
                "Catalogs cover common tabular sklearn families only.",
                "Predictive ranking — not causal identification.",
            ),
            failure_modes=(
                "Claiming NAS or fully automated science from this API.",
                "Ignoring limitations on catalog coverage.",
            ),
            anti_patterns=(
                "Running Session-global impute/scale then AutoML without the allow flag.",
            ),
            worked_example_pattern=(
                "split (unpoisoned) → run_automl(method='randomized', selection='cv') "
                "→ evaluate_automl(partition='test').",
            ),
            related_concepts=(
                "automl-recipe-strategy-search",
                "automl-selection-honesty",
                "leakage-boundary",
            ),
        ),
        _note(
            key="automl-recipe-strategy-search",
            title="Fold-local recipe strategy search",
            summary="AutoML can search discrete impute/scale/encode/select combinations as PreprocessRecipe strategies, refit per fold.",
            definition=(
                "A recipe strategy is a named PreprocessRecipe configuration (which steps "
                "and strategy enums). Unlike SAFE_RECIPE_KNOBS on a fixed recipe, strategy "
                "search chooses among discrete preprocess pipelines, each refit on fold-train "
                "only during selection."
            ),
            intuition=(
                "Instead of only asking 'how many features to keep?', also ask "
                "'should we scale? one-hot or ordinal?' — but always fit those choices "
                "inside the training fold."
            ),
            formal_idea=(
                "For fold k, fit recipe r on train\\k, transform train\\k and fold k, "
                "fit f_θ, score fold k. Never use Session test rows."
            ),
            why_it_matters=(
                "Session-global prep before CV/search leaks fold-eval information.",
                "Recipe knobs alone cannot change impute/scale/encode strategy enums.",
            ),
            how_buildml_uses=(
                "run_automl(include_recipe_search=True) enumerates DEFAULT_RECIPE_STRATEGIES.",
                "Same LeakageError refusal as cv_score when Session-global plans exist.",
            ),
            interpretation_rules=(
                "Inspect best_recipe_strategy and the recipe dict on AutoMLPlan.",
                "Passthrough strategies assume a clean design matrix.",
            ),
            assumptions=("Strategies stay inside PreprocessRecipe capabilities.",),
            failure_modes=(
                "Expecting resample or arbitrary custom transforms inside recipe search.",
            ),
            anti_patterns=(
                "Fitting Session.impute/encode/scale on full train, then AutoML.",
            ),
            worked_example_pattern=(
                "ingest → roles → split → run_automl(include_recipe_search=True) "
                "without Session-global prep.",
            ),
            related_concepts=(
                "automl-beyond-hpo",
                "leakage-boundary",
                "automl-selection-honesty",
            ),
        ),
        _note(
            key="automl-selection-honesty",
            title="AutoML selection modes and held-out honesty",
            summary="cv / nested / validation rank candidates without touching Session test; confirm on test after freezing.",
            definition=(
                "selection='cv' ranks by train-fold CV; 'nested' adds an outer train "
                "estimate after inner selection; 'validation' ranks on the Session "
                "validation partition. Session test stays untouched during search."
            ),
            intuition=(
                "Practice exams (CV/validation) pick the student; the final exam (test) "
                "is taken once after the choice is frozen."
            ),
            formal_idea=(
                "θ* = argmax_θ S_select(θ); report S_outer or S_test(θ*) only after selection."
            ),
            why_it_matters=(
                "Using test for AutoML ranking is selection leakage.",
                "Train-CV ranks alone are optimistic relative to nested/outer estimates.",
            ),
            how_buildml_uses=(
                "Session.run_automl(selection=...).",
                "evaluate_automl(partition='test') for post-selection confirmation.",
                "Nested outer_score_mean/std recorded on AutoMLPlan when selection='nested'.",
            ),
            interpretation_rules=(
                "Prefer nested or validation+test confirm for strong claims.",
                "Read limitations and disclosures on AutoMLResult.",
            ),
            assumptions=("SplitPlan train/validation/test partitions do not overlap.",),
            failure_modes=(
                "selection='validation' without a validation partition.",
                "Re-running AutoML after peeking at test metrics.",
            ),
            anti_patterns=(
                "Reporting the best trial's train-CV mean as final test performance.",
            ),
            worked_example_pattern=(
                "run_automl(selection='nested') → evaluate_automl(partition='test').",
            ),
            related_concepts=(
                "automl-beyond-hpo",
                "cross-validation",
                "evaluation-partitions",
            ),
        ),
        _note(
            key="automl-bundle-boundary",
            title="AutoML bundles vs checkpoints and pipelines",
            summary="buildml.automl_bundle.v1 stores AutoMLPlan disclosures; checkpoints do not embed them.",
            definition=(
                "An AutoML bundle persists the selected AutoMLPlan (family/recipe "
                "disclosures + fitted estimator) and FitResult contract. Session "
                "checkpoints store data/roles/splits/history; classical pipelines store "
                "Session-global plans + estimator."
            ),
            intuition=(
                "The search report card and the resume snapshot are different artifacts."
            ),
            formal_idea=(
                "automl_bundle = (AutoMLPlan, FitResult?); checkpoint ⊄ automl_bundle."
            ),
            why_it_matters=(
                "Operators expect checkpoint_load to restore AutoMLPlan — it does not.",
            ),
            how_buildml_uses=(
                "save_automl_bundle / load_automl_bundle.",
                "Prefer save_pipeline when Session-global preprocess plans must travel.",
            ),
            interpretation_rules=(
                "Check meta.json format=buildml.automl_bundle.v1.",
            ),
            assumptions=("Feature columns match on load.",),
            failure_modes=("Incomplete bundle missing automl_plan.joblib.",),
            anti_patterns=("Treating AutoML bundles as Session checkpoints.",),
            worked_example_pattern=(
                "run_automl → save_automl_bundle → load_automl_bundle → evaluate_automl.",
            ),
            related_concepts=("automl-beyond-hpo", "ensemble-bundle-boundary"),
        ),
    )
}
