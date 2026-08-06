"""Extra ML-engineering lessons beyond the core readiness spine (stages 00-05)."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._factory import L, with_starter
from buildml.dashboard.academy_curriculum._helpers import (
    first_feature,
    fmt_n,
    is_classification,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec


def lessons() -> list[LessonSpec]:
    """Gap-fill concepts for a serious EDA -> modeling readiness path."""
    return [
        L(
            slug="batch-leakage",
            stage=3,
            order=55,
            concept_key="batch-leakage",
            tags=("batch", "leakage"),
            search_terms=("batch leakage", "group leakage", "same-day"),
            plain=(
                "Batch leakage happens when rows that share a production batch, day, or file "
                "are split randomly - the model memorizes batch quirks.",
            ),
            technical=(
                "Treat batch ids like groups: inject_split by batch. "
                "Related to group-structure but specifically about processing cohorts.",
            ),
            why=("Same-batch twins in train and test inflate scores."),
            formula=None,
            calculation=lambda ctx: (
                f"Review id-like columns for batch keys: "
                f"{', '.join(ctx.get('idLike') or []) or '<batch_id>'}."
            ),
            session_evidence=lambda ctx: (
                f"Id-like columns to audit for batch keys: {ctx.get('idLike') or []}."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                'session.learn("batch-leakage", level="beginner")',
                "# If you have a batch column, split with inject_split by batch "
                "# (see group-structure example).",
            ),
            what_to_change=("Identify batch/day/file keys; forbid cross-split sharing."),
            pitfalls=("Random splits on heavily batched extracts."),
            decide="If rows share processing batches, validate by batch.",
            read_steps=("Find batch keys in lineage.", "Measure score drop under batch holdout."),
        ),
        L(
            slug="evaluation-partitions",
            stage=3,
            order=35,
            concept_key="evaluation-partitions",
            tags=("train", "validation", "test"),
            search_terms=("validation", "holdout", "partitions"),
            plain=(
                "Train fits, validation chooses, test assesses once. Mixing those jobs "
                "is how honest metrics disappear.",
            ),
            technical=(
                "session.split(validation_size=...) creates three-way membership. "
                "Tune thresholds and hyperparameters on validation; touch test rarely.",
            ),
            why=("A twice-used test set is a validation set with better PR."),
            formula=None,
            calculation=lambda ctx: (
                f"With n={fmt_n(ctx.get('rows'))}, a 60/20/20 split yields about "
                f"{fmt_n(int(0.6*(ctx.get('rows') or 0)))}/"
                f"{fmt_n(int(0.2*(ctx.get('rows') or 0)))}/"
                f"{fmt_n(int(0.2*(ctx.get('rows') or 0)))} rows."
            ),
            session_evidence=lambda ctx: f"n={fmt_n(ctx.get('rows'))}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "# Prefer an explicit validation partition when tuning",
                "session = session.split(",
                "    test_size=0.2,",
                "    validation_size=0.2,",
                f"    stratify={str(is_classification(ctx))},",
                "    random_state=0,",
                ")",
                'session.learn("evaluation-partitions", level="beginner")',
            ),
            what_to_change=("Set sizes; keep test frozen until the final card."),
            pitfalls=("Early-stopping on test.", "Threshold tuning on test."),
            decide="Write which partition may be used for fit / select / report.",
            read_steps=("Confirm three-way membership when tuning.", "Log every test peek."),
        ),
        L(
            slug="early-stopping-partition",
            stage=4,
            order=65,
            concept_key="early-stopping-partition",
            tags=("early stopping",),
            search_terms=("early stopping", "patience"),
            plain=(
                "Early stopping needs a validation signal. Using the test set to stop training "
                "quietly turns test into tuning.",
            ),
            technical=(
                "Keep a validation partition for stopping/selection; report final metrics on test once.",
            ),
            why=("Stopped-on-test models overstate generalisation."),
            formula=None,
            calculation=lambda ctx: (
                f"Allocate validation from n={fmt_n(ctx.get('rows'))} before iterative fitting."
            ),
            session_evidence=lambda ctx: f"n={fmt_n(ctx.get('rows'))}; task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "session = session.split(test_size=0.2, validation_size=0.2, "
                f"stratify={str(is_classification(ctx))}, random_state=0)",
                'session.learn("early-stopping-partition", level="beginner")',
                "# Iterative models: monitor validation only for stopping.",
            ),
            what_to_change=("Always point early stopping at validation, never test."),
            pitfalls=("Reuse of test for patience checks."),
            decide="Name the partition allowed to stop training - it must not be test.",
            read_steps=("Check training configs for eval_set/partition.", "Audit history for test peeks."),
        ),
        L(
            slug="overfitting",
            stage=4,
            order=55,
            concept_key="overfitting",
            tags=("overfit", "generalisation"),
            search_terms=("overfitting", "generalization", "capacity"),
            plain=(
                "Overfitting is memorising the training quirks so holdout performance collapses. "
                "More features, deeper trees, and repeated test peeks all feed it.",
            ),
            technical=(
                "Watch train-validation gaps, learning curves, and nested CV. "
                "Regularise, simplify, or gather data - do not chase test.",
            ),
            why=("The demo metric that only works on train is a liability."),
            formula=None,
            calculation=lambda ctx: (
                f"rows/feature ~ {fmt_n(int((ctx.get('rows') or 1) / max(int(ctx.get('eligible') or 1), 1)))}; "
                "small ratios raise overfit risk for flexible models."
            ),
            session_evidence=lambda ctx: (
                f"eligible={fmt_n(ctx.get('eligible'))}; n={fmt_n(ctx.get('rows'))}."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
                "",
                "session = session.impute().encode()",
                "est = RandomForestClassifier(max_depth=4, random_state=0) if "
                f"{is_classification(ctx)} else RandomForestRegressor(max_depth=4, random_state=0)",
                "cv = session.cv_score(est, cv=5)",
                "print(cv)  # compare to a more flexible max_depth",
                'session.learn("overfitting", level="beginner")',
            ),
            what_to_change=("Constrain capacity; use nested CV when searching."),
            pitfalls=("Growing model complexity to chase a single holdout."),
            decide="Require train/validation gap checks before accepting a complex model.",
            read_steps=("Compare simple vs complex CV scores.", "Inspect learning curves."),
        ),
        L(
            slug="feature-selection",
            stage=2,
            order=85,
            concept_key="feature-selection",
            tags=("selection",),
            search_terms=("feature selection", "select_features"),
            plain=(
                "Selection keeps informative columns and drops noise - but the choice of which "
                "columns to keep must be made on training folds only.",
            ),
            technical=(
                "session.select_features(strategy='variance'|'univariate'|'model', ...) "
                "is train-fitted after split.",
            ),
            why=("Full-frame selection leaks test structure into the feature set."),
            formula=None,
            calculation=lambda ctx: (
                f"Eligible features now: {fmt_n(ctx.get('eligible'))}. Selection should cite a budget."
            ),
            session_evidence=lambda ctx: f"eligible={fmt_n(ctx.get('eligible'))}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "session = session.impute().encode()",
                "session = session.select_features(",
                "    strategy=\"univariate\",",
                "    score_func=\"mutual_info\",",
                "    k=20,  # <-- change budget",
                ")",
                'session.learn("feature-selection", level="beginner")',
            ),
            what_to_change=("Set k/threshold; prefer model-based selection inside nested CV when high stakes."),
            pitfalls=("Selecting on the full frame.", "Using target-aware selection then reporting optimistic test scores without nesting."),
            decide="Declare a feature budget and the selection strategy before fitting competitors.",
            read_steps=("List current eligible features.", "Re-check leakage boundary after selection."),
        ),
        L(
            slug="model-selection",
            stage=4,
            order=35,
            concept_key="model-selection",
            tags=("model selection",),
            search_terms=("model selection", "compare_models"),
            plain=(
                "Model selection compares candidates under one primary metric and one validation design. "
                "It is not 'try everything and keep the lucky test winner'.",
            ),
            technical=("session.compare_models / grid_search / nested_cv_score support honest comparison."),
            why=("Unprincipled search overfits the selection procedure itself."),
            formula=None,
            calculation=lambda ctx: (
                f"task={ctx.get('task')}; n={fmt_n(ctx.get('rows'))} - match search breadth to sample size."
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
                "",
                "session = session.impute().encode().scale()",
                "models = {",
                "    \"linear\": LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge(),",
                "    \"forest\": RandomForestClassifier(random_state=0) if "
                f"{is_classification(ctx)} else RandomForestRegressor(random_state=0),",
                "}",
                "comparison = session.compare_models(models, partition=\"test\")",
                "print(comparison)",
                'session.learn("model-selection", level="beginner")',
            ),
            what_to_change=("Fix primary metric; prefer validation/nested designs for selection."),
            pitfalls=("Selecting on test.", "Changing metric after seeing winners."),
            decide="Pre-register the candidate set and primary metric before fitting them.",
            read_steps=("List candidates.", "Record selection partition.", "Keep final test for the card."),
        ),
        L(
            slug="mi-vs-correlation",
            stage=2,
            order=45,
            concept_key="mutual-information",
            tags=("MI", "correlation", "comparison"),
            search_terms=("mutual information vs correlation", "nonlinear", "pearson"),
            plain=(
                "Correlation (especially Pearson) asks about straight-line co-movement. "
                "Mutual information asks about any dependence. They answer different questions.",
            ),
            technical=(
                "Use both: correlation for redundancy among numerics; MI for target screens "
                "that may be non-linear or categorical.",
            ),
            why=("Choosing only Pearson misses curved signal; choosing only MI misses signed linear redundancy."),
            formula="Pearson ∈ [-1,1]; MI ≥ 0 (no direction)",
            calculation=lambda ctx: (
                f"Corr pairs recorded: {fmt_n(len(ctx.get('corrPairs') or []))}; "
                f"MI rows: {fmt_n(len(ctx.get('mi') or []))}."
            ),
            session_evidence=lambda ctx: (
                f"Strongest corr pairs: {fmt_n(len(ctx.get('corrPairs') or []))}; "
                f"MI available: {bool(ctx.get('mi'))}."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "biv = report.to_dict().get(\"bivariate\", {})",
                "print(\"pearson pairs\", biv.get(\"top_abs_pearson_pairs\", [])[:5])",
                "print(\"mi\", biv.get(\"mutual_information_vs_target\", [])[:5])",
                'session.learn("mutual-information", level="intermediate")',
            ),
            what_to_change=("Do not discard features solely because Pearson is near zero."),
            pitfalls=("Treating MI ranks as signed effects.", "Using Pearson on category codes."),
            decide="When screening, read Pearson and MI as complementary, not competing, evidence.",
            read_steps=(
                "Find low-Pearson / high-MI candidates (non-linear).",
                "Find high-Pearson feature pairs (redundancy).",
            ),
        ),
        L(
            slug="train-serve-parity",
            stage=5,
            order=55,
            concept_key="operation-history",
            tags=("serving", "parity"),
            search_terms=("train serve skew", "parity", "pipeline"),
            plain=(
                "The model in notebooks is not the model in production unless the exact preprocessing "
                "recipe travels with it.",
            ),
            technical=(
                "session.save_pipeline / load_pipeline persist fitted plans. "
                "Hygiene and joins upstream must match too.",
            ),
            why=("Train-serve skew is a top cause of 'it worked in the notebook' failures."),
            formula=None,
            calculation=lambda ctx: (
                f"Handoff must cover roles, split policy, and transforms for {fmt_n(ctx.get('eligible'))} features."
            ),
            session_evidence=lambda ctx: f"eligible={fmt_n(ctx.get('eligible'))}; target={target_name(ctx)}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "",
                "session = session.impute().encode().scale()",
                "session = session.fit(",
                "    LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge()",
                ")",
                "session.save_pipeline(\"artifacts/serve_pipeline\")  # <-- change",
                "# Later / other process:",
                "# session2 = Session.ingest(live_frame).load_pipeline(\"artifacts/serve_pipeline\", trusted=True)",
                "# preds = session2.predict_from_pipeline(partition=\"test\")",
            ),
            what_to_change=("Point artifacts at your registry; pin versions; test parity on a golden batch."),
            pitfalls=("Re-implementing preprocessing by hand in the serve path."),
            decide="Require a golden-batch parity test between training recipe and serve path.",
            read_steps=("Save pipeline.", "Score a frozen batch in both environments.", "Diff predictions."),
        ),
    ]
