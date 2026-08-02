# ruff: noqa: E501
"""Shared concept notes referenced by the operation catalog and Concept Academy."""

from __future__ import annotations

from buildml.explain.schemas import ConceptNote


def _flatten_details(
    *,
    definition: str,
    intuition: str,
    formal_idea: str,
    why_it_matters: tuple[str, ...],
    how_buildml_uses: tuple[str, ...],
    interpretation_rules: tuple[str, ...],
    assumptions: tuple[str, ...],
    failure_modes: tuple[str, ...],
    anti_patterns: tuple[str, ...],
    worked_example_pattern: tuple[str, ...],
) -> tuple[str, ...]:
    """Build a searchable flat paragraph list from structured teaching sections."""
    parts: list[str] = []
    for paragraph in (definition, intuition, formal_idea):
        text = paragraph.strip()
        if text:
            parts.append(text)
    for group in (
        why_it_matters,
        how_buildml_uses,
        interpretation_rules,
        assumptions,
        failure_modes,
        anti_patterns,
        worked_example_pattern,
    ):
        for item in group:
            text = item.strip()
            if text:
                parts.append(text)
    return tuple(parts)


def _note(
    *,
    key: str,
    title: str,
    summary: str,
    definition: str,
    intuition: str,
    formal_idea: str,
    why_it_matters: tuple[str, ...],
    how_buildml_uses: tuple[str, ...],
    interpretation_rules: tuple[str, ...],
    assumptions: tuple[str, ...],
    failure_modes: tuple[str, ...],
    anti_patterns: tuple[str, ...],
    worked_example_pattern: tuple[str, ...],
    related_concepts: tuple[str, ...] = (),
    references: tuple[str, ...] = (),
    details: tuple[str, ...] | None = None,
) -> ConceptNote:
    """Construct a ConceptNote and auto-build searchable ``details`` when omitted."""
    flat = details if details is not None else _flatten_details(
        definition=definition,
        intuition=intuition,
        formal_idea=formal_idea,
        why_it_matters=why_it_matters,
        how_buildml_uses=how_buildml_uses,
        interpretation_rules=interpretation_rules,
        assumptions=assumptions,
        failure_modes=failure_modes,
        anti_patterns=anti_patterns,
        worked_example_pattern=worked_example_pattern,
    )
    return ConceptNote(
        key=key,
        title=title,
        summary=summary,
        details=flat,
        related_concepts=related_concepts,
        references=references,
        definition=definition,
        intuition=intuition,
        formal_idea=formal_idea,
        why_it_matters=why_it_matters,
        how_buildml_uses=how_buildml_uses,
        interpretation_rules=interpretation_rules,
        assumptions=assumptions,
        failure_modes=failure_modes,
        anti_patterns=anti_patterns,
        worked_example_pattern=worked_example_pattern,
    )


CONCEPT_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="column-roles",
            title="Column roles",
            summary="Roles label how each column participates in modeling, not what its dtype happens to be.",
            definition=(
                "A column role is an explicit workflow label—feature, target, identifier, group, time, "
                "weight, or ignore—that tells BuildML which columns may train, which must be predicted, "
                "and which must stay out of the estimator matrix."
            ),
            intuition=(
                "Think of roles as sticky notes on spreadsheet headers. An integer customer id looks "
                "numeric, but treating it as a feature teaches the model account numbers instead of "
                "behavior. Roles separate meaning from storage type."
            ),
            formal_idea=(
                "Let the table columns be C. A role map r: C → {feature, target, id, group, time, "
                "weight, ignore} defines the modeling contract: predictors X come from feature columns, "
                "labels y from target columns, and id/group/time/weight/ignore columns follow their own "
                "rules for splitting, sampling, and scoring."
            ),
            why_it_matters=(
                "Wrong roles silently leak identifiers or drop the true predictors from the fit matrix.",
                "Downstream defaults (impute, encode, scale, EDA plots) usually key off feature and target roles.",
                "Group and time roles change which split designs are valid when rows are not exchangeable.",
                "Ignoring a column is safer than deleting it when you still need it for audit or joins.",
            ),
            how_buildml_uses=(
                "Session.set_roles(...) stores roles on the dataset; Session.ingest may infer provisional roles.",
                "Transform and model helpers select feature columns by role rather than by dtype alone.",
                "EDA Teaching Studio and explain() surfaces warn when target or feature roles look incomplete.",
                "Checkpoint bundles persist roles so reattach restores the same modeling contract.",
            ),
            interpretation_rules=(
                "Treat inferred roles as a draft: confirm every target, id, and ignore assignment before split.",
                "A high-cardinality integer labeled feature is a review flag for accidental id use.",
                "Multiple target roles imply a multitarget or mislabeled setup—resolve before fitting.",
                "Weight roles affect training emphasis; verify they are not post-outcome quantities.",
            ),
            assumptions=(
                "Role labels match the intended prediction task and data-collection timing.",
                "Feature roles exclude values unavailable at score time in the deployment setting.",
                "Group/time roles, when set, reflect true dependence structure among rows.",
            ),
            failure_modes=(
                "Identifier or future-looking columns left as features create optimistic metrics.",
                "Missing target role blocks supervised fit and confuses diagnostic labeling.",
                "Dtype-based guesses promote postal codes or enums into continuous features.",
            ),
            anti_patterns=(
                "Choosing features solely because columns are numeric.",
                "Dropping ids instead of assigning an id/ignore role when you still need them later.",
                "Changing roles after comparing models without re-running the full train-fit path.",
            ),
            worked_example_pattern=(
                "List columns and ask: known at prediction time? label? row identity? grouping key?",
                "Assign feature/target/id/ignore, then verify the intended X/y columns before split.",
                "Re-open roles after EDA if a column looks like an id, leakage carrier, or constant.",
            ),
            related_concepts=("leakage-boundary", "feature-schema", "data-splitting"),
        ),
        _note(
            key="leakage-boundary",
            title="Train-only learning",
            summary="Any statistic, vocabulary, or model parameter must be learned from training rows only.",
            definition=(
                "The leakage boundary is the rule that evaluation rows must not influence parameters used "
                "to transform or score them—imputers, encoders, scalers, selectors, samplers, and estimators "
                "all learn on train (or nested train folds) and apply a frozen plan elsewhere."
            ),
            intuition=(
                "Imagine studying for a quiz using the answer key of the quiz itself. Filling missing ages "
                "with the median of train+test, or encoding categories seen only in test, gives the model "
                "peeking rights that disappear in production."
            ),
            formal_idea=(
                "Partition rows into train / validation / test. For any learner L that maps data to "
                "parameters θ, compute θ = L(train) and apply the fixed map f_θ to validation and test. "
                "Using L(train ∪ eval) or selecting models on test moves information across the boundary."
            ),
            why_it_matters=(
                "Leakage inflates holdout scores and produces models that fail on truly unseen traffic.",
                "Preprocessing leakage is as damaging as label leakage and harder to notice in notebooks.",
                "Fair model comparison requires the same freeze point for every candidate pipeline.",
            ),
            how_buildml_uses=(
                "Session.assert_can_fit and split-aware transforms expect a split before train-fitted steps.",
                "Impute, encode, scale, and resample learn from the train partition and apply outward.",
                "Explain catalog notes and Concept Academy call out leakage risks per operation.",
                "Checkpoints store split membership so resumed work does not silently rejoin partitions.",
            ),
            interpretation_rules=(
                "If a transform was fit on all rows, treat subsequent validation/test scores as contaminated.",
                "Unknown categories or new missingness at score time are expected; they are not a license to refit on test.",
                "Nested or repeated validation choices that touch test make test a tuning set.",
            ),
            assumptions=(
                "Partition membership is defined before fitting reusable transformers.",
                "Score-time inputs follow the same availability constraints as training features.",
                "No post-outcome fields remain in the feature matrix.",
            ),
            failure_modes=(
                "Global median/mode/scale computed before splitting.",
                "Feature selection or PCA fit on the full table, then evaluated on a subset.",
                "Threshold or hyperparameter search finalized on the test partition.",
            ),
            anti_patterns=(
                "Cleaning the whole CSV, then calling train_test_split as a last step.",
                "Refitting encoders after seeing test errors to 'fix' unknown labels.",
                "Using test metrics to pick which leakage-prone columns to keep.",
            ),
            worked_example_pattern=(
                "Split first; fit imputer/encoder/scaler on train only; transform validation and test with frozen params.",
                "Compare a deliberately leaky full-data impute versus train-only impute on the same holdout.",
                "Document any statistic that required looking at non-train rows and discard it from the pipeline.",
            ),
            related_concepts=("data-splitting", "evaluation-partitions", "encoding-imputation-scaling"),
        ),
        _note(
            key="data-splitting",
            title="Partition design",
            summary="A split assigns each row a membership that controls what may train, guide, and assess the model.",
            definition=(
                "Data splitting partitions rows into disjoint sets—commonly train, validation, and test—so "
                "learning, iterative choices, and final estimation use separate observations under a stated "
                "sampling design (random, stratified, grouped, or temporal)."
            ),
            intuition=(
                "Splitting is not shuffling for luck; it is deciding which rows are allowed to teach the "
                "recipe and which rows only taste it. Related rows (same customer, same day) that land in "
                "both sides make the taste test too easy."
            ),
            formal_idea=(
                "A split plan is a function s: rows → {train, validation, test, ...} with disjoint images. "
                "Random splits assume exchangeability; stratified splits preserve class proportions in "
                "expectation; grouped/temporal splits keep dependence units intact across partitions."
            ),
            why_it_matters=(
                "Partition design determines whether holdout metrics estimate deployment risk or notebook luck.",
                "Unstable splits make model rankings jitter across seeds without any real improvement.",
                "Grouped or time-ordered data needs non-random splits or metrics become misleadingly high.",
            ),
            how_buildml_uses=(
                "Session.split(...) records membership on the dataset for later train-fitted operations.",
                "Session.group_split(...) and Session.time_split(...) provide first-class entity and clock partitions.",
                "Stratify options stabilize class balance when a target role is set and classes allow it.",
                "Checkpoint bundles persist split plans so mid-loop resume keeps the same partitions.",
                "EDA Teaching Studio and diagnostics should be read with the active partition in mind.",
            ),
            interpretation_rules=(
                "Random-split metrics assume rows are exchangeable; reject that reading for panels or time series.",
                "Stratification fixes marginal class rates, not duplicate-row or entity leakage.",
                "Keep the same membership when comparing preprocessors and estimators.",
                "Tiny validation sets (e.g. dozens of positives) make ranking noisy—treat gaps cautiously.",
            ),
            assumptions=(
                "The sampling design matches how new data will arrive (i.i.d., by group, or over time).",
                "Labels and features in holdouts follow the same measurement process as training.",
                "No row copies of the same entity straddle train and evaluation unintentionally.",
            ),
            failure_modes=(
                "Customer or event ids split across train and test inflate scores.",
                "Future rows placed in train and past rows in test reverse time.",
                "Re-splitting between experiments invalidates prior comparisons.",
            ),
            anti_patterns=(
                "Picking the seed that yields the nicest test score.",
                "Using a random split on strongly time-ordered logs without a time holdout.",
                "Stratifying on a column that is not the modeling target and calling it 'balanced'.",
            ),
            worked_example_pattern=(
                "State the deployment unit (row, customer, day), then choose random, stratified, grouped, or temporal split.",
                "Fix membership, fit only on train, score validation; reserve test until the recipe is frozen.",
                "Sanity-check overlap of group keys and label rates across partitions before modeling.",
            ),
            related_concepts=("leakage-boundary", "evaluation-partitions", "dataset-drift", "cross-validation"),
        ),
        _note(
            key="evaluation-partitions",
            title="Validation and test use",
            summary="Validation supports iterative choices; test estimates performance after those choices are locked.",
            definition=(
                "Evaluation partitions are holdout sets with different jobs: validation (or cross-validation "
                "folds) guide thresholds, features, and model choices; a final test set estimates risk after "
                "those choices stop changing."
            ),
            intuition=(
                "Validation is the practice exam you may retake after changing study methods. Test is the "
                "one sitting you report. If you keep rewriting the exam based on the official score, you no "
                "longer have an independent grade."
            ),
            formal_idea=(
                "Let C be a choice set (models, thresholds, features). Selection uses scores on validation "
                "partitions; the reported generalization estimate uses a test partition that did not enter "
                "argmax_C. Reusing test for selection biases the reported score upward in expectation."
            ),
            why_it_matters=(
                "Partition misuse is a common source of overconfident launch decisions.",
                "Train-versus-holdout gaps diagnose variance, shift, and overfitting better than a single number.",
                "Stakeholders need the partition name beside every metric to interpret it.",
            ),
            how_buildml_uses=(
                "Session metrics and diagnostic helpers accept or imply a partition name in results.",
                "Model comparison workflows expect the same holdout definition across candidates.",
                "Explain after-operation notes remind you which partition a score came from.",
                "Checkpoints preserve membership so validation and test do not quietly remix.",
            ),
            interpretation_rules=(
                "Always read a score with its partition tag (train / validation / test).",
                "A large train–validation gap is a review flag for overfitting, leakage, or split mismatch.",
                "If test was used to pick the winner, treat the test score as optimistic.",
                "Small-n holdouts: wide uncertainty—prefer ranges or repeats over single-point bravado.",
            ),
            assumptions=(
                "Validation and test are disjoint from each other and from training rows used to fit.",
                "The metric matches the decision costs you care about.",
                "Holdout collection process matches the intended deployment population.",
            ),
            failure_modes=(
                "Tuning threshold, features, and model family on the test set.",
                "Reporting train accuracy as if it were generalization.",
                "Comparing models evaluated on different membership definitions.",
            ),
            anti_patterns=(
                "Calling the only holdout 'test' while repeatedly selecting on it.",
                "Peeking at test early 'just to check', then continuing to iterate.",
                "Averaging train and test scores into one vanity number.",
            ),
            worked_example_pattern=(
                "Freeze split; iterate model and threshold choices on validation only.",
                "Once frozen, score test once and report partition, metric, and sample size.",
                "If test disappoints, diagnose with train/validation curves—do not mine test for fixes.",
            ),
            related_concepts=("data-splitting", "model-selection", "overfitting"),
        ),
        _note(
            key="feature-schema",
            title="Feature schema stability",
            summary="Training and scoring must share the same feature names, meanings, and encodings.",
            definition=(
                "A feature schema is the contract of column names, dtypes/roles, and post-transform layout "
                "that an estimator expects. Stability means score-time inputs can be mapped into that same "
                "layout without silent renames, drops, or encoding drift."
            ),
            intuition=(
                "The model learned to read a form with fixed blanks. If production sends renamed fields, "
                "reordered one-hots, or a new category policy, you are grading answers written on a different form."
            ),
            formal_idea=(
                "An estimator implements a map f: X → ŷ where X lives in a coordinate system defined by "
                "feature names (and encoding columns). Any transform that changes that coordinate system "
                "between fit and score breaks f unless the same fitted pipeline rebuilds X identically."
            ),
            why_it_matters=(
                "Schema drift causes hard failures or, worse, quiet misaligned columns.",
                "Fair experiments require identical feature contracts across candidates.",
                "Production monitoring needs a baseline schema to detect broken joins and renames.",
            ),
            how_buildml_uses=(
                "Roles plus train-fitted encode/impute/scale define the practical feature contract.",
                "Checkpoint reattach validates data-state compatibility; renamed semantics still need human review.",
                "Model bundles should travel with the fitted preprocessing that created their columns.",
                "EDA Teaching Studio profiles columns by name so schema surprises surface early.",
            ),
            interpretation_rules=(
                "After encoding, count and name of model-input columns should match the fit-time layout.",
                "A successful load does not prove two columns with the same name mean the same thing.",
                "New categories or missing columns at score time need an explicit policy, not ad-hoc fixes.",
            ),
            assumptions=(
                "Column names used at fit still identify the same measurements at score time.",
                "Fitted preprocessing is applied before the estimator in the same order.",
                "Dropped or ignored columns remain excluded consistently.",
            ),
            failure_modes=(
                "One-hot column sets differ between train and score because categories changed.",
                "Manual column drops after fit without updating the persisted pipeline.",
                "Joining an extra field that shifts positions in a nameless array pipeline.",
            ),
            anti_patterns=(
                "Fitting on a notebook DataFrame slice and scoring a differently cleaned export.",
                "Relying on column position instead of name through preprocessing.",
                "Treating checkpoint success as proof of semantic equivalence after renames.",
            ),
            worked_example_pattern=(
                "Record feature names after the full train-fitted transform on train.",
                "Apply the same frozen pipeline to validation and assert identical column sets.",
                "Introduce a deliberate rename and show the failure or misalignment before deploy.",
            ),
            related_concepts=("categorical-encoding", "checkpoint-integrity", "column-roles"),
        ),
        _note(
            key="missing-data",
            title="Missing-data treatment",
            summary="Imputation fills gaps with a train-learned rule; it does not prove missingness is harmless.",
            definition=(
                "Missing-data treatment decides how nulls and sentinel missings are handled—drop, impute, "
                "or model natively—using policies justified by the missingness mechanism and by what will "
                "be available at score time."
            ),
            intuition=(
                "A blank cell is information about the measurement process, not just an inconvenience. "
                "Filling age with the training median makes the matrix complete, but if 'age missing' "
                "marks rushed applications, you may want an indicator as well."
            ),
            formal_idea=(
                "For feature j, imputation learns a fill rule g_j from training observations where j is "
                "observed (e.g. median, mode) and applies g_j at transform time. This targets MCAR/MAR-style "
                "convenience under strong assumptions; MNAR structure can remain in residuals and indicators."
            ),
            why_it_matters=(
                "Many estimators reject nulls; the fill rule becomes part of the model.",
                "Train-only imputation prevents holdout means from leaking into features.",
                "Missingness patterns can shift between train and production and degrade quality.",
            ),
            how_buildml_uses=(
                "Session.impute(...) fits strategies on train feature columns and applies them outward.",
                "EDA Teaching Studio reports missing rates to prioritize which columns need a policy.",
                "Roles decide which columns are imputed as features versus left as id/ignore.",
                "Explain notes stress that imputation is a representation choice, not a causal fix.",
            ),
            interpretation_rules=(
                "Missing rate > ~20–30% in a key feature is a review flag for drop, indicator, or domain fix.",
                "Median fills suit skewed numerics; mean fills are more pullable by tails.",
                "Most-frequent category fill can erase rare-but-important levels—inspect value counts.",
                "If missingness correlates with the target, treat that as association to investigate, not proof of mechanism.",
            ),
            assumptions=(
                "The chosen mechanism assumption is plausible for each column.",
                "Score-time missingness resembles the patterns seen in training.",
                "Sentinel values (e.g. -999) are converted to true nulls before imputation.",
            ),
            failure_modes=(
                "Fitting impute statistics on all rows before splitting.",
                "Imputing identifiers or leakage-prone fields that should be ignored.",
                "Silent sentinel values treated as real magnitudes.",
            ),
            anti_patterns=(
                "Dropping every row with any null without checking which columns drive the loss.",
                "Imputing first, then discovering the column is mostly empty noise.",
                "Using target-aware fills that peek at labels for feature columns.",
            ),
            worked_example_pattern=(
                "Profile missing rates by column and by partition after split.",
                "Fit a train-only median/mode imputer; transform validation; compare distributions.",
                "Optionally add missing indicators and see whether validation metric moves.",
            ),
            related_concepts=("leakage-boundary", "encoding-imputation-scaling", "diagnostic-uncertainty"),
        ),
        _note(
            key="categorical-encoding",
            title="Categorical encoding",
            summary="Encoding maps category labels into numeric columns an estimator can consume.",
            definition=(
                "Categorical encoding is a deterministic mapping from discrete level labels to numeric "
                "features—commonly one-hot/dummy columns or ordinal integers—fit from the training "
                "vocabulary and applied with an explicit unknown-level policy at score time."
            ),
            intuition=(
                "Models that expect numbers cannot read the word 'red'. One-hot gives separate switches "
                "per color; ordinal numbering invents an order (red < blue) that only makes sense if that "
                "order is real in the domain."
            ),
            formal_idea=(
                "Given training level set V_j for column j, one-hot builds indicators 1[x=v] for v in V_j "
                "(often dropping one level for linear models). Ordinal encoding assigns a rank map "
                "π: V_j → ℤ. Unseen levels at score time require a declared handling (error, ignore, other)."
            ),
            why_it_matters=(
                "Encoding choice changes dimensionality, collinearity, and what distance-based models see.",
                "High-cardinality one-hots can dominate memory and dilute signal.",
                "Unknown levels in production are inevitable; policy must be fixed before deploy.",
            ),
            how_buildml_uses=(
                "Session.encode(...) learns vocabularies on train feature columns and transforms other partitions.",
                "Feature schema after encoding becomes the estimator input contract.",
                "EDA cardinality/entropy views help decide one-hot versus alternatives before encode.",
                "Leakage-boundary teaching stresses never refitting vocabularies on test to hide unknowns.",
            ),
            interpretation_rules=(
                "Cardinality in the hundreds+ is a review flag for one-hot width and rare-level noise.",
                "Ordinal encoding without a domain order is a review flag for false metric structure.",
                "A spike of 'unknown' at score time signals vocabulary shift or data-quality change.",
            ),
            assumptions=(
                "Level labels are consistently spelled and cased across partitions.",
                "Chosen encoding matches estimator family (trees tolerate ordinal codes better than linear models).",
                "Rare levels are stable enough to estimate or are deliberately pooled.",
            ),
            failure_modes=(
                "Fitting the encoder on all rows so rare test-only levels appear during training.",
                "One-hot exploding after free-text fields were mistaken for low-cardinality categories.",
                "Different unknown policies between offline eval and online serving.",
            ),
            anti_patterns=(
                "Label-encoding unordered cities and feeding them to a linear model as magnitudes.",
                "Target encoding without nested leakage control.",
                "Dropping unknowns at score time only when they hurt the reported metric.",
            ),
            worked_example_pattern=(
                "On train, list levels and cardinalities; choose one-hot or ordinal with a written reason.",
                "Fit encoder on train; transform validation; count unknown-level occurrences.",
                "Compare model family sensitivity: tree versus regularized linear on the same encoding.",
            ),
            related_concepts=("feature-schema", "leakage-boundary", "encoding-imputation-scaling"),
        ),
        _note(
            key="feature-scaling",
            title="Feature scaling",
            summary="Scaling rewrites numeric units; it adds no new information and must be train-fitted.",
            definition=(
                "Feature scaling applies an affine or robust transform to numeric columns—standardization, "
                "min–max, or quantile-style maps—so magnitudes become comparable for scale-sensitive "
                "learners, using parameters estimated on training rows only."
            ),
            intuition=(
                "Measuring height in millimeters and income in dollars makes 'distance' meaningless. "
                "Scaling puts columns on comparable footings. Trees that split on thresholds mostly shrug; "
                "penalized linear models and k-NN do not."
            ),
            formal_idea=(
                "Standard scaling uses train estimates μ̂_j, σ̂_j and maps x ↦ (x − μ̂_j)/σ̂_j. Min–max uses "
                "train min/max. Because parameters are estimated, fitting them on evaluation rows leaks "
                "distributional information across the leakage boundary."
            ),
            why_it_matters=(
                "Unscaled inputs can let one column dominate regularization or distance geometry.",
                "Scaling interacts with imputation order and outlier handling.",
                "Unnecessary scaling adds moving parts without helping tree-only pipelines.",
            ),
            how_buildml_uses=(
                "Session.scale(...) fits scalers on train numeric features and applies the frozen map outward.",
                "Session.apply_preprocess_plans(...) replays a restored ScalePlan at score time without refitting.",
                "PCA and many linear baselines assume scaled inputs; Teaching Studio calls this out.",
                "Engine choice does not change the statistical need to freeze scale parameters on train.",
                "Pipeline order is typically impute → encode → scale before model fit.",
                "Soft materialization gates warn near 250 MiB; hard gates refuse when configured via "
                "hard_limit_bytes or BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES.",
            ),
            interpretation_rules=(
                "After standard scaling, train means near 0 and variances near 1 are sanity checks—not proof of Gaussianity.",
                "Min–max to [0, 1] on heavy-tailed data can crush typical values—review outlier influence.",
                "If only tree models are compared, treat scaling as optional unless downstream steps need it.",
                "Treat soft materialization warnings as scale signals; configure a hard limit when copies must refuse.",
            ),
            assumptions=(
                "Numeric columns are true quantities, not ids mislabeled as features.",
                "Train range/variance estimates are stable enough for the chosen scaler.",
                "Score-time units match training units (no silent currency or scale changes).",
            ),
            failure_modes=(
                "Scaling before split using global mean and variance.",
                "Scaling one-hot columns unnecessarily and obscuring interpretability.",
                "Applying a scaler fit on a different feature set than the model expects.",
            ),
            anti_patterns=(
                "Scaling as a ritual before every model regardless of family.",
                "Winsorizing or clipping using test quantiles.",
                "Comparing scaled and unscaled models without keeping the rest of the pipeline fixed.",
            ),
            worked_example_pattern=(
                "Pick a scale-sensitive model and an unscaled baseline on the same split.",
                "Fit standard scaling on train only; transform validation; compare validation metrics.",
                "Inspect a wide-range column before/after to confirm units changed, not rank information.",
            ),
            related_concepts=("leakage-boundary", "principal-components", "encoding-imputation-scaling"),
        ),
        _note(
            key="class-imbalance",
            title="Class imbalance",
            summary="Unequal class frequencies change what accuracy and default thresholds can tell you.",
            definition=(
                "Class imbalance is a large disparity in label frequencies. It affects prior odds, metric "
                "interpretation, sampling strategies, and the mapping from predicted probabilities to decisions."
            ),
            intuition=(
                "If 95% of rows are negative, a model that always says negative scores 95% accuracy while "
                "finding zero positives. Imbalance does not forbid learning; it forbids lazy metrics and "
                "unexamined 0.5 thresholds."
            ),
            formal_idea=(
                "Let π = P(Y=1). When π is small, accuracy and ROC can look strong while precision at useful "
                "recall stays poor. Resampling changes the training distribution to π'; probability "
                "outputs and thresholds may need recalibration against the original π."
            ),
            why_it_matters=(
                "Metric choice must reflect false-positive versus false-negative costs under prevalence π.",
                "Resampling alters training balance but must not rewrite validation/test prevalence.",
                "Threshold and calibration work often matter more than exotic samplers.",
            ),
            how_buildml_uses=(
                "Session.resample(...) adjusts training rows after split; holdouts keep natural prevalence.",
                "Stratified splits help keep class rates stable across partitions when feasible.",
                "Diagnostics and Teaching Studio emphasize precision–recall thinking for rare positives.",
                "Class-weight and threshold tools are alternatives surfaced alongside sampling.",
            ),
            interpretation_rules=(
                "Prevalence under ~5–10% is a review flag to prefer PR curves over accuracy headlines.",
                "After oversampling train, judge models on unsamples validation/test.",
                "A probability of 0.5 is not 'balanced risk' when π is tiny—set thresholds from costs.",
            ),
            assumptions=(
                "Label definitions are stable; imbalance is not an artifact of labeling backlog.",
                "Deployment prevalence is close to the evaluation partition's prevalence.",
                "Positive class is correctly identified in the target role.",
            ),
            failure_modes=(
                "Resampling before splitting so duplicates leak across partitions.",
                "Reporting accuracy alone on a 99:1 problem.",
                "Synthetic minority oversampling with leakage-prone neighbors across entities.",
            ),
            anti_patterns=(
                "Oversampling the whole dataset, then splitting.",
                "Tuning for ROC AUC while the product needs precision at fixed recall.",
                "Forcing 50/50 training balance without recalibrating decisions to real prevalence.",
            ),
            worked_example_pattern=(
                "Compute class rates on train/validation/test after a stratified split.",
                "Fit a baseline and a candidate; compare PR and confusion counts on validation.",
                "Try class weights or threshold move before adding synthetic sampling complexity.",
            ),
            related_concepts=("thresholds", "evaluation-partitions", "probability-calibration", "baselines"),
        ),
        _note(
            key="model-selection",
            title="Model comparison",
            summary="A ranked score supports one criterion under one protocol—not a universal best model.",
            definition=(
                "Model selection compares candidate estimators (and often pipelines) under a fixed data "
                "partitioning protocol and metric, then chooses a candidate for deployment or further testing "
                "while recording uncertainty and non-score constraints."
            ),
            intuition=(
                "Leaderboards answer 'who won this race on this track with these rules?' They do not answer "
                "'who wins every future race.' A slightly lower score with stable errors and simpler ops can "
                "be the better ship."
            ),
            formal_idea=(
                "Given candidates {f_m} and a scoring rule S on a validation design D, selection returns "
                "m* ≈ argmax S(f_m, D). The reported test score of f_{m*} is optimistically biased if D "
                "included the test partition or if many m were mined without accounting for multiplicity."
            ),
            why_it_matters=(
                "Inconsistent splits or transforms make rankings non-comparable.",
                "Metric mismatch to business costs selects the wrong winner.",
                "Complexity, latency, and monitoring load can outweigh small S gains.",
            ),
            how_buildml_uses=(
                "Session.cv_score, grid_search, and randomized_search rank candidates on train-fold CV only.",
                "Session.compare_models ranks on a single holdout partition under shared transforms.",
                "Explain and diagnostics attach partition-tagged metrics for comparison.",
                "Baselines are first-class so 'better than nothing' is visible.",
            ),
            interpretation_rules=(
                "Require identical partitions and preprocessing before trusting a rank order.",
                "Prefer mean±std across folds for selection; treat gaps within fold std as fragile.",
                "Inspect failure slices and calibration, not only the headline score.",
                "If many models were tried, expect selection bias on the validation winner.",
            ),
            assumptions=(
                "The metric approximates decision value under deployment costs.",
                "Candidates were evaluated under the same leakage-safe protocol.",
                "Validation sample size supports distinguishing meaningful gaps.",
            ),
            failure_modes=(
                "Picking the model with best test score after unrestricted peeking.",
                "Changing feature recipes per model without documenting the apples-to-oranges gap.",
                "Ignoring variance: crowning a winner from one noisy fold.",
            ),
            anti_patterns=(
                "Automating selection solely on accuracy for imbalanced tasks.",
                "Equating 'more complex model' with 'better engineering'.",
                "Discarding the baseline because it is not fashionable.",
            ),
            worked_example_pattern=(
                "Fix split and preprocessing; define metric and constraints (latency, interpretability).",
                "Score baseline + candidates with CV or validation; note gaps versus fold uncertainty.",
                "Freeze the winner; score test once; record why non-score factors did or did not veto.",
            ),
            related_concepts=(
                "evaluation-partitions",
                "diagnostic-uncertainty",
                "baselines",
                "overfitting",
                "cross-validation",
            ),
        ),
        _note(
            key="cross-validation",
            title="Cross-validation",
            summary="Fold-wise refitting estimates score spread without consuming the final holdout.",
            definition=(
                "Cross-validation repeatedly fits on a subset of training rows and scores the held-out "
                "fold, then summarizes mean and spread. It estimates selection uncertainty inside train "
                "while keeping a Session test partition untouched."
            ),
            intuition=(
                "Instead of one lucky validation slice, CV rotates who sits out. The mean says typical "
                "performance; the fold standard deviation says how jumpy that estimate is."
            ),
            formal_idea=(
                "For folds {(T_k, E_k)} partitioning the train population, compute S_k = score(fit(T_k), E_k). "
                "Report mean(S) and std(S). Group and time splitters constrain which rows may co-occur in T_k."
            ),
            why_it_matters=(
                "Single holdout rankings are noisy on small data.",
                "Fold spread warns when a 'winner' is fragile.",
                "Nested preprocess-in-fold prevents train-global statistics from leaking into fold scores.",
            ),
            how_buildml_uses=(
                "Session.cv_score draws folds only from train and never scores Session test rows.",
                "Optional PreprocessRecipe refits dates/text/impute/encode/binning/scale/"
                "reduce(pca)/select(variance|univariate|model)/outliers(cap|detect) "
                "on each fold-train. Custom transforms and resample stay Session-global.",
                "grid_search and randomized_search nest train-fold CV for hyperparameter trials.",
                "Session.nested_cv_score adds an outer loop so post-selection estimates do not "
                "reuse the same folds that chose estimator params or fold-local recipe knobs "
                "(select_k, n_bins, and other SAFE_RECIPE_KNOBS). Inner search may be grid, "
                "randomized, or Optuna (inner_search='optuna' with buildml[optuna]).",
                "cv_strategy selects stratified, group, stratified_group, or time fold builders.",
            ),
            interpretation_rules=(
                "Always report population=train, fold count, strategy, and mean±std for the primary metric.",
                "If Session impute/scale ran before CV without a fold recipe, treat preprocess honesty as limited.",
                "After hyperparameter search, prefer nested_cv_score outer mean±std over inner search means.",
                "Confirm the selected recipe once on validation or test after search.",
                "Large fold std relative to mean→std gaps means ranks are unstable.",
            ),
            assumptions=(
                "Fold construction matches the dependence structure (i.i.d., group, or time).",
                "The scoring metric matches the selection goal.",
                "Enough rows/groups exist to form the requested number of folds.",
            ),
            failure_modes=(
                "Using test rows inside CV folds.",
                "Fitting scalers on full train before fold scoring without documenting the leak.",
                "Group CV with fewer groups than folds.",
                "Reporting inner search scores as if they were untouched outer estimates.",
            ),
            anti_patterns=(
                "Tuning until CV looks perfect, then reporting that CV mean as the final test claim.",
                "Ignoring fold std when two configs differ by less than the noise.",
                "Applying shuffled K-fold to strong time series without a time strategy.",
            ),
            worked_example_pattern=(
                "Split (group/time/random) → optional fold PreprocessRecipe → cv_score or search on train.",
                "For tuned selection claims, run nested_cv_score and read outer mean±std.",
                "Refit winner on full train; evaluate once on the held-out partition.",
            ),
            related_concepts=("leakage-boundary", "model-selection", "evaluation-partitions", "data-splitting"),
        ),
        _note(
            key="diagnostic-uncertainty",
            title="Diagnostic uncertainty",
            summary="Metrics, curves, and importances are sample estimates—not exact truths about the world.",
            definition=(
                "Diagnostic uncertainty is the recognition that reported figures—scores, p-values, "
                "importance ranks, drift stats—vary with sample size, split, and data composition, and "
                "must be read with those limits attached."
            ),
            intuition=(
                "A reliability curve built on forty positives will wiggle. That wiggle is the diagnostic "
                "talking about your sample, not necessarily about a broken model. Uncertainty asks you to "
                "slow down before acting on a single dramatic number."
            ),
            formal_idea=(
                "A statistic T computed on a finite sample is a random variable. Without repeats, intervals, "
                "or stable partitions, observing T = t does not identify the deployment value of T. Cutoffs "
                "on T are review prompts unless a domain-validated threshold exists."
            ),
            why_it_matters=(
                "Overconfident diagnostics drive unnecessary feature drops and false alarms.",
                "Small holdouts make model ranks and calibration plots unstable.",
                "Stakeholders deserve uncertainty language beside single-point KPIs.",
            ),
            how_buildml_uses=(
                "EDA Teaching Studio and model diagnostics pair figures with caveats and sample context.",
                "Explain findings can carry confidence and limitations fields for evidence.",
                "Concept notes mark conventional cutoffs (VIF, p-values) as review flags.",
                "Permutation importance and drift tools expect partition-aware reading.",
            ),
            interpretation_rules=(
                "Prefer comparing diagnostics across seeds/folds when ranks jump.",
                "Large-n tiny p-values: inspect effect size and plots, not stars alone.",
                "Importance near zero on a small validation set is weak evidence for deletion.",
                "Associations in diagnostics are not causal effects.",
            ),
            assumptions=(
                "The scored partition is relevant to the decision being made.",
                "Measurement noise and labeling error are not dominating T.",
                "Users will not treat exploratory cutoffs as hard launch gates without domain backing.",
            ),
            failure_modes=(
                "Acting on a single noisy subgroup rate as if it were a stable truth.",
                "Cherry-picking the split where a diagnostic looks clean.",
                "Conflating statistical significance with practical importance.",
            ),
            anti_patterns=(
                "Deleting features because one importance run ranked them last.",
                "Declaring 'no drift' from an underpowered comparison.",
                "Reporting four-decimal metrics on n = 30.",
            ),
            worked_example_pattern=(
                "Compute a metric on validation; repeat with another seed or fold; note the spread.",
                "Downsample the holdout and redraw a curve to see instability.",
                "Write the decision you might take and the uncertainty that should block it.",
            ),
            related_concepts=("model-selection", "feature-importance", "dataset-drift", "overfitting"),
        ),
        _note(
            key="probability-calibration",
            title="Probability calibration",
            summary="Calibration asks whether predicted probabilities match observed event frequencies.",
            definition=(
                "A model is calibrated if, among cases given score p, the long-run frequency of the event "
                "is about p. Calibration is separate from discrimination (ranking ability)."
            ),
            intuition=(
                "If you forecast 70% rain for ten days, it should rain about seven of them. A model can "
                "rank rainy days above dry days yet still say 90% when the true rate is 60%."
            ),
            formal_idea=(
                "Discrimination concerns ordering by score; calibration concerns P(Y=1 | s(x)=p) ≈ p. "
                "Brier score decomposes calibration and sharpness components; reliability diagrams estimate "
                "the calibration curve in bins. Calibrators must be fit on data not used to evaluate them."
            ),
            why_it_matters=(
                "Thresholding and expected-value decisions need trustworthy probabilities.",
                "Resampling and class weights can distort probability scales.",
                "Ranking metrics can look fine while probability-based policies fail.",
            ),
            how_buildml_uses=(
                "Model diagnostics expose reliability-style views and Brier-related summaries where available.",
                "Teaching Studio separates 'orders well' from 'probabilities mean what they say'.",
                "Threshold tooling should be read together with calibration quality.",
                "Leakage rules apply: fit calibrators without reusing their evaluation rows.",
            ),
            interpretation_rules=(
                "Points near the diagonal on a reliability plot support calibration; systematic bows do not.",
                "Brier score mixes calibration and sharpness—pair it with a curve.",
                "With few positives, bin estimates are noisy—treat wiggles as uncertainty.",
                "After heavy resampling, expect to re-check calibration on natural-prevalence data.",
            ),
            assumptions=(
                "Scores are intended as probabilities, not merely ranking margins.",
                "Evaluation labels are reliable enough to estimate frequencies.",
                "Binning or isotonic/platt fitting uses an appropriate held-out design.",
            ),
            failure_modes=(
                "Fitting a calibrator on the same rows used to judge calibration.",
                "Trusting raw tree vote fractions as probabilities under imbalance.",
                "Calibrating on outdated prevalence after a policy shift.",
            ),
            anti_patterns=(
                "Equating high AUC with calibrated probabilities.",
                "Applying temperature scaling without a dedicated calibration split.",
                "Ignoring that multiclass calibration needs per-class or joint checks.",
            ),
            worked_example_pattern=(
                "On validation, bin predictions and compare mean predicted p to observed rate.",
                "Note discrimination (AUC/PR) and calibration (curve/Brier) as separate findings.",
                "If decisions need costs, adjust thresholds only after reading calibration quality.",
            ),
            related_concepts=("thresholds", "evaluation-partitions", "class-imbalance"),
        ),
        _note(
            key="thresholds",
            title="Decision thresholds",
            summary="A threshold turns scores or probabilities into discrete actions under stated costs.",
            definition=(
                "A decision threshold t maps a continuous score s(x) to an action, typically predict "
                "positive if s(x) ≥ t. Choosing t trades precision, recall, and asymmetric error costs; "
                "it does not retrain the underlying scorer."
            ),
            intuition=(
                "The model outputs a dial from 'probably no' to 'probably yes'. The threshold is where you "
                "decide to act—send an alert, block a payment, call a patient. 0.5 is only special if costs "
                "and calibration make it so."
            ),
            formal_idea=(
                "For binary decisions, pick t to optimize an expected-cost or utility criterion on "
                "validation scores, optionally under constraints (max false-positive rate, min recall). "
                "The ROC/PR operating point moves with t; the ranking model stays fixed."
            ),
            why_it_matters=(
                "Default 0.5 rarely matches product false-positive/false-negative costs.",
                "Threshold search on test contaminates the final estimate.",
                "Imbalance makes useful t often far from 0.5 even when calibrated.",
            ),
            how_buildml_uses=(
                "Session.tune_threshold sweeps cutoffs and can minimize expected cost via fp_cost/fn_cost.",
                "Diagnostics expose confusion matrices, operating points, and optional HTML boards.",
                "Session workflows expect threshold choices after a scorer is fixed on validation.",
                "Explain notes warn against mining test for the prettiest confusion matrix.",
            ),
            interpretation_rules=(
                "Select t on validation; confirm once on untouched test if needed for reporting.",
                "Plot metric-versus-threshold; do not trust a single default.",
                "If probabilities are miscalibrated, threshold meaning shifts—fix calibration language first.",
                "Cost ratio changes preferred t more than tiny AUC differences do.",
            ),
            assumptions=(
                "Scores are monotonically related to positive risk (better ranking helps).",
                "Validation reflects deployment prevalence and labeling.",
                "Action costs are roughly known or constrained.",
            ),
            failure_modes=(
                "Hard-coding 0.5 after training on resampled data.",
                "Choosing t on test and advertising that confusion matrix.",
                "Changing t in production without monitoring precision/recall drift.",
            ),
            anti_patterns=(
                "Reporting only the threshold that maximizes accuracy under imbalance.",
                "Retuning t daily on a tiny live sample without variance controls.",
                "Treating threshold tuning as a substitute for better features or labels.",
            ),
            worked_example_pattern=(
                "Fix a model; sweep t on validation; record precision, recall, and cost proxy.",
                "Pick t under an explicit constraint (e.g. recall ≥ 0.8).",
                "Apply that t once to test; report partition-tagged confusion counts.",
            ),
            related_concepts=("probability-calibration", "class-imbalance", "evaluation-partitions"),
        ),
        _note(
            key="checkpoint-integrity",
            title="Checkpoint integrity",
            summary="A checkpoint preserves data state, split membership, and workflow history for safe resume.",
            definition=(
                "Checkpoint integrity means a saved bundle can be reattached only when row/column "
                "fingerprints, schema expectations, and bundle contents match enough to restore a coherent "
                "Session—or fail loudly when they do not."
            ),
            intuition=(
                "A checkpoint is a save file for the lab notebook: data version, who is train versus test, "
                "and what you already did. Loading someone else's edited CSV into that save file should "
                "raise alarms, not silently continue."
            ),
            formal_idea=(
                "A bundle B stores payload plus a manifest M of hashes/schema/split/history. Reattach "
                "succeeds when validate(M, current_source) passes. data_only loads intentionally drop "
                "workflow semantics; full loads restore them when compatible."
            ),
            why_it_matters=(
                "Mid-loop exit and resume are unsafe without membership and history fidelity.",
                "Silent schema drift after resume recreates leakage and schema bugs.",
                "Model artifacts and checkpoint bundles solve different persistence problems.",
            ),
            how_buildml_uses=(
                "Session.checkpoint_save / checkpoint_load write and validate resumable bundles.",
                "Manifest checks catch changed rows/columns and incompatible bundle content.",
                "data_only=True loads data without claiming prior workflow semantics still hold.",
                "Roles, splits, and history ride with full checkpoints for Teaching Studio continuity.",
            ),
            interpretation_rules=(
                "A validation failure is a stop sign—reconcile data or start a new session path.",
                "Success means structural compatibility, not that renamed fields kept meaning.",
                "Prefer full checkpoint resume when you need the same split and operation history.",
            ),
            assumptions=(
                "Underlying bytes were not hand-edited in conflicting ways.",
                "Package versions remain compatible with the bundle format.",
                "Users distinguish workflow checkpoints from exported model files.",
            ),
            failure_modes=(
                "Replacing the data file but keeping an old manifest expectation.",
                "Resuming with data_only and assuming old transforms still apply.",
                "Sharing bundles across incompatible BuildML versions without migration.",
            ),
            anti_patterns=(
                "Disabling validation to 'just load it'.",
                "Using checkpoints as a substitute for dataset provenance and source control.",
                "Mixing model-bundle paths with checkpoint_load APIs.",
            ),
            worked_example_pattern=(
                "Split and impute; checkpoint_save; reload and confirm membership and roles match.",
                "Alter a column or row counts; show reattach validation failing.",
                "Contrast data_only load versus full load on the same bundle.",
            ),
            related_concepts=("feature-schema", "reproducibility", "data-splitting"),
        ),
        _note(
            key="reproducibility",
            title="Reproducibility",
            summary="Seeds, versions, inputs, and recorded choices are all required to recreate a result.",
            definition=(
                "Reproducibility is the ability for someone (including future you) to regenerate the same "
                "modeling artifacts and metrics from documented inputs, package versions, seeds, and "
                "operation parameters—within the limits of platform nondeterminism."
            ),
            intuition=(
                "A result without its recipe is a story. Seeds fix supported randomness; they do not freeze "
                "library upgrades, data refreshes, or unordered parallel reductions on every platform."
            ),
            formal_idea=(
                "A run is specified by (data identifier, code/version vector, parameter vector, seed). "
                "Bitwise identical outputs are a strong form; weaker forms match metrics within tolerance. "
                "History logs record calls but do not replace immutable source snapshots."
            ),
            why_it_matters=(
                "Unreproducible winners cannot be audited or safely deployed.",
                "Debugging regressions needs the same split and preprocessing knobs.",
                "Teaching and review require replayable Session paths.",
            ),
            how_buildml_uses=(
                "Session history records operation names and parameters for audit trails.",
                "Checkpoint bundles capture data state, splits, roles, and history together.",
                "Seeds control supported stochastic steps (splits, some samplers/models) when passed.",
                "Engine/materialization choices should be recorded because they can affect numerics.",
            ),
            interpretation_rules=(
                "Same seed + same inputs should match for supported RNGs; investigate if not.",
                "Metric drift after a dependency bump is a review flag, not a mystery.",
                "A history log without data hashes is incomplete provenance.",
            ),
            assumptions=(
                "Source data can be retrieved at the recorded version.",
                "Users pin or record environment versions for serious runs.",
                "Nondeterministic kernels are known when exact bitwise match is required.",
            ),
            failure_modes=(
                "Refreshing a CSV in place while citing old metrics.",
                "Omitting seeds on split/resample and calling the run reproducible.",
                "Relying on notebook cell order instead of Session history.",
            ),
            anti_patterns=(
                "Publishing scores without code, data id, or split definition.",
                "Changing multiple knobs at once when diagnosing a metric move.",
                "Assuming checkpoints alone replace dataset versioning in object storage.",
            ),
            worked_example_pattern=(
                "Run split+fit twice with the same seed; confirm matching membership and metrics.",
                "Change one library version; note which outputs move.",
                "Save a checkpoint and replay explain()/metrics after reload.",
            ),
            related_concepts=("checkpoint-integrity", "engine-choice", "data-splitting"),
        ),
        _note(
            key="engine-choice",
            title="Execution engine choice",
            summary="The engine controls data access and materialization, not the meaning of the estimator.",
            definition=(
                "An execution engine (pandas, polars, duckdb, ...) is the compute backend behind a Dataset "
                "handle. It determines how scans, filters, and aggregates run and when data materializes to "
                "sklearn-facing frames—not which loss the classifier optimizes."
            ),
            intuition=(
                "Engines are kitchens, not recipes. A larger kitchen helps when ingredients do not fit on "
                "one counter. Switching kitchens mid-cooking can copy food around; it does not improve the "
                "seasoning by magic."
            ),
            formal_idea=(
                "Dataset operations are evaluated under engine e ∈ {pandas, polars, duckdb, ...} and mode "
                "m ∈ {memory, lazy}. Estimator fit consumes a materialized design matrix; "
                "engine choice affects how that matrix is produced and at what memory cost."
            ),
            why_it_matters=(
                "Wrong engine/mode choices cause OOM or unnecessary conversions.",
                "Conversions can copy data and perturb performance measurements.",
                "Team interoperability may favor Arrow/Polars/DuckDB even when models still need pandas.",
            ),
            how_buildml_uses=(
                "Session.ingest can recommend engine/mode from scale estimates; users may override.",
                "Pandas remains the canonical sklearn materialization path in the current release.",
                "prepare_design_matrix projects requested columns first and can sample on Polars/DuckDB "
                "before the Pandas design matrix is produced; it does not enable out-of-core sklearn.",
                "Soft gates warn near ~250 MiB at fit, scale, and to_pandas boundaries; hard gates refuse "
                "when hard_limit_bytes or BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES is set.",
                "check_materialization returns nbytes telemetry plus guidance to keep prep lazy and "
                "materialize only the train design matrix.",
                "Polars and DuckDB are optional dependencies for larger or SQL-friendly workflows.",
                "with_engine / with_mode adjust the handle without changing role or split semantics.",
            ),
            interpretation_rules=(
                "Prefer lazy mode when row/byte estimates exceed comfortable RAM—not as a fashion choice.",
                "Treat soft materialization warnings as scale signals; hard limits abort intentional oversized copies.",
                "Measure end-to-end wall time including materialization, not only SQL fragments.",
                "If metrics change after engine switches, suspect conversion/dtype issues before celebrating gains.",
            ),
            assumptions=(
                "Optional engine packages are installed when selected.",
                "Semantics of filters and null handling are understood per engine.",
                "Final ML fit still materializes a compatible in-memory matrix when required.",
            ),
            failure_modes=(
                "Forcing pandas memory mode on multi-GB inputs.",
                "Ping-ponging engines and attributing noise to model improvements.",
                "Assuming DuckDB/Polars removes the need for train-fitted sklearn transforms.",
            ),
            anti_patterns=(
                "Converting engines as a ritual on tiny frames.",
                "Treating engine logos as model-quality signals.",
                "Skipping scale estimates at ingest and guessing.",
            ),
            worked_example_pattern=(
                "Ingest a medium frame; note recommended engine/mode from the ingest report.",
                "Run the same filter/profile path under two engines; compare time and peak memory.",
                "Materialize to pandas only at the model boundary; keep roles/splits unchanged.",
            ),
            related_concepts=("reproducibility", "checkpoint-integrity", "feature-schema"),
        ),
        _note(
            key="baselines",
            title="Baselines",
            summary="A baseline anchors whether added model complexity improves the decision-relevant metric.",
            definition=(
                "A baseline is a simple, transparent predictor—prevalence/majority class, mean/median "
                "target, or a shallow policy—evaluated under the same partition and metric protocol as "
                "candidate models."
            ),
            intuition=(
                "Before praising a complex model, ask whether predicting the everyday average already "
                "gets you most of the way. Baselines keep you honest when the leaderboard looks exciting "
                "but barely beats 'always say no'."
            ),
            formal_idea=(
                "Let f_0 be a restricted hypothesis (constant classifier/regressor or simple rule). "
                "Candidates f must beat S(f_0) on the evaluation design by a margin that justifies cost. "
                "f_0 parameters, when any, are learned from train only."
            ),
            why_it_matters=(
                "Prevents shipping complexity that does not buy decision value.",
                "Frames metric movement in absolute terms stakeholders understand.",
                "Surfaces label leakage when fancy models only match a trivial rule on shuffled labels.",
            ),
            how_buildml_uses=(
                "Model workflows encourage comparing candidates against simple reference scores.",
                "Teaching Studio and Concept Academy treat baselines as part of selection discipline.",
                "Same Session split and metric wiring should score baseline and candidates together.",
                "Imbalance teaching pairs majority-class baselines with PR-focused metrics.",
            ),
            interpretation_rules=(
                "Always report baseline metric beside candidate metric on the same partition.",
                "Gains smaller than repeat noise are ties with the baseline.",
                "A regression median baseline is often stronger than mean under heavy tails.",
            ),
            assumptions=(
                "Baseline uses only information allowed at score time.",
                "Metric matches the product decision.",
                "Train-only statistics define any constant predictors.",
            ),
            failure_modes=(
                "Computing a 'baseline' using test labels or full-data prevalence after peeking.",
                "Comparing a tuned pipeline to an untuned baseline unfairly—or the reverse.",
                "Ignoring that a strong baseline may already be the production policy.",
            ),
            anti_patterns=(
                "Skipping baselines because the model family is sophisticated.",
                "Reporting lift without stating the reference.",
                "Training a complex model to rediscover the majority class.",
            ),
            worked_example_pattern=(
                "Define majority/mean/median baseline from train statistics only.",
                "Score baseline and candidate on validation with the same metric.",
                "Decide whether complexity is justified before touching test.",
            ),
            related_concepts=("model-selection", "evaluation-partitions", "class-imbalance"),
        ),
        _note(
            key="overfitting",
            title="Overfitting",
            summary="Overfitting is a harmful gap between training fit and behavior on relevant unseen rows.",
            definition=(
                "Overfitting occurs when a model captures training-sample idiosyncrasies—noise, spurious "
                "correlations, or leakage—so that training performance looks strong while performance on "
                "appropriately held-out data is materially worse."
            ),
            intuition=(
                "Memorizing yesterday's trivia questions is not the same as understanding the subject. "
                "Huge train scores with soft validation scores mean the model memorized the homework."
            ),
            formal_idea=(
                "With hypothesis class capacity H and finite train sample, empirical risk R̂_train(f) can "
                "fall far below risk on a fresh draw. Learning curves and train–validation gaps help "
                "separate high-variance fits from data-limited regimes and from leakage artifacts."
            ),
            why_it_matters=(
                "Overfit models fail in production despite impressive notebook screenshots.",
                "Gap diagnosis guides whether you need more data, less capacity, or leakage fixes.",
                "Early stopping, regularization, and simpler features are tools—not moral virtues.",
            ),
            how_buildml_uses=(
                "Partition-tagged metrics make train-versus-validation gaps visible.",
                "assert_can_fit and train-fitted transforms reduce classic preprocessing leakage paths.",
                "Diagnostics and Teaching Studio encourage learning-curve style reasoning.",
                "Model selection notes warn that mining validation aggressively overfits the selection process.",
            ),
            interpretation_rules=(
                "Compare the same metric on train and validation; a large gap is a review flag.",
                "If both are poor, you may be underfit or missing features—not classic overfit.",
                "A gap that appears only after adding a suspicious feature suggests leakage first.",
                "Tiny validation sets make gaps noisy—confirm with another split/seed.",
            ),
            assumptions=(
                "Holdout rows are relevantly unseen (not near-duplicates of train).",
                "Metric definitions match across partitions.",
                "Training actually optimized the reported training metric (or a related surrogate).",
            ),
            failure_modes=(
                "Tuning until validation peaks, then calling that peak an unbiased estimate.",
                "Mistaking label leakage for 'great generalization'.",
                "Heavy memorizing models on tiny n without regularization.",
            ),
            anti_patterns=(
                "Reporting only training accuracy.",
                "Adding features until train loss is zero without holdout checks.",
                "Equating any train > validation gap with immediate feature deletion panics.",
            ),
            worked_example_pattern=(
                "Fit a high-capacity model and a regularized/simpler sibling on the same split.",
                "Plot or tabulate train versus validation scores as capacity or epochs change.",
                "If the gap is large, try leakage audit and capacity control before new feature factories.",
            ),
            related_concepts=("evaluation-partitions", "diagnostic-uncertainty", "leakage-boundary", "model-selection"),
        ),
        _note(
            key="feature-importance",
            title="Feature importance",
            summary="Importance describes fitted-model reliance under a dataset and score—not causal effect.",
            definition=(
                "Feature importance methods quantify how much a fitted model's score degrades (or how much "
                "impurity/gain concentrates) when a feature's information is removed or perturbed, relative "
                "to a chosen evaluation set and metric."
            ),
            intuition=(
                "If shuffling 'loan amount' wrecks accuracy, the model was using that column. That does "
                "not prove loan amount causes defaults; a correlated substitute might share the credit, "
                "and a confounder might be the real story."
            ),
            formal_idea=(
                "Permutation importance estimates Δ = S(f, D) − S(f, D with column j shuffled). Large Δ "
                "means reliance under (f, D, S). Correlated features can split Δ; causal effects require "
                "identification assumptions this diagnostic does not provide."
            ),
            why_it_matters=(
                "Guides audit questions and error analysis without pretending to be causal discovery.",
                "Unstable ranks on small holdouts mislead feature deletion sprees.",
                "Stakeholders often misread importance as 'drivers'—teaching must correct that.",
            ),
            how_buildml_uses=(
                "Model diagnostics can compute permutation-style reliance on a chosen partition.",
                "Concept Academy and Teaching Studio stress association-versus-causality language.",
                "Importances inherit the fitted pipeline, including encode/impute/scale choices.",
                "Explain outputs should carry partition and metric context with ranks.",
            ),
            interpretation_rules=(
                "Report partition, metric, and repeat variability with every importance table.",
                "Near-ties among correlated features are expected—do not over-order them.",
                "Low importance ≠ safe to drop without a refit experiment.",
                "Importance is not a leakage test by itself; timing/semantics still need review.",
            ),
            assumptions=(
                "The evaluation set is large enough for shuffle noise to average out somewhat.",
                "The metric matches the question you are asking about reliance.",
                "Feature j is meaningfully defined after preprocessing.",
            ),
            failure_modes=(
                "Interpreting ranks as causal drivers for policy changes.",
                "Running importance on train and concluding the model 'understands' structure.",
                "One-hot fragments scattering a single concept's importance across many columns.",
            ),
            anti_patterns=(
                "Auto-deleting the bottom half of an importance list.",
                "Comparing importances across differently preprocessed models as if commensurate.",
                "Using importance to justify keeping a clearly post-outcome column.",
            ),
            worked_example_pattern=(
                "Fit a model; compute permutation importance on validation with repeats.",
                "Note top features and correlated companions; form audit questions, not causes.",
                "Ablate a top feature with a refit to confirm metric impact.",
            ),
            related_concepts=("diagnostic-uncertainty", "model-selection", "mutual-information", "leakage-boundary"),
        ),
        _note(
            key="dataset-drift",
            title="Dataset drift",
            summary="Drift is a measured distribution change across defined populations or time windows.",
            definition=(
                "Dataset drift is a statistically detected difference in feature and/or label "
                "distributions between two collections—train versus recent traffic, batch A versus batch B—"
                "without automatically proving that model quality changed."
            ),
            intuition=(
                "If last quarter's ages look older than training ages, something in the world or in the "
                "pipeline moved. That something might break the model—or it might be harmless seasonality. "
                "Drift is a smoke alarm, not a fire report."
            ),
            formal_idea=(
                "Compare samples P and Q via distances or tests on marginals/joints (PSI-like scores, "
                "KS statistics, classifier two-sample tests). Rejecting P = Q is not the same as proving "
                "a drop in utility of f; label-free drift diagnoses inputs, not errors."
            ),
            why_it_matters=(
                "Train–serve skew and temporal change are common silent failures.",
                "Invalid splits can look like 'drift' between partitions.",
                "Monitoring needs drift cues when labels arrive late.",
            ),
            how_buildml_uses=(
                "EDA and diagnostics can contrast distributions across partitions or snapshots.",
                "Teaching Studio frames drift findings with effect-size and data-collection questions.",
                "Split membership lets you compare train versus holdout as a first drift screen.",
                "Checkpoints help freeze a reference snapshot for later comparison.",
            ),
            interpretation_rules=(
                "Treat significant tests as review flags; inspect magnitude, support, and collection changes.",
                "Train–test drift under a random split suggests leakage of structure or bad randomization.",
                "Feature drift without labels cannot quantify accuracy drop—pair with delayed labels when possible.",
                "Large-n tiny differences can be 'significant' yet operationally minor.",
            ),
            assumptions=(
                "Compared cohorts are defined cleanly (time windows, filters).",
                "Schemas align so similarly named columns are comparable.",
                "Missingness and category policies are applied consistently before comparison.",
            ),
            failure_modes=(
                "Alerting on every significant p-value without effect-size triage.",
                "Comparing raw train to heavily filtered production logs.",
                "Ignoring label drift while only watching features.",
            ),
            anti_patterns=(
                "Retraining automatically on any drift alert without diagnosis.",
                "Using drift scores as a substitute for holdout metrics when labels exist.",
                "Declaring 'no drift' from a single weak univariate check.",
            ),
            worked_example_pattern=(
                "Define reference = train features and candidate = recent unlabeled batch.",
                "Compute per-column drift screens; rank by effect size, not only p-value.",
                "Trace top movers to collection or product changes before altering the model.",
            ),
            related_concepts=("data-splitting", "diagnostic-uncertainty", "feature-schema"),
        ),
        _note(
            key="mutual-information",
            title="Mutual information",
            summary="Mutual information scores shared information between a feature and the target without assuming linearity.",
            definition=(
                "Mutual information (MI) between feature X_j and target Y measures how much knowing X_j "
                "reduces uncertainty about Y. It captures nonlinear association under the estimator used, "
                "but it is not a causal effect and not by itself proof of leakage."
            ),
            intuition=(
                "If knowing zip code sharply narrows which churn label you expect, MI is high. That can "
                "mean a useful pattern, a proxy for something else, or a peek at information you should "
                "not use—MI alone does not say which."
            ),
            formal_idea=(
                "I(X; Y) = H(Y) − H(Y|X) ≥ 0, zero under independence for the population quantities. "
                "Practical estimates depend on discretization, encoding, sample size, and bias-correction "
                "choices; ranks are usually more trustworthy than raw magnitudes across heterogeneous types."
            ),
            why_it_matters=(
                "Prioritizes which columns deserve semantic and timing review before modeling.",
                "Flags surprisingly strong associations that may indicate leakage carriers.",
                "Complements linear correlation screens for nonlinear dependence.",
            ),
            how_buildml_uses=(
                "EDA Teaching Studio can surface MI-style association ranks for feature triage.",
                "Concept Academy insists association ≠ causality and ≠ automatic column keep/drop.",
                "Roles should exclude ids before interpreting MI as predictive signal.",
                "Use MI alongside leakage-boundary checks on timing and availability.",
            ),
            interpretation_rules=(
                "High MI is a review flag: confirm semantics, timing, and proxies before fitting.",
                "Near-zero MI does not prove uselessness under interactions or later transforms.",
                "Compare MI ranks within similar cardinality/type bands when possible.",
                "Treat estimate instability on small n as uncertainty, not precise ordering.",
            ),
            assumptions=(
                "Encoding/binning choices are stated and reasonable for the column type.",
                "Sample used for MI is not the same tiny set you will overinterpret for launch gates.",
                "Target definition matches the modeling label.",
            ),
            failure_modes=(
                "Keeping a post-outcome field because MI is large.",
                "Comparing raw MI across one-hot fragments and continuous columns naively.",
                "Computing MI on full data then leaking selection into the same evaluation.",
            ),
            anti_patterns=(
                "Auto-dropping all low-MI features in one shot.",
                "Calling MI a causal driver in stakeholder slides.",
                "Using MI alone as a leakage detector without a timing audit.",
            ),
            worked_example_pattern=(
                "After roles are set, compute MI ranks of features versus target on train.",
                "Open the top few columns: ask when they become known and what they proxy.",
                "Fit with/without a suspicious high-MI column and compare validation metrics.",
            ),
            related_concepts=("feature-importance", "leakage-boundary", "diagnostic-uncertainty", "column-roles"),
        ),
        _note(
            key="variance-inflation",
            title="Variance inflation factors",
            summary="VIF estimates how much linear dependence among numeric features inflates coefficient variance.",
            definition=(
                "The variance inflation factor for feature j is a collinearity diagnostic from linear "
                "modeling: it estimates how much the variance of a regression coefficient is inflated "
                "because X_j is linearly predictable from the other numeric features."
            ),
            intuition=(
                "If two columns are nearly the same ruler marked in different units, a linear model "
                "cannot stably split credit between them. VIF measures that tangle. Trees may still "
                "predict well; coefficients may not."
            ),
            formal_idea=(
                "VIF_j = 1 / (1 − R_j²) where R_j² comes from regressing X_j on the other features. "
                "VIF_j = 1 means no linear inflation; large VIF signals linear dependence. Conventional "
                "cutoffs (e.g. 5 or 10) are review flags, not universal harm proofs for every estimator."
            ),
            why_it_matters=(
                "Unstable linear coefficients undermine explanation and regularization paths.",
                "Collinearity complicates importance and coefficient stories even when predictions hold.",
                "Guides whether PCA, dropping, or combining features is worth considering for linear models.",
            ),
            how_buildml_uses=(
                "EDA diagnostics can report VIF-style screens on numeric feature columns.",
                "Teaching Studio pairs VIF with estimator-family context (linear vs trees).",
                "Roles should exclude ids/constants before computing VIF.",
                "Concept notes mark thresholds such as VIF > 5 as review prompts.",
            ),
            interpretation_rules=(
                "VIF above a conventional flag (often 5, sometimes 10) triggers collinearity review—not automatic deletion.",
                "Trees can remain predictive under high VIF; linear coefficient stories should not.",
                "VIF is undefined or unhelpful for constant columns and tiny complete-case n.",
                "Association among features ≠ causal structure.",
            ),
            assumptions=(
                "Features are numeric (or encoded numerically in a way that makes linear dependence meaningful).",
                "Enough complete cases exist to estimate the auxiliary regressions.",
                "Interest includes linear-model stability, not only predictive accuracy.",
            ),
            failure_modes=(
                "Dropping features solely to beautify VIF while hurting validated metrics.",
                "Computing VIF on one-hot dummines without understanding the induced dependence.",
                "Ignoring that scaling changes coefficients but not the underlying collinearity geometry.",
            ),
            anti_patterns=(
                "Treating VIF < 5 as proof that features are independent in every sense.",
                "Using VIF as a causality screen.",
                "Applying VIF cutoffs blindly to tree-only workflows.",
            ),
            worked_example_pattern=(
                "Select numeric features with adequate complete cases; compute VIF ranks.",
                "Inspect pairs among high-VIF columns for duplicated measurements.",
                "For a linear model, compare coefficients before/after dropping or combining one member of a pair.",
            ),
            related_concepts=("principal-components", "feature-schema", "model-selection", "feature-scaling"),
        ),
        _note(
            key="principal-components",
            title="Principal component analysis",
            summary="PCA finds orthogonal linear combinations that capture shared numeric variance—not guaranteed predictive value.",
            definition=(
                "Principal component analysis rotates numeric features into uncorrelated components ordered "
                "by captured variance. It is a compression and collinearity tool; explained variance is not "
                "a synonym for supervised utility."
            ),
            intuition=(
                "If many columns move together, PCA builds a few 'summary directions' of that joint motion. "
                "Those directions may help a linear model or visualization—or they may summarize noise you "
                "did not care about."
            ),
            formal_idea=(
                "For centered (usually scaled) matrix X, PCA uses eigenvectors of the covariance (or SVD of X). "
                "Component k maximizes remaining variance subject to orthogonality. Fitting PCA on all rows "
                "before splitting leaks covariance structure into later evaluation."
            ),
            why_it_matters=(
                "Can reduce dimension and ease collinearity for linear methods.",
                "Unscaled inputs let large-magnitude columns dominate components.",
                "Supervised tasks may prefer components chosen by predictive criteria, not variance alone.",
            ),
            how_buildml_uses=(
                "EDA can present explained-variance profiles as unsupervised structure screens.",
                "Session.reduce_dimensions(method='pca') fits PCA on train and stores explained-variance ratios.",
                "PreprocessRecipe(reduce='pca') refits the rotation on each CV fold-train.",
                "Teaching Studio warns that variance explained ≠ target association.",
                "Scaling notes are linked because PCA is scale-sensitive.",
            ),
            interpretation_rules=(
                "A knee in cumulative explained variance is a review cue for compression—not a magic k.",
                "High variance components can still have low mutual information with Y.",
                "Loadings tell linear feature contributions to a component; they are not causal effects.",
                "If one raw feature dominates PC1, check whether scaling was skipped.",
            ),
            assumptions=(
                "Linear combinations are a sensible summary for the numeric block.",
                "Missing values were handled before PCA.",
                "Train-only fit when PCA is inside a predictive pipeline.",
            ),
            failure_modes=(
                "Full-data PCA before split, then modeling on components.",
                "Interpreting PC plots as proof of clusters that matter for Y.",
                "Applying PCA to ids or poorly encoded categoricals.",
            ),
            anti_patterns=(
                "Keeping enough components to hit 95% variance by habit without validation metric checks.",
                "Using PCA to 'fix' leakage-prone features instead of removing them.",
                "Comparing models with different PCA fits as if features matched.",
            ),
            worked_example_pattern=(
                "Scale numeric train features; fit PCA on train; transform validation with the same rotation.",
                "Plot cumulative explained variance; pick a candidate k.",
                "Compare validation metric with raw features versus k components for a linear model.",
            ),
            related_concepts=("variance-inflation", "leakage-boundary", "feature-scaling", "mutual-information"),
        ),
        _note(
            key="normality-screens",
            title="Normality screens",
            summary="Normality tests ask whether a numeric sample is compatible with a Gaussian reference—not whether ML is allowed.",
            definition=(
                "A normality screen applies a statistical test or visual check (histogram, Q–Q) to assess "
                "compatibility of a numeric sample with a normal distribution under the chosen procedure."
            ),
            intuition=(
                "Bell-curve checks answer a narrow question: does this sample look Gaussian enough for "
                "methods that care? Most modern ML estimators do not require Gaussian features. With huge n, "
                "even trivial wiggles look 'significant'."
            ),
            formal_idea=(
                "Tests such as Shapiro–Wilk or D'Agostino compute a statistic sensitive to departures from "
                "normality and a p-value under a null of normality (with known limitations). As n grows, "
                "power rises; practical decisions should pair p-values with skew, tails, and plots."
            ),
            why_it_matters=(
                "Informs optional transforms or robust statistics—not mandatory model families.",
                "Prevents p-value theater from dictating pipelines.",
                "Helps interpret mean/std summaries that assume symmetric noise.",
            ),
            how_buildml_uses=(
                "EDA Teaching Studio may show normality flags beside skew and outlier context.",
                "Screens are diagnostic context for numeric features, not pass/fail gates for Session.fit.",
                "Concept Academy ties screens to scaling and uncertainty literacy.",
                "Roles should focus screens on true numeric measures, not ids.",
            ),
            interpretation_rules=(
                "Small p-value at large n is a review flag: inspect shape and effect, not automatic transform.",
                "Failed normality does not require a parametric model or a log transform.",
                "Many tree and neural estimators do not assume Gaussian features.",
                "Use Q–Q/skew alongside the test name you ran.",
            ),
            assumptions=(
                "Observations used in the test are appropriately i.i.d. for the test's derivation.",
                "The column is continuous enough for the chosen test.",
                "Users will not confuse feature normality with residual normality after a linear fit.",
            ),
            failure_modes=(
                "Forcing Gaussianizing transforms that hurt validation metrics.",
                "Testing after leaking full-data outlier caps into the sample.",
                "Screening ids and categorical codes as if they were continuous measures.",
            ),
            anti_patterns=(
                "Rejecting a model family solely because a feature failed Shapiro–Wilk.",
                "Reporting p-values without n or plots.",
                "Chasing normality until every feature is warped beyond interpretation.",
            ),
            worked_example_pattern=(
                "Pick a skewed numeric feature; view histogram/Q–Q and a normality screen on train.",
                "Note p-value, n, and skew; decide whether a robust imputer or nonlinear model matters more.",
                "Compare validation metric with and without a transform motivated by the screen.",
            ),
            related_concepts=("feature-scaling", "diagnostic-uncertainty", "missing-data"),
        ),
        _note(
            key="outlier-handling",
            title="Outlier handling",
            summary="Outlier fences are heuristic screens; learn them on train and choose detect, cap, or drop deliberately.",
            definition=(
                "Outlier handling estimates numeric fences—commonly IQR Tukey bounds or z-score thresholds—on "
                "training rows, then detects, caps (winsorizes), or drops values outside those frozen bounds."
            ),
            intuition=(
                "A few extreme ages or incomes can dominate means and distance models. Fences ask which points "
                "look unusual under a simple rule. Unusual is not the same as wrong."
            ),
            formal_idea=(
                "For train sample x_j, IQR fences use q1 − k·IQR and q3 + k·IQR. Z-score fences use "
                "μ̂ ± τ·σ̂. Cap replaces x with clip(x; L, U). Drop removes rows with any flagged column. "
                "Parameters (L, U) must be θ̂ = fit(train)."
            ),
            why_it_matters=(
                "Scale-sensitive models can overweight extremes that are measurement errors.",
                "Dropping rows changes effective sample size and can bias evaluation if holdouts are culled casually.",
                "Domain-valid rare events must not be deleted just because a fence fired.",
            ),
            how_buildml_uses=(
                "Session.handle_outliers(...) fits fences on train and supports detect, cap, and drop actions.",
                "Drop rebuilds SplitPlan membership so partitions stay disjoint after row removal.",
                "Structured PreprocessResult records flagged counts, fences, and limitations.",
                "EDA outlier boards remain screening evidence and do not mutate the dataset.",
            ),
            interpretation_rules=(
                "Treat fence hits as review flags unless a contamination mechanism is known.",
                "Prefer detect or cap when row membership must stay aligned with an external key.",
                "If drop removes most of a class or an entire partition, widen fences or stop.",
            ),
            assumptions=(
                "Feature distributions are roughly unimodal enough for IQR/z-score heuristics.",
                "Train support represents the scoring domain's typical range.",
                "Missing values were handled or are acceptable for the chosen numeric coercion.",
            ),
            failure_modes=(
                "Fitting fences on full data so holdout extremes define training bounds.",
                "Dropping test rows to improve a published metric.",
                "Applying the same fence to multimodal mixtures without segment review.",
            ),
            anti_patterns=(
                "Deleting every flagged point by default.",
                "Reporting 'outliers removed' without method, partition, and counts.",
                "Using IsolationForest labels from EDA as silent row filters without a plan.",
            ),
            worked_example_pattern=(
                "Split; run handle_outliers(action='detect') and inspect n_flagged_train.",
                "Decide cap versus drop with a domain reason; re-check train/holdout ranges.",
                "Fit a scale-sensitive model and compare validation metrics with and without capping.",
            ),
            related_concepts=("leakage-boundary", "diagnostic-uncertainty", "feature-scaling"),
        ),
        _note(
            key="feature-binning",
            title="Feature binning",
            summary="Binning replaces numeric magnitudes with train-fitted intervals for compactness or stepwise effects.",
            definition=(
                "Feature binning discretizes a numeric column into ordered intervals whose edges are estimated "
                "from training data—quantile occupancy or uniform width—and then emits ordinal codes or one-hot "
                "indicators for those intervals."
            ),
            intuition=(
                "Sometimes 'about 30–40' is more stable than the exact age 37. Binning trades within-bin detail "
                "for simpler, explainable intervals. The cut-points must come from train pages only."
            ),
            formal_idea=(
                "Learn edges e0 < … < eB on train (quantile or uniform). Map x to bin index i with e_i ≤ x < e_{i+1}, "
                "using open end bins (−∞, ∞) so score-time extremes remain defined. Encode as i or as 1[bin=i]."
            ),
            why_it_matters=(
                "Can stabilize noisy continuous inputs and make partial-dependence style explanations easier.",
                "Discards within-bin magnitude that trees might have used as split thresholds.",
                "Edge choice changes occupancy and can collapse under discrete or sparse train support.",
            ),
            how_buildml_uses=(
                "Session.bin(...) fits edges on train and replaces source columns with ordinal or one-hot bins.",
                "PreprocessResult exposes edges_ and limitations about information loss.",
                "Explain catalog links binning to schema and leakage-boundary concepts.",
                "Open ±inf ends keep transform defined when scoring values exceed train range.",
            ),
            interpretation_rules=(
                "If many rows land in end bins at score time, review drift before trusting the discretization.",
                "Ordinal bin codes are ranks, not guaranteed equal interval distances.",
                "Tree-only workflows often skip binning unless interpretability or monotonic constraints demand it.",
            ),
            assumptions=(
                "Within-bin variation is expendable for the decision task.",
                "Requested n_bins is compatible with train unique-value support.",
                "Roles correctly identify which numeric columns should be discretized.",
            ),
            failure_modes=(
                "Learning quantiles on all rows so holdout cut-points leak into the recipe.",
                "One-hot exploding after choosing too many bins on high-cardinality discrete numerics.",
                "Refitting edges after peeking at test residuals.",
            ),
            anti_patterns=(
                "Binning every numeric column by habit before trying the raw scale.",
                "Treating collapsed two-bin outputs as rich ordinal structure.",
                "Hiding target-based cut-points inside unsupervised binning claims.",
            ),
            worked_example_pattern=(
                "On train, compare quantile versus uniform edges for one skewed feature.",
                "Fit Session.bin(...); inspect edges_ and validation occupancy.",
                "Compare a linear model on bins versus a tree on the original continuous column.",
            ),
            related_concepts=("feature-schema", "leakage-boundary", "encoding-imputation-scaling"),
        ),
        _note(
            key="target-encoding",
            title="Target encoding",
            summary="Target encoding replaces categories with smoothed label means and needs out-of-fold discipline on train.",
            definition=(
                "Target encoding maps each category to a shrinkage estimate of the training label mean for that "
                "level. Leakage-safe application writes out-of-fold means on train rows and full-train means on "
                "holdout rows."
            ),
            intuition=(
                "If 'premium' customers approve more often, the word premium can become a number near that rate. "
                "Computing that rate with the same row's label still in the average is cheating on train."
            ),
            formal_idea=(
                "For level v, let n_v and ȳ_v be train count and mean. Smoothed mean is "
                "(n_v·ȳ_v + α·ȳ)/(n_v + α). For train row i in fold f, estimate ȳ_v from rows with fold ≠ f. "
                "Holdout rows use the full-train smoothed map."
            ),
            why_it_matters=(
                "High-cardinality categoricals often need a compact supervised encoding.",
                "Without OOF discipline, models memorize label noise and validation metrics lie.",
                "Nested CV is still required when the encoding itself is tuned during model selection.",
            ),
            how_buildml_uses=(
                "Session.encode(method='target') applies OOF train values and full-train holdout maps.",
                "Safer alternatives (one-hot, ordinal, infrequent pooling) remain first-line defaults.",
                "PreprocessResult warnings remind callers to keep selection-time encoding fold-local.",
                "Leakage-boundary teaching treats target means as supervised statistics.",
            ),
            interpretation_rules=(
                "Large gaps between OOF train encodings and holdout maps signal unstable levels.",
                "Rare levels shrink toward the global prior; do not over-interpret their codes.",
                "If model selection reshuffles folds, prefer recipe-based fold-local encoding over a frozen Session plan.",
            ),
            assumptions=(
                "Target labels are available for the train partition and match the modeling task.",
                "Category labels are consistently spelled across partitions.",
                "Smoothing α is large enough to stabilize rare levels for the available n.",
            ),
            failure_modes=(
                "Fitting target means on all rows before splitting.",
                "Using in-fold means on train features that later enter the same model.",
                "Tuning encoding hyperparameters on the test partition.",
            ),
            anti_patterns=(
                "Target-encoding ids or timestamps.",
                "Claiming 'target-safe' while fitting on full data.",
                "Skipping holdout confirmation after a large cardinality collapse.",
            ),
            worked_example_pattern=(
                "Split; compare infrequent one-hot versus target encoding on a high-cardinality column.",
                "Inspect OOF warnings and validation metric lift.",
                "If selecting models with CV, move supervised encoding into fold-local preparation.",
            ),
            related_concepts=("categorical-encoding", "leakage-boundary", "cross-validation"),
        ),
        _note(
            key="feature-selection",
            title="Feature selection",
            summary="Feature selection chooses a subset using train-only scores and must be frozen before honest evaluation.",
            definition=(
                "Feature selection retains a subset of candidate predictors according to a rule—variance filters, "
                "univariate association scores, or model-based importance—fit exclusively on training rows and then "
                "applied as a frozen keep-list."
            ),
            intuition=(
                "Not every column earns its place. Selection asks which features look useful on train under a stated "
                "rule, then sticks to that shortlist when scoring holdouts."
            ),
            formal_idea=(
                "Estimate a score s_j = S(x_j^{train}, y^{train}) or a model-derived weight, choose keep-set K = "
                "policy(s), and transform by projecting onto columns in K. Evaluating policy(s) with holdout labels "
                "inside the selection loop is selection leakage."
            ),
            why_it_matters=(
                "Reduces noise, collinearity, and width before fragile estimators.",
                "Univariate ranks miss interactions; model-based ranks inherit estimator bias.",
                "Tuning the subset on test invents optimistic metrics.",
            ),
            how_buildml_uses=(
                "Session.select_features(...) supports variance, univariate, and model-based strategies on train.",
                "Target/id/group/time/weight columns are preserved outside the feature keep-list.",
                "PreprocessResult exposes scores_, selected_features_, and dropped_features_.",
                "Permutation importance remains a diagnostic and does not mutate the schema.",
            ),
            interpretation_rules=(
                "Re-check validation metrics after selection before claiming improvement.",
                "Zero-variance drops are mechanical; univariate/model drops need a modeling rationale.",
                "If k or thresholds are searched, nest selection inside CV.",
            ),
            assumptions=(
                "Features are numeric and non-null on train after preparation.",
                "The score function matches the task family (classification versus regression).",
                "Protected roles are assigned so ids and targets are not dropped accidentally.",
            ),
            failure_modes=(
                "Selecting with full-data scores or test labels.",
                "Running model-based selection with an estimator family different from the final model without review.",
                "Empty keep-sets after aggressive thresholds.",
            ),
            anti_patterns=(
                "Using selection as a substitute for domain role assignment.",
                "Reporting selected features as causal drivers.",
                "Re-selecting after every peek at test error.",
            ),
            worked_example_pattern=(
                "Encode/impute on train; run variance filter; then univariate top-k.",
                "Fit the intended estimator; compare validation metrics against the full feature set.",
                "Only then evaluate once on test with the frozen keep-list.",
            ),
            related_concepts=("leakage-boundary", "feature-importance", "feature-schema"),
        ),
        _note(
            key="encoding-imputation-scaling",
            title="Imputation, encoding, and scaling",
            summary="These transforms change representation and must be train-fitted inside a modeling pipeline.",
            definition=(
                "Imputation, categorical encoding, and feature scaling are representation transforms: they "
                "fill missing values, map categories to numbers, and rewrite numeric units. In supervised "
                "pipelines they are estimators of their own—fit on train, applied frozen to other partitions."
            ),
            intuition=(
                "Before a model studies the data, you often translate the data into a language the model "
                "speaks: no blanks, numbers instead of labels, comparable units. The dictionary for that "
                "translation must be written from the training pages only."
            ),
            formal_idea=(
                "A preprocessing pipeline P_θ with θ = (impute params, vocabularies, scale params) is learned "
                "as θ̂ = fit(P, train) and reused as transform(P_θ̂, ·). Any step that estimates θ from "
                "evaluation rows crosses the leakage boundary. Order usually matters: impute → encode → scale."
            ),
            why_it_matters=(
                "Representation choices dominate many classical ML outcomes as much as the final estimator.",
                "Mixed policies between offline and serving cause silent score-time failures.",
                "Fair model comparison requires a frozen transform recipe across candidates.",
            ),
            how_buildml_uses=(
                "Session.impute / encode / scale implement train-fitted transforms after split.",
                "assert_can_fit and explain() reinforce ordering and leakage constraints.",
                "EDA Teaching Studio guides which columns need which transform family.",
                "Checkpoints and history record parameters so resume keeps the same recipe.",
            ),
            interpretation_rules=(
                "After the trio, verify nulls are gone (or intentionally allowed), dtypes are model-ready, and column sets match.",
                "Unknown categories and new missingness need declared policies at transform time.",
                "Choose transforms to match the estimator family you will compare, then freeze before test.",
                "If validation distributions look wildly different after transform, audit fit partition and outliers.",
            ),
            assumptions=(
                "Roles correctly mark which columns enter the transform set.",
                "Split membership exists before fitting reusable transform parameters.",
                "Score-time data can execute the same frozen recipe.",
            ),
            failure_modes=(
                "Fitting any of the three on full data before splitting.",
                "Scaling before imputing so nulls break or leak through inconsistently.",
                "Encoding paths that emit different column sets at score time.",
            ),
            anti_patterns=(
                "Copy-pasting ad-hoc pandas fills outside Session train-fit helpers.",
                "Retuning encodings after peeking at test errors.",
                "Applying heavy transforms for tree-only experiments without a reason.",
            ),
            worked_example_pattern=(
                "Split; impute train-fitted; encode train-fitted; scale train-fitted; then fit the estimator.",
                "Transform validation with the same frozen objects; confirm schema and null policies.",
                "Only after validation selection, apply once to test for the reported estimate.",
            ),
            related_concepts=("missing-data", "categorical-encoding", "feature-scaling", "leakage-boundary"),
        ),
        _note(
            key="text-features",
            title="Text feature extraction",
            summary="Count, hashing, and TF-IDF turn string fields into numeric bag-of-n-gram features with train-only fits.",
            definition=(
                "Text feature extraction maps free-text columns to numeric vectors using token counts, "
                "hashed token buckets, or TF-IDF weights learned from a training corpus."
            ),
            intuition=(
                "A short product review is not a usable float. Token features turn repeated words into "
                "columns a linear or tree model can consume—still ignoring most grammar."
            ),
            formal_idea=(
                "Given documents D_train, a vectorizer learns a mapping f: text → R^k. Score-time texts "
                "use the same f. Fitting f on all rows before splitting leaks vocabulary and IDF mass "
                "from holdout documents."
            ),
            why_it_matters=(
                "Classical tabular models need numeric inputs for text columns.",
                "Vocabulary and IDF statistics are fit-scope sensitive.",
                "Wide expansions change model capacity and interpretation.",
            ),
            how_buildml_uses=(
                "Session.text_features(...) fits count/TF-IDF/hashing vectorizers on train only.",
                "PreprocessRecipe(text=...) refits the same vectorizer families on each CV "
                "fold-train for selection-time honesty.",
                "Plans serialize into pipeline/checkpoint payloads for score-time replay.",
                "Missing text is treated as empty strings before vectorization.",
            ),
            interpretation_rules=(
                "Review max_features before claiming a metric gain from text expansion.",
                "Hashing collisions are irreversible; prefer TF-IDF when token names matter.",
                "High sparsity/width can dominate linear models without regularization.",
            ),
            assumptions=(
                "Token n-grams are a useful approximation for the prediction task.",
                "Train documents cover enough of the scoring vocabulary for the chosen width.",
            ),
            failure_modes=(
                "Fitting the vectorizer on all rows before splitting.",
                "Passing numeric columns as if they were text.",
                "Materializing unbounded vocabularies on large corpora without max_features.",
            ),
            anti_patterns=(
                "Tuning max_features on the test partition.",
                "Treating hashed feature indices as stable semantic tokens across runs with different widths.",
            ),
            worked_example_pattern=(
                "Split; call text_features on the review column with tfidf and a modest max_features.",
                "Fit a linear baseline; compare validation metrics against a no-text model.",
                "Inspect feature width and holdout errors before widening the vocabulary.",
            ),
            related_concepts=("feature-schema", "leakage-boundary", "feature-scaling"),
        ),
        _note(
            key="custom-transforms",
            title="Custom transforms",
            summary="Registered callables may extend preparation when they honor the train-fit contract and a stable output schema.",
            definition=(
                "A custom transform is a named pair of fit/transform callables registered with BuildML. "
                "Fit may inspect training rows only; transform applies the frozen artifact to any partition."
            ),
            intuition=(
                "When a domain clip, score, or lookup is not built in, register it once, fit it on train, "
                "and reuse the same artifact at score time—same discipline as a scaler."
            ),
            formal_idea=(
                "Registration stores fit: X_train → A and transform: (X, A) → X'. Leakage occurs if fit "
                "reads holdout rows or labels that should not define A. Persistence requires a picklable A "
                "and a still-registered transform name for replay."
            ),
            why_it_matters=(
                "Professionals need escape hatches without abandoning Session leakage rules.",
                "Unregistered or non-serializable artifacts break checkpoint/pipeline reload.",
                "Schema drift in transform outputs breaks estimator feature contracts.",
            ),
            how_buildml_uses=(
                "Session.register_transform / apply_custom_transform enforce train-only fit scope.",
                "Fitted CustomTransformPlan objects travel in plans.joblib when serializable.",
                "Score-time apply_preprocess_plans requires the name to remain registered.",
            ),
            interpretation_rules=(
                "Read the registered description and output column contract before trusting reload.",
                "Treat caller-owned fit logic as unaudited beyond partition scope checks.",
            ),
            assumptions=(
                "The registered fit function does not reach outside the provided train frame.",
                "Transform preserves row alignment and a stable column contract.",
            ),
            failure_modes=(
                "Fitting with holdout statistics inside the callable.",
                "Returning a differently indexed frame from transform.",
                "Forgetting to re-register before loading a pipeline in a new process.",
            ),
            anti_patterns=(
                "Closing over global mutable state that changes between fit and score.",
                "Marking serializable=True for unpicklable artifacts.",
            ),
            worked_example_pattern=(
                "Register a train-quantile clipper; apply_custom_transform on selected columns after split.",
                "Save a pipeline; restart; re-register the same name; score new rows.",
            ),
            related_concepts=("leakage-boundary", "feature-schema", "reproducibility"),
        ),
        _note(
            key="dry-run-plans",
            title="Dry-run plans",
            summary="Dry-run previews intended operations, prerequisites, and leakage notes without mutating Session state.",
            definition=(
                "A dry-run is a read-only preview that resolves catalog operations against current Session "
                "state and reports availability, blockers, estimated effects, and known leakage risks."
            ),
            intuition=(
                "Before you run a risky transform, ask what would happen: what is missing, what would "
                "change, and which leakage notes apply—without writing history or fitting anything."
            ),
            formal_idea=(
                "Dry-run maps an operation (or sequence) through the explanation catalog and workflow "
                "resolver. It does not execute side effects; availability ≠ appropriateness."
            ),
            why_it_matters=(
                "Large or irreversible steps benefit from an explicit preview.",
                "Blocked prerequisites are cheaper to fix before a failed fit.",
                "Teaching and review flows need a non-mutating inspection surface.",
            ),
            how_buildml_uses=(
                "Session.dry_run(...) returns a DryRunReport without appending history.",
                "Default previews focus on likely next transform/model/split/inspect steps.",
                "Unresolved risks from history heuristics are attached to the report.",
            ),
            interpretation_rules=(
                "Treat blocked reasons as API prerequisites, not domain advice.",
                "Re-run dry-run after state changes; stale previews are not cached as truth.",
            ),
            assumptions=(
                "The operation catalog and resolver accurately describe current public Session methods.",
            ),
            failure_modes=(
                "Interpreting available as recommended.",
                "Expecting dry-run to simulate numeric outputs of transforms.",
            ),
            anti_patterns=(
                "Using dry-run as a substitute for holdout evaluation.",
            ),
            worked_example_pattern=(
                "After ingest and roles, dry_run('split') then dry_run(['impute','scale','fit']).",
                "Resolve blockers, execute, then summarize_history for open risks.",
            ),
            related_concepts=("leakage-boundary", "reproducibility", "diagnostic-uncertainty"),
        ),
        _note(
            key="operation-history",
            title="Operation history",
            summary="Versioned Session history records calls and state transitions; it is not proof of source lineage.",
            definition=(
                "Operation history is an ordered, JSON-safe log of Session method calls with parameters, "
                "decision origin, warnings, and before/after workflow state."
            ),
            intuition=(
                "History is the session notebook: what ran, in what order, with which knobs—and which "
                "warnings fired. It cannot prove where a CSV originally came from."
            ),
            formal_idea=(
                "Each record is a schema-versioned event. summarize_history aggregates counts and surfaces "
                "heuristic unresolved risks; walkthrough joins the same log to resolver statuses."
            ),
            why_it_matters=(
                "Debugging and teaching need a reconstructable call sequence.",
                "Checkpoints persist history so mid-loop reentry keeps context.",
                "Warnings mentioning leakage or lineage deserve explicit review.",
            ),
            how_buildml_uses=(
                "Most state-changing Session methods append history records.",
                "Session.summarize_history() returns counts, recent ops, and unresolved risks.",
                "Explain/walkthrough surfaces read history without recursively logging themselves.",
            ),
            interpretation_rules=(
                "Absence of a warning is not absence of risk.",
                "Decision origin labels automatic vs explicit choices; review automatic ones.",
            ),
            assumptions=(
                "Callers use Session methods rather than silently mutating underlying frames.",
            ),
            failure_modes=(
                "External edits to data without recording a Session operation.",
                "Treating history as a compliance audit of data provenance.",
            ),
            anti_patterns=(
                "Clearing history to hide experimental dead-ends before handoff.",
            ),
            worked_example_pattern=(
                "Run a short prep+fit path; call summarize_history(); resolve listed risks before export.",
            ),
            related_concepts=("reproducibility", "leakage-boundary", "dry-run-plans"),
        ),
        _note(
            key="batch-leakage",
            title="Batch and loader leakage",
            summary="Train-only shuffling and train-fit batch transforms must not remix evaluation rows into learning.",
            definition=(
                "Batch leakage occurs when evaluation-partition rows influence training batches—through "
                "shared shuffling, oversampling, or statistics (normalize/augment) fit on more than train."
            ),
            intuition=(
                "If the DataLoader that updates weights can see test rows, or if batch normalize peeks at "
                "holdout values, the network practices on the exam."
            ),
            formal_idea=(
                "Let partitions be disjoint. A train DataLoader may shuffle within train only. Any transform "
                "parameters θ_batch = L(train) apply frozen to validation/test loaders."
            ),
            why_it_matters=(
                "Loader mistakes create optimistic Torch metrics that classical split discipline alone cannot catch.",
                "Normalize fit on all partitions is the neural analogue of scaling before train_test_split.",
            ),
            how_buildml_uses=(
                "Session.make_torch_loaders shuffles the train loader only.",
                "Optional standardize fits mean/std on train and freezes them on validation/test.",
                "Catalog leakage notes call out shuffle and normalize scope for Torch ops.",
            ),
            interpretation_rules=(
                "If shuffle was enabled on validation/test loaders, treat subsequent scores as contaminated.",
                "Empty holdout loaders are a data issue, not a reason to merge partitions.",
            ),
            assumptions=(
                "Split membership is defined before loader construction.",
                "Feature matrices are prepared without refitting on evaluation rows.",
            ),
            failure_modes=(
                "Concatenating partitions into one Dataset with a single shuffle flag.",
                "Global StandardScaler fit before building partition loaders.",
            ),
            anti_patterns=(
                "Building one shuffled DataLoader over the full table, then slicing batches by index later.",
            ),
            worked_example_pattern=(
                "Split → make_torch_loaders(shuffle_train=True) → assert validation/test loaders do not shuffle.",
                "Compare train-fit normalize versus full-table normalize on the same holdout.",
            ),
            related_concepts=("leakage-boundary", "evaluation-partitions", "data-splitting"),
        ),
        _note(
            key="early-stopping-partition",
            title="Early-stopping partition",
            summary="Stopping rules may read validation metrics; the test partition remains a final estimate only.",
            definition=(
                "Early stopping selects a training epoch using a monitor partition—almost always validation—"
                "so that test metrics stay out of the stopping decision."
            ),
            intuition=(
                "Validation tells you when to put the pencil down. If you watch the official test score to "
                "decide when to stop, the official score is no longer independent."
            ),
            formal_idea=(
                "Choose epoch t* = argmin_t M(validation_t). Report generalization with M(test_{t*}) only after "
                "t* is fixed. Using M(test) inside the argmin biases the reported score."
            ),
            why_it_matters=(
                "Neural nets overfit easily; stopping on test hides that overfitting.",
                "Teaching and model cards need the monitor partition named beside the selected epoch.",
            ),
            how_buildml_uses=(
                "fit_torch early_stopping_patience monitors validation (default monitor=val_loss).",
                "TrainResult.early_stop records triggered/best_epoch/reason and restore_best_weights.",
                "evaluate_torch defaults to partition='test' for final scoring after training choices freeze.",
                "Catalog anti-patterns warn against test-tuned stopping.",
            ),
            interpretation_rules=(
                "Read every curve with its partition tag.",
                "If stopping used test, treat the test metric as optimistic.",
            ),
            assumptions=(
                "A validation partition exists when early stopping will be enabled.",
                "Train/val/test membership stays fixed across the run.",
            ),
            failure_modes=(
                "Selecting the best test epoch after the fact and reporting that test score.",
                "Retuning patience repeatedly against the same test split.",
            ),
            anti_patterns=(
                "Using test loss as the early-stopping monitor.",
            ),
            worked_example_pattern=(
                "fit_torch(..., early_stopping_patience=3) → read early_stop.reason → "
                "evaluate_torch(partition='test') once.",
            ),
            related_concepts=("evaluation-partitions", "leakage-boundary", "batch-leakage", "training-curves"),
        ),
        _note(
            key="training-curves",
            title="Training curves",
            summary="Epoch loss/metric trajectories need device, monitor partition, and honesty limits beside the plot.",
            definition=(
                "A training curve is the time series of train (and optional validation) losses or "
                "metrics across epochs, optionally with learning-rate steps from a scheduler."
            ),
            intuition=(
                "Curves show whether the network is still learning, plateauing, or memorizing. "
                "Without naming the validation monitor and device, the picture is incomplete."
            ),
            formal_idea=(
                "For epochs t=1..T, record L_train(t) and optionally L_val(t). Early stopping "
                "selects t* from L_val. Claims about generalization require a separate held-out "
                "estimate after t* is fixed."
            ),
            why_it_matters=(
                "Batch losses are noisy; epoch aggregates are the teaching default.",
                "Validation improvement is not a test result; curves alone do not prove deployment risk.",
            ),
            how_buildml_uses=(
                "TrainResult.history and TrainingCurveReport store epoch series plus disclosures.",
                "Session.torch_training_curve() and walkthrough torch_training_status surface limits.",
                "Teaching Studio cockpit discloses early-stop partition and resolved device when a trainer exists.",
            ),
            interpretation_rules=(
                "Prefer epoch aggregates over batch spikes when comparing runs.",
                "If train falls while validation rises, treat later epochs as overfitting risk.",
                "Read early_stop.partition before quoting a selected epoch.",
            ),
            assumptions=(
                "History was logged under a fixed split and feature contract.",
                "Scheduler and clipping settings are part of the run identity.",
            ),
            failure_modes=(
                "Comparing curves from different devices or normalize contracts without disclosure.",
                "Reading resume-appended history as a single uninterrupted LR schedule when the scheduler changed.",
            ),
            anti_patterns=(
                "Publishing a loss plot without stating validation vs test scope.",
            ),
            worked_example_pattern=(
                "fit_torch → torch_training_curve → read disclosures → evaluate_torch(partition='test').",
            ),
            related_concepts=("early-stopping-partition", "evaluation-partitions", "batch-leakage"),
        ),
        _note(
            key="rag-eval-contamination",
            title="RAG eval contamination",
            summary="Evaluation answers must stay out of the indexed corpus or retrieval metrics become circular.",
            definition=(
                "Eval contamination in retrieval is indexing documents that contain labeled answers or "
                "passages used only for evaluation queries, so the index can return the answer by identity "
                "rather than by generalization over held-out text."
            ),
            intuition=(
                "If the answer sheet is in the library, finding it is not evidence that search works on "
                "new questions. Mark eval-only documents and keep them out of the index build."
            ),
            formal_idea=(
                "Let C_index be the indexed corpus and Q_eval the evaluation query set with relevance "
                "labels over documents D_eval. Require C_index ∩ D_eval = ∅ for any claim that retrieval "
                "metrics estimate performance on unseen answers."
            ),
            why_it_matters=(
                "Contaminated indexes inflate recall@k and MRR without improving real retrieval.",
                "Teaching and model cards need an explicit corpus vs eval-query disclosure.",
            ),
            how_buildml_uses=(
                "Documents may carry role=index or role=eval_only.",
                "rag_embed_and_index raises LeakageError when any eval_only document is present.",
                "Catalog leakage fields warn against indexing labeled eval answers.",
            ),
            interpretation_rules=(
                "Read which corpus was indexed beside every recall@k / MRR number.",
                "If eval answers were indexed, treat metrics as invalid for generalization claims.",
            ),
            assumptions=(
                "Callers label eval-only material before indexing.",
                "Qrels refer to document ids that exist in the index corpus when measuring hits.",
            ),
            failure_modes=(
                "Silently concatenating a FAQ answer key into the index folder.",
                "Reusing the same docs for indexing and as the only relevant targets without disclosure.",
            ),
            anti_patterns=(
                "Indexing eval_only documents to 'make the demo look good'.",
            ),
            worked_example_pattern=(
                "rag_ingest_corpus(index docs) → rag_embed_and_index → rag_evaluate(qrels on held-out queries).",
            ),
            related_concepts=("leakage-boundary", "rag-chunk-index-boundary", "evaluation-partitions"),
        ),
        _note(
            key="rag-chunk-index-boundary",
            title="RAG chunk and index boundary",
            summary="Chunking and indexing are retrieval prep steps; they are not classical fit and not a Session checkpoint.",
            definition=(
                "The chunk/index boundary is the contract that documents are split into chunks, embedded, "
                "and stored in a vector index artifact separate from Session workflow checkpoints and "
                "Torch trainer bundles."
            ),
            intuition=(
                "Think of the index as a searchable card catalog built from the books you chose to shelve. "
                "Saving your lab notebook (Session checkpoint) does not shelve the books for you."
            ),
            formal_idea=(
                "A RAG bundle records chunk config, embedder id/dim, chunk metadata, and embeddings under "
                "schema buildml.rag_bundle.v1. Session checkpoints omit that payload; Torch bundles are "
                "orthogonal supervised-training artifacts."
            ),
            why_it_matters=(
                "Mixing artifact kinds causes failed loads and false resume expectations.",
                "Chunk size/overlap change the retrieval unit; ids must stay deterministic for audits.",
            ),
            how_buildml_uses=(
                "Session.rag_chunk / rag_embed_and_index build an in-memory RagIndex.",
                "save_rag_bundle / load_rag_bundle round-trip buildml.rag_bundle.v1.",
                "Wrong schema ids raise ValidationError with an explicit expected format.",
            ),
            interpretation_rules=(
                "Never imply checkpoint_load restored a vector index.",
                "State embedder id and dimension beside retrieve/eval results.",
            ),
            assumptions=(
                "Index corpus membership is fixed for a given bundle.",
                "Query embedding uses a compatible embedder after load.",
            ),
            failure_modes=(
                "Passing a Session checkpoint path to load_rag_bundle.",
                "Changing chunk config between index build and eval without rebuilding.",
            ),
            anti_patterns=(
                "Embedding the vector index inside a Session checkpoint.",
            ),
            worked_example_pattern=(
                "rag_ingest_corpus → rag_chunk → rag_embed_and_index → save_rag_bundle.",
            ),
            related_concepts=("rag-eval-contamination", "reproducibility", "leakage-boundary"),
        ),
        _note(
            key="rag-retrieval-metrics",
            title="RAG retrieval metrics",
            summary=(
                "recall@k, MRR, nDCG@k, and hit-rate@k measure ranking quality against gold labels; "
                "they are not classification accuracy."
            ),
            definition=(
                "Retrieval metrics score whether relevant documents or chunks appear in the top-k ranked "
                "hits for each evaluation query, using gold relevance labels (qrels)."
            ),
            intuition=(
                "Ask whether the right book showed up near the top of the search results—not whether a "
                "classifier predicted a class label."
            ),
            formal_idea=(
                "For query q with relevant set R_q, recall@k = |{ids in top-k} ∩ R_q| / |R_q|. "
                "MRR averages 1/rank of the first relevant hit. nDCG@k discounts later ranks; "
                "hit-rate@k is the fraction of queries with at least one relevant hit."
            ),
            why_it_matters=(
                "A single unlabeled 'accuracy' hides k, relevance mode, and corpus identity.",
                "Document-level vs chunk-level relevance change what a hit means.",
            ),
            how_buildml_uses=(
                "rag_evaluate supports relevance_mode=document (default) or chunk, plus retrieve mode overrides.",
                "RagEvalResult exposes recall_at_k, mrr, ndcg_at_k, hit_rate_at_k, and disclosures.",
                "compare_retrieval_configs rebuilds indexes per config row for side-by-side metrics.",
            ),
            interpretation_rules=(
                "Always read metrics with k, relevance_mode, and retrieve_mode.",
                "Do not call recall@k 'accuracy'.",
            ),
            assumptions=(
                "Qrels ids match the claimed relevance_mode (doc_id or chunk_id).",
                "The same embedder/index pair is used for every eval query in the run.",
            ),
            failure_modes=(
                "Comparing recall@5 from one embedder to recall@20 from another without disclosure.",
                "Treating hashing-embedder demos as semantic retrieval quality claims.",
            ),
            anti_patterns=(
                "Reporting only top-1 hit rate as proof the RAG system is production-ready.",
            ),
            worked_example_pattern=(
                "rag_embed_and_index → rag_evaluate(qrels, k=5) → read recall_at_k, mrr, and ndcg_at_k.",
            ),
            related_concepts=("rag-eval-contamination", "rag-chunk-index-boundary", "evaluation-partitions"),
        ),
        _note(
            key="ai-egress-privacy",
            title="AI Egress Privacy",
            summary=(
                "User-controlled data egress before any information leaves the machine to an external LLM provider."
            ),
            definition=(
                "Egress privacy is the set of controls that determine what data (schema, statistics, samples, raw rows) "
                "leaves the user's machine when calling an external LLM API. BuildML provides four egress levels: "
                "SCHEMA_ONLY (column names/types), STATS_ONLY (aggregates), REDACTED_SAMPLE (masked rows), and "
                "FULL_SAMPLE (raw rows with explicit opt-in)."
            ),
            intuition=(
                "Think of egress levels as airport security zones. SCHEMA_ONLY shows only the boarding pass (column names). "
                "STATS_ONLY adds aggregate flight statistics without passenger details. REDACTED_SAMPLE masks sensitive "
                "passenger info. FULL_SAMPLE shares everything—use only when necessary and after inspection."
            ),
            formal_idea=(
                "The egress manifest is a typed record of what will be (or was) sent: columns, row count, estimated tokens, "
                "and warnings about PII-like columns. ai_egress_preview returns the manifest without making an API call."
            ),
            why_it_matters=(
                "External LLM providers see whatever payload the user approves.",
                "Sensitive column names, statistics, or raw values may leak if egress is not controlled.",
                "Regulatory and security requirements often restrict what data can leave internal systems.",
            ),
            how_buildml_uses=(
                "ai_configure sets the default egress level (STATS_ONLY by default).",
                "ai_egress_preview shows the manifest before any call.",
                "ai_dry_run returns the full prompt payload for local inspection.",
                "Column allow/deny lists filter what columns appear in egress.",
                "Transcripts record egress manifests, not raw data (unless FULL_SAMPLE).",
            ),
            interpretation_rules=(
                "STATS_ONLY sends aggregates and schema, never raw row values.",
                "FULL_SAMPLE requires explicit opt-in and sends raw data.",
                "Always inspect ai_egress_preview before first AI call on sensitive data.",
            ),
            assumptions=(
                "The user reviews egress manifests before approving data-heavy calls.",
                "Column allow/deny lists are configured appropriately for the sensitivity level.",
            ),
            failure_modes=(
                "Using FULL_SAMPLE without reviewing what data leaves the machine.",
                "Column names themselves may be sensitive even in SCHEMA_ONLY mode.",
            ),
            anti_patterns=(
                "Auto-approving FULL_SAMPLE egress without inspection.",
                "Assuming STATS_ONLY protects against all data leakage (aggregates can still reveal patterns).",
            ),
            worked_example_pattern=(
                "ai_configure → ai_egress_preview → review manifest → ai_advisor (if manifest acceptable).",
            ),
            related_concepts=("leakage-boundary", "ai-tool-trust"),
        ),
        _note(
            key="ai-tool-trust",
            title="AI Tool Trust",
            summary=(
                "Tools are allowlisted, mapped to Session methods, and gated by a propose-confirm-execute flow."
            ),
            definition=(
                "The AI operator can only execute tools that are explicitly registered in the ToolRegistry. Each tool "
                "maps to a Session method, has a confirmation policy (auto, confirm, always_confirm), and cannot bypass "
                "existing Session guards (leakage, validation). Destructive operations always require confirmation."
            ),
            intuition=(
                "The tool registry is like a hotel safe with an approved guest list. The AI can suggest using items "
                "from the safe, but can only access what's on the list, and certain items require the owner's signature "
                "before they can be taken out."
            ),
            formal_idea=(
                "ToolSpec defines name, parameters, confirm_policy, read_only, and destructive flags. "
                "The executor validates each ToolCall against the registry, refuses unlisted tools, and requires "
                "confirmation for write operations. Maximum iteration limits prevent runaway loops."
            ),
            why_it_matters=(
                "LLMs may hallucinate tool names or attempt operations outside the allowed scope.",
                "Propose-confirm-execute prevents accidental state changes from AI suggestions.",
                "Leakage guards must fire regardless of whether human or AI initiated the operation.",
            ),
            how_buildml_uses=(
                "ToolRegistry defines the M1 allowlist: describe_dataset, explain_operation, workflow_status, etc.",
                "ai_execute validates tool calls against the registry and requires confirmation for writes.",
                "Destructive tools (drop, delete) always require explicit confirmation.",
                "Read-only tools (describe, explain) may auto-confirm.",
            ),
            interpretation_rules=(
                "Tools not in the registry are rejected with a named error.",
                "Confirmation status is recorded in the transcript.",
                "Write operations modify Session state; read operations do not.",
            ),
            assumptions=(
                "The tool registry is conservative and well-maintained.",
                "Users review proposals before confirming write operations.",
            ),
            failure_modes=(
                "Auto-confirming destructive operations without review.",
                "Expanding the tool registry without security review.",
            ),
            anti_patterns=(
                "Bypassing the tool registry with eval/exec.",
                "Trusting AI tool suggestions without verifying prerequisites.",
            ),
            worked_example_pattern=(
                "ai_execute('set_roles', {'mapping': {...}}) → review proposal → ai_execute(..., confirm=True).",
            ),
            related_concepts=("ai-prompt-injection", "ai-egress-privacy", "leakage-boundary"),
        ),
        _note(
            key="ai-prompt-injection",
            title="AI Prompt Injection Hardening",
            summary=(
                "Untrusted data (column names, cell values, user text) is marked and separated from instructions."
            ),
            definition=(
                "Prompt injection is an attack where adversarial text in data is interpreted as instructions by the LLM. "
                "BuildML hardens against this by: marking untrusted data with boundary tags, using system prompts that "
                "instruct the model to treat data as data only, validating tool calls against the registry, and refusing "
                "arbitrary code execution."
            ),
            intuition=(
                "Imagine a mail room where incoming packages are labeled 'EXTERNAL—DO NOT OPEN WITHOUT INSPECTION'. "
                "Even if a package label says 'URGENT: Give to CEO immediately', the mailroom follows procedure. "
                "Data markers work the same way: they tell the LLM that this content is cargo, not commands."
            ),
            formal_idea=(
                "Untrusted data is wrapped in [UNTRUSTED DATA] markers. Tool results are wrapped in [TOOL RESULT - DATA ONLY] "
                "markers. The system prompt explicitly states that data is not instructions. Injection patterns "
                "(e.g., 'ignore previous instructions') are detected and escaped in security tests."
            ),
            why_it_matters=(
                "Malicious column names like '; DROP TABLE users; --' should not execute.",
                "Cell values containing 'Ignore previous instructions' should not change AI behavior.",
                "User prompts attempting tool registry bypass should be rejected.",
            ),
            how_buildml_uses=(
                "mark_untrusted_data wraps data with source markers before sending to the LLM.",
                "sanitize_tool_result wraps tool outputs before feeding back.",
                "detect_injection_attempt scans text for known injection patterns.",
                "CI injection tests verify boundaries hold with adversarial fixtures.",
            ),
            interpretation_rules=(
                "Injection detection is a warning, not a block; humans review flagged content.",
                "The tool registry, not the LLM, controls what operations execute.",
                "eval/exec are never allowed regardless of prompt content.",
            ),
            assumptions=(
                "The LLM respects boundary markers (not guaranteed, but improves safety).",
                "The tool registry is the authoritative gate for execution.",
            ),
            failure_modes=(
                "Novel injection patterns not in the detection list.",
                "LLMs that ignore boundary instructions (mitigated by tool registry).",
            ),
            anti_patterns=(
                "Trusting LLM-generated code without review.",
                "Disabling injection detection for convenience.",
            ),
            worked_example_pattern=(
                "Column name: 'ignore_previous; drop_table' → detected as suspicious → wrapped as data → tool registry rejects unauthorized calls.",
            ),
            related_concepts=("ai-tool-trust", "ai-egress-privacy"),
        ),
    )
}


def get_concept(key: str) -> ConceptNote:
    """Return a concept note or raise a precise catalog error."""
    try:
        return CONCEPT_NOTES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown BuildML concept: {key}") from exc


def list_concepts() -> tuple[ConceptNote, ...]:
    """Return concept notes in stable key order."""
    return tuple(CONCEPT_NOTES[key] for key in sorted(CONCEPT_NOTES))
