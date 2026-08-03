# ruff: noqa: E501
"""Beginner layers for the classical tabular foundations.

These are the concepts a newcomer meets first, so they are written for someone
who has never split a dataset before. Every other domain assumes them.
"""

from __future__ import annotations

from buildml.explain.beginner._builder import (
    ADVANCED,
    CORE,
    FOUNDATION,
    BeginnerLayer,
    _index,
    _layer,
)

CLASSICAL_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "column-roles",
        plain=(
            "A role is a label you attach to each column saying how BuildML is allowed to use it: "
            "the thing you are predicting (target), the clues you may use (features), the "
            "customer ID you must never learn from, the date, the group key, and so on. "
            "The computer can see that a column holds numbers; it cannot see what those numbers mean."
        ),
        analogy=(
            "Think of assigning seats before a meeting. Everyone in the room is a person, but the "
            "note-taker, the decision-maker, and the observer play completely different parts. "
            "Roles are the seating plan for your columns."
        ),
        steps=(
            "Look at each column and ask: at the moment I would make this prediction in real life, would I actually know this value?",
            "Pick exactly one column as the target — the answer you want the model to produce.",
            "Mark the columns you genuinely have at prediction time as features.",
            "Mark row identifiers (customer_id, order_id) as `id` and anything you must not learn from as `ignore`.",
            "Mark special columns: `time` for the timestamp, `group` for the entity rows repeat over, `weight` for row importance.",
            "Call `set_roles(...)` so BuildML enforces the plan for every later operation.",
        ),
        use=(
            "Always, before any operation that needs to know which column is the answer — fitting, evaluating, encoding, or target-aware EDA.",
            "Whenever you load a new dataset or change its columns, because roles do not survive a schema change automatically.",
        ),
        avoid=(
            "Do not lean on roles to decide *meaning* — BuildML will happily accept a role you assigned wrongly.",
            "Do not mark a column as a feature just because it is numeric and correlates well; that is exactly how leakage starts.",
        ),
        myths=(
            (
                "The data type tells BuildML what the column is for.",
                "An integer column can be a measurement, a category code, a customer ID, or a date in disguise. Only you know which, which is why roles exist.",
            ),
            (
                "Any column present in the file is fair game as a feature.",
                "A column recorded *after* the outcome (like `refund_processed`) is unavailable when you actually need a prediction. It must be `ignore`.",
            ),
        ),
        example=(
            "session = Session.ingest(frame)",
            "session.set_roles({",
            "    'churned': 'target',      # the answer we want",
            "    'customer_id': 'id',      # never learn from an ID",
            "    'signup_date': 'time',    # ordering, not a feature",
            "    'refund_issued': 'ignore' # only known after churn happens",
            "})",
        ),
        check=(
            "For every feature you kept: could you fill in its value one second before the outcome happens?",
            "Which of your columns would be constant, missing, or unknown in production?",
        ),
        tools=("set_roles", "metadata", "eda", "assert_can_fit"),
        terms=("role", "target", "feature", "leakage"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "leakage-boundary",
        plain=(
            "Leakage is when your model learns from information it would not have in real life. "
            "The classic version is subtle: you compute an average, a category list, or a scaling "
            "factor using *all* your rows, including the ones you later use as an exam. The model "
            "has effectively peeked at the answers, so your score looks great and reality does not match."
        ),
        analogy=(
            "It is studying for an exam using the actual exam paper. You will score brilliantly on that "
            "paper and learn nothing about how you would do on a fresh one."
        ),
        steps=(
            "Split your rows first, before you compute anything at all from the data.",
            "Learn every number you need — medians for filling blanks, category lists, scaling factors, model weights — using training rows only.",
            "Freeze those numbers into a stored plan.",
            "Apply the frozen plan to validation and test rows without recomputing anything.",
            "When a score looks suspiciously good, retrace this list before celebrating.",
        ),
        use=(
            "On every project. This is not an advanced technique; it is the difference between a real score and a fiction.",
            "Especially when a column could have been recorded after, or because of, the outcome.",
        ),
        avoid=(
            "There is no situation where you want leakage. What varies is the *boundary*: for forecasting it is time, for repeat customers it is the customer, not the row.",
            "Do not treat BuildML's enforcement as proof of safety — it can stop you refitting on test, but it cannot know that `discount_after_complaint` is a leaked column.",
        ),
        myths=(
            (
                "Leakage means accidentally including the target column as a feature.",
                "That is only the most obvious case. Scaling with all rows, choosing features by looking at test scores, or imputing with a global mean all leak too.",
            ),
            (
                "A very high score means the model is very good.",
                "A very high score is more often the first symptom of leakage. Unexpectedly good results deserve suspicion, not a release.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)   # boundary first",
            "session.impute(strategy='median')              # medians from train only",
            "session.scale()                                # scaler fitted on train only",
            "session.fit(LogisticRegression())",
            "session.evaluate(partition='test')             # untouched rows",
        ),
        check=(
            "Name every number in your pipeline that was computed from data. Which partition did each one come from?",
            "If a colleague handed you this score, what is the first thing you would ask to verify?",
        ),
        tools=("split", "impute", "encode", "scale", "fit", "dry_run"),
        terms=("leakage", "train", "test", "plan"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "data-splitting",
        plain=(
            "Splitting means dividing your rows into groups that serve different jobs: rows the model "
            "learns from, rows you use to compare options, and rows you keep sealed until the very end. "
            "Every row gets exactly one membership, and that membership decides what it is allowed to influence."
        ),
        analogy=(
            "A driving instructor teaches you on practice roads (train), rehearses you on a mock route "
            "(validation), and the examiner uses a route nobody rehearsed (test). Rehearsing on the "
            "examiner's route makes the licence meaningless."
        ),
        steps=(
            "Decide what a single independent unit really is: one row? one customer? one week?",
            "Choose the split style that respects that unit — random for independent rows, `group_split` for repeated entities, `time_split` when the future must stay in the future.",
            "For classification with a rare class, use stratification so each partition keeps the same class mix.",
            "Run the split before any transform, and record the seed so the same rows land in the same places next time.",
            "If your boundary is too subtle for a built-in rule, design it yourself and hand it over with `inject_split`.",
        ),
        use=(
            "Before every modeling operation. BuildML blocks `fit`, `impute`, `encode`, `scale`, and `resample` until a split exists, on purpose.",
            "Again from scratch whenever you change what a row represents, such as after aggregating to one row per customer.",
        ),
        avoid=(
            "Do not use a plain random split when the same person, household, device, or store appears in many rows — near-duplicates will straddle the boundary and inflate your score.",
            "Do not use a random split when you are predicting the future; sort by time instead.",
        ),
        myths=(
            (
                "80/20 is the correct split.",
                "It is a convention, not a rule. What matters is whether the evaluation partition has enough rows — especially enough *positive* rows — to give a stable number.",
            ),
            (
                "Any split is fine as long as the model never sees the test rows.",
                "A random split across repeated customers technically hides the test rows, yet still leaks, because a near-copy of each test row sits in training.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)",
            "# repeated customers instead:",
            "session.group_split(group_column='customer_id', test_size=0.2, random_state=0)",
            "# forecasting instead:",
            "session.time_split(time_column='order_date', test_size=0.2)",
        ),
        check=(
            "If the same customer appears twice, can those two rows end up on opposite sides of your split?",
            "How many positive-class rows are in your test partition? Is that enough to trust a percentage?",
        ),
        tools=("split", "group_split", "time_split", "inject_split", "partition"),
        terms=("split", "stratified", "group split", "time split"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "evaluation-partitions",
        plain=(
            "Validation and test look the same but have different jobs. Validation is the partition you "
            "are allowed to look at repeatedly while you decide things — which model, which settings, "
            "which cutoff. Test is looked at once, after every decision is locked, to estimate how the "
            "chosen system will actually behave."
        ),
        analogy=(
            "Validation is a dress rehearsal you can run as many times as you like. Test is opening night. "
            "Running opening night twenty times and picking the best performance is not a performance review."
        ),
        steps=(
            "Do all comparing, tuning, threshold picking, and feature deciding against validation.",
            "Write down the final configuration and stop changing it.",
            "Score test exactly once and report that number.",
            "If the test result makes you change something, be honest: test has become another validation set and you need fresh held-out data for the real claim.",
        ),
        use=(
            "Whenever you will make more than one modeling decision — which is essentially always.",
            "When a stakeholder needs a number that stands for future performance rather than for your search process.",
        ),
        avoid=(
            "Do not use test to choose a threshold, a model, or a feature set. Each peek quietly turns it into training data.",
            "Do not skip validation and tune against cross-validation *and* test at the same time without tracking how many looks you have had.",
        ),
        myths=(
            (
                "Validation and test are interchangeable names for held-out data.",
                "They differ in how many times you are allowed to read them. That budget is the whole point.",
            ),
            (
                "Checking test a few times is harmless.",
                "Every check leaks a little selection information. After enough checks, the test score measures your search, not your model.",
            ),
        ),
        example=(
            "ranked = session.compare_models(candidates, partition='validation')",
            "session.tune_threshold(partition='validation')",
            "final = session.evaluate(partition='test')   # once, at the end",
        ),
        check=(
            "How many times have you looked at a test-partition number on this project?",
            "Which decisions in your notebook were influenced by a test score?",
        ),
        tools=("evaluate", "compare_models", "tune_threshold", "cv_score"),
        terms=("validation", "test", "holdout", "metric"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "feature-schema",
        plain=(
            "The schema is the contract between training and scoring: the same column names, in the same "
            "meaning, with the same encoding. If training saw a column called `plan_tier_gold` and scoring "
            "produces `tier_gold`, the model is reading gibberish even though nothing crashed."
        ),
        analogy=(
            "It is a plug and a socket. The pins have to be in the same places. A plug that *almost* fits "
            "either does not go in or quietly connects the wrong wires."
        ),
        steps=(
            "After every transform that adds or removes columns, look at the resulting column list.",
            "Note which columns were generated (one-hot expansions, date parts, bins) — those are now part of the contract.",
            "Save the preprocessing plans together with the model so scoring rebuilds the identical columns.",
            "At scoring time, apply the stored plans rather than re-deriving the transforms by hand.",
            "Decide in advance what should happen when a brand-new category shows up in production.",
        ),
        use=(
            "Before saving a model you intend to reuse, and every time you load one.",
            "Any time you add a transform mid-project and are unsure whether earlier artifacts still match.",
        ),
        avoid=(
            "Do not rely on column *order* as the contract; rely on names and stored plans.",
            "Do not hand-write the scoring transforms in a separate script — that is where the two paths drift apart.",
        ),
        myths=(
            (
                "If prediction runs without an error, the schema matched.",
                "Many estimators accept any numeric matrix of the right width. Silent misalignment is the dangerous failure, not a crash.",
            ),
            (
                "One-hot encoding produces the same columns every time.",
                "It produces one column per category *observed in training*. New or missing categories change the width unless the plan is stored and replayed.",
            ),
        ),
        example=(
            "session.save_pipeline('artifacts/pipeline')",
            "scoring = Session.ingest(new_frame).load_pipeline('artifacts/pipeline')",
            "scoring.apply_preprocess_plans()   # rebuilds the exact training columns",
            "predictions = scoring.predict_from_pipeline()",
        ),
        check=(
            "How many columns does your estimator expect, and can you list them from the saved artifact?",
            "What happens to your pipeline when production sends a category that never appeared in training?",
        ),
        tools=("save_pipeline", "load_pipeline", "apply_preprocess_plans", "prepare_design_matrix"),
        terms=("schema", "one-hot encoding", "pipeline", "plan"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "missing-data",
        plain=(
            "Imputation fills blank cells with a stand-in value — often the median or the most common "
            "category — computed from the training rows. It makes the table usable by estimators that "
            "refuse blanks. It does not recover the missing information, and it never tells you *why* "
            "the value was missing."
        ),
        analogy=(
            "A form arrives with the income box empty. Writing in the typical income lets you process the "
            "form, but it is a guess. And if the box is blank precisely because the person had no income, "
            "your guess is systematically wrong in one direction."
        ),
        steps=(
            "Count the blanks per column and ask what a blank means in that specific column.",
            "Separate 'not measured' from 'not applicable' from 'genuinely zero' — they need different treatments.",
            "Choose a strategy: median for skewed numbers, mean for symmetric ones, most-frequent or a literal 'Missing' category for text.",
            "Consider adding a was-missing indicator column when the missingness itself might be informative.",
            "Fit the strategy on train only, then apply the same frozen values to validation and test.",
        ),
        use=(
            "Whenever the estimator you chose cannot handle blanks — which includes most linear models, SVMs, and distance-based methods.",
            "When blanks are scattered and few, and dropping the rows would waste useful data.",
        ),
        avoid=(
            "Skip it when your estimator handles missing values natively and does so meaningfully — HistGradientBoosting learns a direction for blanks, which is often better than a guess.",
            "Do not impute a column that is 80% blank and expect the result to mean anything; consider dropping it or turning it into a yes/no 'was present' flag.",
        ),
        myths=(
            (
                "Filling blanks fixes the missing-data problem.",
                "It only removes the error message. If values are missing for a reason connected to the target, imputation bakes that bias in more quietly.",
            ),
            (
                "The mean is the safe default.",
                "The mean is dragged around by outliers and skew. The median is usually the safer automatic choice for numbers.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.impute(strategy='median')                       # numeric columns",
            "session.impute(columns=['region'], strategy='most_frequent')",
            "# the learned fill values are stored and replayed on test",
        ),
        check=(
            "For your highest-missingness column, what does a blank actually mean in the source system?",
            "Would a model trained on the was-missing indicator alone predict the target better than chance?",
        ),
        tools=("impute", "eda", "metadata"),
        terms=("missing value", "imputation", "plan"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "categorical-encoding",
        plain=(
            "Most estimators only read numbers, so text categories such as `region` or `plan_tier` have to "
            "be converted. Encoding is that conversion. The important part is choosing a conversion that "
            "does not invent facts — turning `red`, `green`, `blue` into 1, 2, 3 tells the model that green "
            "sits between red and blue, which is nonsense."
        ),
        analogy=(
            "Translating a menu into another language. One-hot encoding is giving each dish its own line. "
            "Ordinal encoding is numbering the dishes 1-2-3 — fine for a tasting sequence, absurd for flavours."
        ),
        steps=(
            "List your category columns and count how many distinct values each one has.",
            "For unordered categories with a manageable number of values, use one-hot: one yes/no column per category.",
            "For genuinely ordered categories (`small` < `medium` < `large`), ordinal encoding preserves real information.",
            "For very high-cardinality columns like postcodes, prefer grouping, hashing, or target encoding over thousands of one-hot columns.",
            "Fit the category vocabulary on train only, and decide what an unseen category does at scoring time.",
        ),
        use=(
            "Before fitting any estimator that cannot consume strings — which is nearly all of scikit-learn.",
            "When you want a linear model to treat each category independently rather than as a magnitude.",
        ),
        avoid=(
            "Skip one-hot for a column with thousands of distinct values; you will create thousands of nearly-empty columns and slow everything down.",
            "Do not use ordinal encoding on unordered categories with a linear or distance-based model — the fake ordering becomes a real (wrong) signal.",
        ),
        myths=(
            (
                "Encoding is a lossless formatting step.",
                "Ordinal encoding injects an ordering that may not exist. Target encoding injects information about the answer. Both change what the model can learn.",
            ),
            (
                "Trees do not care about encoding.",
                "Trees tolerate ordinal codes better than linear models, but the split points still depend on the arbitrary numbering, which affects both accuracy and interpretability.",
            ),
        ),
        example=(
            "session.encode(strategy='onehot', columns=['region', 'channel'])",
            "session.encode(strategy='ordinal', columns=['size'])   # small < medium < large",
            "# new columns appear: region_north, region_south, ...",
            "print(session.metadata()['n_columns'])",
        ),
        check=(
            "Which of your category columns have a real order, and which only look like they might?",
            "How many columns will one-hot add in total, and is your row count large enough to support them?",
        ),
        tools=("encode", "text_features", "eda"),
        terms=("encoding", "one-hot encoding", "ordinal encoding", "cardinality"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "feature-scaling",
        plain=(
            "Scaling rewrites numeric columns so they share a comparable range — for example subtracting "
            "the mean and dividing by the spread. It adds zero new information. It exists because some "
            "algorithms measure distances or penalize coefficients, and those algorithms would otherwise "
            "let a column measured in millions drown out one measured in units."
        ),
        analogy=(
            "Comparing runners where one time is in seconds and another in hours. Converting both to the "
            "same unit does not make anyone faster; it just stops the arithmetic from being meaningless."
        ),
        steps=(
            "Check whether your estimator cares: distance-based (kNN, SVM, k-means), gradient-based (neural nets), and regularized linear models do; plain trees do not.",
            "Pick a scaler — standard (mean 0, spread 1) for roughly symmetric data, robust (median and quartiles) when outliers are present, min-max when you need a bounded range.",
            "Fit it on the training rows so it learns training's centre and spread.",
            "Apply the frozen scaler to validation and test.",
            "Remember the model's coefficients are now in scaled units when you interpret them.",
        ),
        use=(
            "Before kNN, SVM, k-means, PCA, and any neural network.",
            "Before ridge or lasso, since the penalty size depends directly on the units of each coefficient.",
        ),
        avoid=(
            "Skip it for decision trees, random forests, and gradient boosting — they split on order, and order does not change under scaling.",
            "Do not scale one-hot columns or identifiers by reflex; scaling a 0/1 column rarely helps and makes the output harder to read.",
        ),
        myths=(
            (
                "Scaling improves model quality.",
                "It changes representation, not information. For a tree the score should be essentially unchanged; for kNN it can be the difference between working and not.",
            ),
            (
                "Scaling makes data normally distributed.",
                "Standardization moves and stretches the distribution. A skewed column stays exactly as skewed afterwards.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.scale(strategy='standard')      # mean/std learned on train",
            "session.fit(KNeighborsClassifier(n_neighbors=5))",
            "# use strategy='robust' when a few extreme values dominate",
        ),
        check=(
            "Does your chosen estimator use distances, gradients, or a coefficient penalty?",
            "If you rescale one column by 1000, would your model's predictions change?",
        ),
        tools=("scale", "fit", "reduce_dimensions"),
        terms=("scaling", "regularization", "distribution"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "class-imbalance",
        plain=(
            "Imbalance means one outcome is much rarer than the other — 2% fraud, 1% churn, 0.1% defects. "
            "It breaks the habits that work on balanced data: accuracy becomes meaningless, the default "
            "0.5 cutoff predicts 'no' for everything, and a model can look excellent while catching nothing."
        ),
        analogy=(
            "A smoke alarm that never goes off is right 99.9% of the time. Nobody would call it a good "
            "smoke alarm, because its accuracy is measured against the wrong question."
        ),
        steps=(
            "Measure the actual prevalence: what fraction of rows are positive, and how many positives exist in each partition?",
            "Switch your primary metric to precision, recall, F1, or PR-AUC — anything that looks at the rare class specifically.",
            "Compare against the trivial baseline of always predicting the majority class.",
            "Choose the cutoff deliberately using a validation sweep, instead of accepting 0.5.",
            "Only then consider resampling or class weights, and only on training rows.",
        ),
        use=(
            "Whenever the positive class is below roughly 20%, and urgently below 5%.",
            "Whenever the cost of a miss and the cost of a false alarm are very different.",
        ),
        avoid=(
            "Do not resample validation or test — they must keep the real-world prevalence or your estimates become fiction.",
            "Do not reach for SMOTE first. Class weights and threshold tuning are simpler, cheaper, and often as good or better.",
        ),
        myths=(
            (
                "Imbalance must always be corrected before modeling.",
                "Often the model is fine and only the metric and threshold were wrong. Fix the measurement before changing the data.",
            ),
            (
                "Oversampling gives the model more information about the rare class.",
                "Duplicating rows adds no new information. It changes the training prevalence, which shifts the decision boundary and the probability calibration.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, stratify=True, random_state=0)",
            "session.fit(LogisticRegression(class_weight='balanced'))",
            "session.tune_threshold(partition='validation', fp_cost=1.0, fn_cost=10.0)",
            "session.evaluate(partition='test', metrics=['precision', 'recall', 'average_precision'])",
        ),
        check=(
            "What score would 'always predict the majority class' get on your metric?",
            "How many positive rows are in your test partition — 500, or 7?",
        ),
        tools=("resample", "resample_strategies", "tune_threshold", "evaluate"),
        terms=("class imbalance", "precision", "recall", "PR-AUC", "resampling", "threshold"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "model-selection",
        plain=(
            "Model selection is choosing between candidates by comparing them under identical conditions: "
            "same rows, same preparation, same metric, same partition. The winner is the best under that "
            "one setup — not the best model in general, and often not the best for your decision."
        ),
        analogy=(
            "A bake-off only means something if every entrant used the same oven, the same time, and was "
            "judged on the same criterion. Change any of those and the ranking stops being a comparison."
        ),
        steps=(
            "Fix your preparation and split before comparing anything.",
            "Choose one primary metric that reflects the decision you actually care about.",
            "Score every candidate on validation — never on test.",
            "Look at the size of the gap, not just the ordering; a 0.003 difference is usually noise.",
            "Weigh the non-score factors: training cost, latency, interpretability, and how the model fails.",
        ),
        use=(
            "When you have several plausible model families and no strong reason to prefer one.",
            "After a baseline, to check whether extra complexity is buying anything.",
        ),
        avoid=(
            "Do not compare on test — the moment you rank on test, that partition is a selection set.",
            "Do not compare models trained under different preprocessing and call it a model comparison; you compared pipelines.",
        ),
        myths=(
            (
                "The top of the leaderboard is the best model.",
                "It is the best on one metric, one partition, one sample. Rerun with a different seed and the top two often swap.",
            ),
            (
                "More candidates means a better final choice.",
                "More candidates also means more chances for one to win by luck. Wide searches need cross-validation or a fresh holdout to stay honest.",
            ),
        ),
        example=(
            "ranked = session.compare_models(",
            "    {'logreg': LogisticRegression(), 'forest': RandomForestClassifier(random_state=0)},",
            "    partition='validation',",
            ")",
            "session.cv_score(cv=5)          # is the winner's lead bigger than the fold spread?",
        ),
        check=(
            "Is the gap between your top two models larger than the variation across cross-validation folds?",
            "Would you still pick the winner if it were ten times slower to serve?",
        ),
        tools=("compare_models", "cv_score", "nested_cv_score", "evaluate"),
        terms=("model", "metric", "validation", "baseline"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "cross-validation",
        plain=(
            "Cross-validation splits your training rows into a few equal folds, then trains several times — "
            "each time holding out one fold to score on. You end up with several scores instead of one, "
            "which tells you not just how good the model is but how much that number wobbles."
        ),
        analogy=(
            "Instead of one practice exam, you sit five, each on a different chapter you skipped while "
            "revising. The average tells you your level; the spread tells you how much luck was involved."
        ),
        steps=(
            "Keep your final test partition completely out of this — cross-validation happens inside train.",
            "Choose the number of folds; 5 is the common default, more folds means more compute and less data held out each time.",
            "Use stratified folds for classification so each fold keeps the class mix.",
            "Use grouped or time-ordered folds when rows repeat over entities or time.",
            "Read the mean *and* the spread; a model with a lower mean but much tighter spread is often the safer choice.",
        ),
        use=(
            "When your dataset is small and a single validation split would be too noisy to trust.",
            "When tuning hyperparameters, so the settings are not chosen to fit one lucky split.",
        ),
        avoid=(
            "Skip it when a single fit is expensive and you have plenty of data — one large validation partition may be enough.",
            "Do not use plain k-fold on time series; it trains on the future to predict the past.",
        ),
        myths=(
            (
                "Cross-validation prevents overfitting.",
                "It measures overfitting more reliably. It does not stop it — regularization, simpler models, and more data do that.",
            ),
            (
                "The cross-validation score is an unbiased estimate of final performance.",
                "Once you use it to *choose* something, it becomes optimistic. That is what nested cross-validation is for.",
            ),
        ),
        example=(
            "result = session.cv_score(cv=5, scoring='roc_auc')",
            "print(result.mean_score, result.std_score)   # level and wobble",
            "nested = session.nested_cv_score(inner_cv=3, outer_cv=5)   # honest score for the whole search",
        ),
        check=(
            "Is your fold-to-fold spread bigger than the difference you are trying to detect?",
            "Do any of your folds contain rows from the same customer as another fold?",
        ),
        tools=("cv_score", "nested_cv_score", "grid_search", "randomized_search"),
        terms=("cross-validation", "fold", "nested cross-validation", "overfitting"),
        difficulty=CORE,
    ),
    _layer(
        "diagnostic-uncertainty",
        plain=(
            "Every number a model report shows you — accuracy, an importance ranking, a curve — was "
            "computed from a limited sample of rows. Draw a different sample and you would get a slightly "
            "different number. Treating those numbers as exact facts is the most common way to over-read a report."
        ),
        analogy=(
            "A poll of 100 people gives you a percentage. It is a real measurement, but quoting it to two "
            "decimal places and acting on a one-point change would be foolish."
        ),
        steps=(
            "For every headline number, note how many rows it was computed from — and how many of the rare class.",
            "Prefer a range or a fold-spread over a single point estimate.",
            "Before acting on a difference, ask whether it is larger than the noise you would see from reshuffling.",
            "Re-run key comparisons with a different random seed and see whether the conclusion survives.",
            "Report the partition, the sample size, and the limitations alongside the number.",
        ),
        use=(
            "Every time you compare two results, and every time you write a number into a document someone will act on.",
            "Especially when a partition is small, a class is rare, or a slice has only a handful of rows.",
        ),
        avoid=(
            "Do not use uncertainty as an excuse to avoid deciding — the point is to size the evidence, not to refuse it.",
            "Do not compute an interval and then quote only the midpoint.",
        ),
        myths=(
            (
                "A metric computed on real data is a fact.",
                "It is an estimate from a sample. The estimate has its own spread, which shrinks only as the sample grows.",
            ),
            (
                "Small differences matter if they are consistent across runs.",
                "Consistency across runs of the *same* split proves nothing; the split itself is the main source of variation.",
            ),
        ),
        example=(
            "cv = session.cv_score(cv=5)",
            "print(cv.mean_score, '+/-', cv.std_score)",
            "slices = session.error_slices(by=['region'])   # small-n segments are down-ranked",
        ),
        check=(
            "How many rows produced your headline metric, and how many were positive?",
            "If you rerun with `random_state=1`, does your conclusion change?",
        ),
        tools=("cv_score", "evaluate", "error_slices", "learning_curve"),
        terms=("metric", "confidence interval", "statistical significance", "error slice"),
        difficulty=CORE,
    ),
    _layer(
        "probability-calibration",
        plain=(
            "A calibrated model is one whose probabilities can be taken literally: among all the cases it "
            "called 70% likely, about 70% actually happened. A model can rank cases perfectly and still be "
            "badly calibrated, which matters the moment you multiply a probability by a cost."
        ),
        analogy=(
            "A weather forecaster who says '70% chance of rain' should be right about 70% of those days. "
            "One who always says '95%' but gets the rainy days in the right order is a good ranker and a "
            "useless planner."
        ),
        steps=(
            "Ask whether you need the probability itself or only the ordering. Ranking-only use cases do not need calibration.",
            "Plot or tabulate predicted probability against observed frequency in bins — a reliability curve.",
            "If the curve bends away from the diagonal, fit a calibrator: Platt scaling (sigmoid) for small data, isotonic for larger data.",
            "Fit the calibrator on validation or via cross-validation, never on the rows used to train the base model.",
            "Re-check calibration on a partition the calibrator never saw.",
        ),
        use=(
            "When probabilities feed a cost calculation, a budget, an expected-value ranking, or a human decision.",
            "After training boosted trees or SVMs, which are known to produce distorted probabilities.",
        ),
        avoid=(
            "Skip it if you only ever threshold the score into yes/no and tune that threshold directly.",
            "Do not calibrate on the training rows — the base model already fits them, so the curve will look perfect and mean nothing.",
        ),
        myths=(
            (
                "High AUC means well-calibrated probabilities.",
                "AUC only measures ordering. A model can score 0.95 AUC and systematically claim 90% when the truth is 40%.",
            ),
            (
                "Calibration improves accuracy.",
                "Monotonic calibration leaves the ranking, and usually the accuracy at a re-tuned threshold, unchanged. It fixes the *meaning* of the number.",
            ),
        ),
        example=(
            "report = session.calibration(partition='validation', n_bins=10)",
            "print(report.brier_score, report.expected_calibration_error)",
            "# then re-check on test after the calibrator is frozen",
        ),
        check=(
            "Of the cases your model called 80% likely, what fraction actually happened?",
            "Does anything downstream multiply your probability by money or capacity?",
        ),
        tools=("calibration", "evaluate", "tune_threshold", "fit_probabilistic"),
        terms=("calibration", "ROC-AUC", "predict_proba", "validation"),
        difficulty=CORE,
    ),
    _layer(
        "thresholds",
        plain=(
            "A classifier gives you a score between 0 and 1. A threshold is the line where you decide to "
            "act. Moving it trades two kinds of mistake against each other: lower the line and you catch "
            "more real cases but raise more false alarms. There is no universally correct line — it depends "
            "on what each mistake costs you."
        ),
        analogy=(
            "Airport security sensitivity. Turn it up and you catch more genuine threats and search far more "
            "innocent travellers. Turn it down and queues move but you miss more. The right setting is a "
            "policy decision, not a technical one."
        ),
        steps=(
            "Write down what a false positive costs and what a false negative costs, in comparable units.",
            "Sweep the threshold across validation and record precision, recall, and expected cost at each point.",
            "Pick the point that minimizes cost, or that hits a constraint you actually have (like 'we can review 200 cases a day').",
            "Freeze that number and confirm it once on test.",
            "Re-check it after any change in prevalence, because the same threshold means something different when the base rate moves.",
        ),
        use=(
            "For every binary decision system where you act on the prediction rather than just report it.",
            "Especially under imbalance, where the default 0.5 is almost always wrong.",
        ),
        avoid=(
            "Do not tune the threshold on test — that converts your final estimate into another tuning set.",
            "Do not keep one fixed threshold across segments with very different base rates without checking each one.",
        ),
        myths=(
            (
                "0.5 is the natural cutoff.",
                "0.5 is only natural when the classes are balanced and both mistakes cost the same. That combination is rare.",
            ),
            (
                "Changing the threshold changes the model.",
                "The model and its scores are untouched. Only the decision rule applied on top of them moves.",
            ),
        ),
        example=(
            "sweep = session.tune_threshold(",
            "    partition='validation', fp_cost=1.0, fn_cost=20.0,",
            ")",
            "print(sweep.best_threshold, sweep.best_expected_cost)",
            "session.evaluate(partition='test', threshold=sweep.best_threshold)",
        ),
        check=(
            "In your problem, how many false alarms is one missed case worth?",
            "How many cases per day can your team actually review, and what threshold produces that volume?",
        ),
        tools=("tune_threshold", "evaluate", "fit_decision_policy", "apply_decisions"),
        terms=("threshold", "precision", "recall", "cost matrix", "expected value"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "checkpoint-integrity",
        plain=(
            "A checkpoint is a saved snapshot of your working state: the data as it currently stands, the "
            "roles you assigned, which rows are in which partition, and the history of what you ran. It "
            "lets you stop and resume without silently changing the experiment. It is not a saved model."
        ),
        analogy=(
            "Saving your progress in a game. You come back to the same room with the same inventory — but "
            "the save file is not the character sheet you would hand to someone else to play with."
        ),
        steps=(
            "Reach a state worth returning to — usually after ingest, roles, and split are settled.",
            "Call `checkpoint_save` with a directory path.",
            "BuildML writes the data, roles, split membership, history, and a manifest of hashes.",
            "Later, `Session.checkpoint_load` verifies those hashes before restoring, so a corrupted or edited bundle fails loudly.",
            "Use `data_only=True` only when you deliberately want the rows without the workflow meaning.",
        ),
        use=(
            "Before a long or risky operation you might want to undo.",
            "When handing a workflow to a colleague, or resuming after days away.",
        ),
        avoid=(
            "Do not use a checkpoint to deploy a model — use `save_model` or `save_pipeline`, which store the fitted estimator and its feature contract.",
            "Do not load a checkpoint from an untrusted source; the format is serialization-based and executing it is a code-execution risk.",
        ),
        myths=(
            (
                "A checkpoint contains my trained model.",
                "It contains data workflow state. Fitted estimators live in model and pipeline bundles, which are separate artifacts by design.",
            ),
            (
                "Loading a checkpoint proves my earlier work was correct.",
                "It restores what you did, hashes intact. It cannot validate that the choices you recorded were good ones.",
            ),
        ),
        example=(
            "session.checkpoint_save('artifacts/after-split')",
            "restored = Session.checkpoint_load('artifacts/after-split')",
            "print(restored.metadata()['n_rows'], restored.split_plan is not None)",
        ),
        check=(
            "If your machine died right now, which artifact would you reload — and does it contain the model or the data state?",
            "Do you know who produced every bundle you load?",
        ),
        tools=("checkpoint_save", "checkpoint_load", "save_model", "save_pipeline"),
        terms=("checkpoint", "bundle", "serialization", "provenance"),
        difficulty=CORE,
    ),
    _layer(
        "reproducibility",
        plain=(
            "Reproducibility means someone else — including future you — can run the same thing and get "
            "the same answer. That needs more than the code: it needs the random seeds, the library "
            "versions, the exact input data, and a record of the choices you made along the way."
        ),
        analogy=(
            "A recipe that says 'bake until done' is not reproducible. Temperature, time, tin size, and "
            "oven type are all part of the recipe even though none of them is an ingredient."
        ),
        steps=(
            "Pass an explicit `random_state` to every operation that involves randomness — splitting, resampling, model fitting, and search.",
            "Record library versions alongside results; a scikit-learn upgrade can change defaults.",
            "Keep the exact input file, or a hash of it, rather than 'the export from last Tuesday'.",
            "Let BuildML's history record the operation sequence, and export it with your results.",
            "Save a checkpoint at meaningful milestones so the state itself is recoverable, not just the script.",
        ),
        use=(
            "On anything that will be reviewed, audited, handed over, or compared against a future run.",
            "Whenever you are debugging a result you cannot explain — reproducibility is the prerequisite for isolating a cause.",
        ),
        avoid=(
            "Do not fix a seed and then treat the resulting single number as the truth; run several seeds when you need to know the spread.",
            "Do not rely on seeds alone for GPU training, where some operations are non-deterministic by default.",
        ),
        myths=(
            (
                "Setting a seed makes results reproducible.",
                "A seed handles randomness. Version drift, changed input data, and unrecorded manual steps break reproducibility just as thoroughly.",
            ),
            (
                "Operation history proves how the data was produced.",
                "History records what you called through the Session. It knows nothing about the SQL that built your input file.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.fit(RandomForestClassifier(random_state=0))",
            "session.checkpoint_save('artifacts/run-2026-08-03')",
            "print(session.summarize_history())",
        ),
        check=(
            "Could a colleague reproduce your headline number from what is committed today?",
            "Which operations in your notebook use randomness without an explicit seed?",
        ),
        tools=("checkpoint_save", "summarize_history", "walkthrough", "metadata"),
        terms=("seed", "reproducibility", "history", "provenance"),
        difficulty=CORE,
    ),
    _layer(
        "engine-choice",
        plain=(
            "The engine is the library that actually holds and moves your rows — pandas by default, with "
            "Polars and DuckDB available for bigger files. Switching engines changes speed and memory "
            "behaviour. It does not change what a model means or what a metric measures."
        ),
        analogy=(
            "Choosing between a hatchback and a van for the same journey. The route and the destination are "
            "identical; what changes is how much you can carry and how fast you get there."
        ),
        steps=(
            "Start on pandas — it is the default, the simplest mental model, and fine for small and medium data.",
            "If loading or filtering a large file is the bottleneck, ingest with `engine='polars'` or `engine='duckdb'`.",
            "Do the heavy filtering, projecting, and aggregating in the native engine before materializing.",
            "Accept that scikit-learn steps still materialize through pandas — this is not out-of-core training.",
            "Close DuckDB connections with `with session:` or `session.close_native()`.",
        ),
        use=(
            "When a file is too large or too slow to filter comfortably in pandas.",
            "When most of your work is selecting a subset of columns and rows before modeling.",
        ),
        avoid=(
            "Do not switch engines hoping for better model scores; the estimator sees the same numbers either way.",
            "Do not reach for DuckDB or Polars on a 50,000-row CSV — the extra dependency buys you nothing there.",
        ),
        myths=(
            (
                "A faster engine trains bigger models.",
                "The engine speeds up data access. scikit-learn still needs the design matrix in memory, so your model size ceiling is unchanged.",
            ),
            (
                "Lazy Polars means out-of-core machine learning.",
                "A lazy plan collects at the scikit-learn boundary. The collected frame still has to fit in memory.",
            ),
        ),
        example=(
            "session = Session.ingest('data/large.parquet', engine='polars')",
            "session = session.with_engine('duckdb')",
            "with session:",
            "    frame = session.to_pandas()   # materialize when sklearn needs it",
        ),
        check=(
            "Is your bottleneck loading and filtering, or is it model fitting?",
            "Will your design matrix fit in memory regardless of which engine loaded it?",
        ),
        tools=("ingest", "with_engine", "to_engine", "to_pandas", "prepare_design_matrix"),
        terms=("Session", "design matrix"),
        difficulty=CORE,
    ),
    _layer(
        "baselines",
        plain=(
            "A baseline is the simplest prediction anyone could make without machine learning: always guess "
            "the most common class, or always guess the average value. It is the bar your model has to "
            "clear before complexity is worth anything, and it is the only way to know whether 91% is "
            "impressive or embarrassing."
        ),
        analogy=(
            "Telling someone you scored 40 points means nothing until you know whether the game is usually "
            "won at 30 or at 300."
        ),
        steps=(
            "Compute the trivial prediction: majority class for classification, mean or median for regression.",
            "Score it on exactly the same partition and metric you will use for your model.",
            "Write the number down before you start modeling, so you cannot rationalize afterwards.",
            "Add a simple-but-real baseline too — a single-feature rule or a plain logistic regression.",
            "Report every model score as a gap over the baseline, not as an absolute.",
        ),
        use=(
            "At the start of every project, before any model is fitted.",
            "Whenever a stakeholder asks whether a score is good.",
        ),
        avoid=(
            "Do not skip it because the task is obviously hard — that is exactly when a baseline recalibrates expectations.",
            "Do not compare your model to a baseline computed on a different partition or metric; the comparison has to be like for like.",
        ),
        myths=(
            (
                "A high accuracy means the model learned something.",
                "With 97% negatives, predicting 'no' forever scores 97%. The baseline is what exposes that.",
            ),
            (
                "The baseline is just a formality.",
                "Plenty of production models never beat a well-chosen baseline. Finding that out on day one is a gift, not a setback.",
            ),
        ),
        example=(
            "from sklearn.dummy import DummyClassifier",
            "session.fit(DummyClassifier(strategy='most_frequent'))",
            "floor = session.evaluate(partition='validation')",
            "session.fit(RandomForestClassifier(random_state=0))",
            "print(session.evaluate(partition='validation').metrics, 'vs', floor.metrics)",
        ),
        check=(
            "What does 'always predict the most common answer' score on your metric?",
            "Is your model's gap over that baseline large enough to justify the extra complexity?",
        ),
        tools=("fit", "compare_models", "evaluate"),
        terms=("baseline", "accuracy", "metric"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "overfitting",
        plain=(
            "Overfitting is when a model memorizes the particular rows it was trained on — including their "
            "noise and coincidences — instead of learning the underlying pattern. You spot it as a large "
            "gap: excellent on training rows, mediocre on rows it has not seen."
        ),
        analogy=(
            "Memorizing the answers to last year's exam paper. Perfect on that paper, lost the moment the "
            "questions change."
        ),
        steps=(
            "Score the same model on training rows and on held-out rows, and compare.",
            "A big gap means overfitting; poor scores on both means underfitting, which needs the opposite treatment.",
            "Reduce capacity: shallower trees, fewer features, stronger regularization, earlier stopping.",
            "Or increase data: more rows, or augmentation where that makes sense.",
            "Use a learning curve to tell 'need more data' apart from 'need a different model'.",
        ),
        use=(
            "As a diagnostic every time you fit something flexible — deep trees, boosting with many rounds, neural networks, or any model with more parameters than you have rows.",
            "Before assuming a disappointing holdout score means the problem is unsolvable.",
        ),
        avoid=(
            "Do not treat any train-vs-holdout gap as a problem; some gap is normal and expected.",
            "Do not fix overfitting by simplifying a model that is already underfitting — check which direction you are in first.",
        ),
        myths=(
            (
                "Overfitting means the training score is 100%.",
                "It means the *gap* is harmful. A model at 78% train and 62% holdout is overfitting; one at 100% train and 99% holdout may not be.",
            ),
            (
                "More features always help.",
                "Every extra column gives the model another way to fit noise. With limited rows, feature selection often beats feature addition.",
            ),
        ),
        example=(
            "train_score = session.evaluate(partition='train')",
            "valid_score = session.evaluate(partition='validation')",
            "print(train_score.metrics, valid_score.metrics)   # mind the gap",
            "curve = session.learning_curve(cv=5)              # more data, or a different model?",
        ),
        check=(
            "What is your train-minus-validation gap, in the units of your primary metric?",
            "Does your learning curve still slope upward at the largest training size?",
        ),
        tools=("learning_curve", "cv_score", "evaluate", "select_features"),
        terms=("overfitting", "underfitting", "regularization", "generalization", "learning curve"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "feature-importance",
        plain=(
            "Feature importance ranks which columns the fitted model leaned on. It is a description of the "
            "model's behaviour on this dataset with this score — not a statement about which factors cause "
            "the outcome, and not a stable ranking you can quote as a fact about the world."
        ),
        analogy=(
            "Asking which ingredients a particular chef relied on for one dish. Useful for understanding "
            "that chef, that dish, that kitchen. It does not tell you what makes food taste good in general."
        ),
        steps=(
            "Fit the model first — importance describes a fitted model, so there is nothing to describe before that.",
            "Choose a method: permutation importance (shuffle a column, see how much the score drops) works for any model.",
            "Compute it on a held-out partition, not on training rows, or you measure memorization.",
            "Read the ranking with correlated features in mind — two near-duplicate columns split the credit and both look weak.",
            "Re-run with a different seed to see how stable the ordering actually is.",
        ),
        use=(
            "To debug a model — an implausible top feature is often the fastest way to spot leakage.",
            "To decide which columns are worth the cost of collecting and maintaining.",
        ),
        avoid=(
            "Do not present it as causal evidence. 'Removing this feature hurts the model' is not 'changing this in the world changes the outcome'.",
            "Do not compute it on a small partition and read the exact ordering; the top few are meaningful, positions 8 through 15 usually are not.",
        ),
        myths=(
            (
                "The top feature is the main driver of the outcome.",
                "It is the column this model relied on most. Swap in a different model family and the ranking often reshuffles substantially.",
            ),
            (
                "A zero-importance feature is useless.",
                "It may be perfectly informative but redundant with a column the model happened to pick first.",
            ),
        ),
        example=(
            "report = session.feature_importance(",
            "    method='permutation', partition='validation', n_repeats=10, random_state=0,",
            ")",
            "for row in report.importances[:5]:",
            "    print(row.feature, row.mean_importance, row.std_importance)",
        ),
        check=(
            "Does your top feature make sense given when the prediction has to be made?",
            "How much does the ranking change between two different random seeds?",
        ),
        tools=("feature_importance", "select_features", "error_slices", "fit_causal"),
        terms=("feature importance", "permutation importance", "SHAP", "causal inference"),
        difficulty=CORE,
    ),
    _layer(
        "dataset-drift",
        plain=(
            "Drift is a measured difference between two groups of data — training versus production, or "
            "January versus June. It tells you the inputs have changed. It does not by itself tell you the "
            "model got worse, because some changes do not matter and some matter enormously."
        ),
        analogy=(
            "Noticing your regular customers are now mostly a different age group. Something real has "
            "changed. Whether your product still suits them is a separate question you have to answer."
        ),
        steps=(
            "Define the two populations precisely — which rows, which window, which filter.",
            "Compare distributions column by column, and note the effect size, not just whether a test fired.",
            "Check sample sizes; a 'significant' shift over 40 rows is usually noise.",
            "Rule out mechanical causes first: a schema change, a unit change, a new collection process, a renamed category.",
            "If labels are available for the new window, measure actual performance — that is the question you really care about.",
        ),
        use=(
            "Before reusing a model on a new time period or a new population.",
            "When a train-versus-test comparison flags a difference, since that often signals an invalid split rather than genuine change.",
        ),
        avoid=(
            "Do not conclude the model degraded from feature drift alone; without labels you have measured the inputs, not the outputs.",
            "Do not compare populations whose schemas, units, or category meanings differ — the comparison is meaningless before you reconcile them.",
        ),
        myths=(
            (
                "Drift detected means retrain.",
                "Retraining costs money and risk. First check whether the drifted columns are ones the model actually relies on, and whether performance moved at all.",
            ),
            (
                "No drift means the model is safe.",
                "The relationship between features and target can change while every feature distribution stays identical. That is concept drift, and input monitoring misses it entirely.",
            ),
        ),
        example=(
            "report = session.eda()",
            "drift = report.drift",
            "print(drift.available, drift.flagged_columns)",
            "# then: are the flagged columns ones the model actually uses?",
        ),
        check=(
            "Which drifted columns appear in your model's top importances?",
            "Do you have any labels from the new period to check actual performance?",
        ),
        tools=("eda", "error_slices", "feature_importance", "evaluate"),
        terms=("drift", "distribution", "statistical significance"),
        difficulty=CORE,
    ),
    _layer(
        "mutual-information",
        plain=(
            "Mutual information measures how much knowing one column tells you about another. Unlike "
            "correlation it is not limited to straight-line relationships, so it can catch a U-shape or a "
            "threshold effect that correlation reports as zero."
        ),
        analogy=(
            "Correlation asks 'do these two move up and down together?'. Mutual information asks the broader "
            "question 'if I tell you this one, how much better can you guess the other?' — no particular "
            "shape required."
        ),
        steps=(
            "Pick the feature and the target you want to compare.",
            "BuildML estimates how much the target's uncertainty drops once the feature is known.",
            "Read the value as a ranking device: higher means more shared information, zero means none detected.",
            "Compare features against each other rather than against an absolute standard — the units are not intuitive.",
            "Follow up on high scorers with a plot, because the score does not tell you the shape of the relationship.",
        ),
        use=(
            "As a screening pass to shortlist features when you have many columns and no strong prior.",
            "When you suspect a relationship exists but correlation reports nothing.",
        ),
        avoid=(
            "Do not use it to select features for a specific model without also checking with that model; it is model-agnostic and therefore ignores what your estimator can actually exploit.",
            "Do not trust small values on small samples — the estimator is noisy and slightly biased upward.",
        ),
        myths=(
            (
                "Zero correlation means no relationship.",
                "A perfect V-shaped relationship has near-zero correlation and high mutual information. That gap is exactly why this measure exists.",
            ),
            (
                "High mutual information means the feature is causal.",
                "It means the two columns share information. A leaked column shares a great deal of information and is worse than useless.",
            ),
        ),
        example=(
            "report = session.eda()",
            "for row in report.associations.mutual_information[:5]:",
            "    print(row.feature, row.score)",
            "# then plot the top scorer against the target to see the shape",
        ),
        check=(
            "Does your top-scoring feature have a shape a linear model could actually use?",
            "Would that feature be available at prediction time?",
        ),
        tools=("eda", "select_features", "feature_importance"),
        terms=("correlation", "feature selection", "target"),
        difficulty=CORE,
    ),
    _layer(
        "variance-inflation",
        plain=(
            "VIF measures how much one numeric feature can be predicted from the others. When several "
            "columns carry nearly the same information, a linear model cannot tell which one deserves the "
            "credit, so its individual coefficients swing wildly even though the overall prediction is fine."
        ),
        analogy=(
            "Three witnesses who all repeat the same rumour. The story is well supported, but you cannot "
            "work out which witness actually knows anything — and asking them separately gives unstable answers."
        ),
        steps=(
            "Restrict attention to numeric features; VIF is defined by regressing each one on the others.",
            "A VIF near 1 means independent. Values above roughly 5-10 are the usual warning band.",
            "Look at which columns cluster together — often it is a derived column sitting beside its source.",
            "Decide: drop one, combine them, or move to a regularized model that tolerates the redundancy.",
            "Re-check after the change, because removing one column changes everyone else's VIF.",
        ),
        use=(
            "When you plan to interpret linear-model coefficients rather than only use its predictions.",
            "When coefficient signs flip between runs or between folds for no obvious reason.",
        ),
        avoid=(
            "Skip it for tree ensembles — they handle redundant columns without unstable coefficients, though importance still gets split between them.",
            "Do not delete columns purely because VIF is high if prediction quality is all you need; redundancy hurts interpretation far more than accuracy.",
        ),
        myths=(
            (
                "High VIF makes predictions worse.",
                "Predictions usually survive. It is the per-coefficient interpretation that becomes unreliable.",
            ),
            (
                "VIF above 10 means you must drop a column.",
                "The threshold is a convention, not a law. The right response depends on whether you need to interpret coefficients at all.",
            ),
        ),
        example=(
            "report = session.eda()",
            "for row in report.multivariate.vif:",
            "    print(row.feature, row.vif)",
            "# consider reduce_dimensions() or a regularized model when several are high",
        ),
        check=(
            "Are you going to quote individual coefficients to anyone?",
            "Which of your features are arithmetic derivatives of each other?",
        ),
        tools=("eda", "reduce_dimensions", "select_features"),
        terms=("multicollinearity", "linear model", "regularization"),
        difficulty=ADVANCED,
    ),
    _layer(
        "principal-components",
        plain=(
            "PCA rewrites your numeric columns as a smaller set of new columns called components. Each "
            "component is a weighted blend of the originals, chosen so the first one captures as much of "
            "the overall variation as possible, the second as much of what remains, and so on."
        ),
        analogy=(
            "Photographing a 3D object. You choose the angle that shows the most detail in a flat image. "
            "You lose something, but the one good angle beats three bad ones."
        ),
        steps=(
            "Scale your numeric columns first — PCA follows variance, and unscaled units decide the answer for you.",
            "Fit PCA on the training rows and choose how many components to keep, often via cumulative explained variance.",
            "Transform every partition with the frozen components.",
            "Inspect the loadings — how much each original column contributes — before naming a component anything.",
            "Feed the components to the model, remembering that they are blends, not original measurements.",
        ),
        use=(
            "When you have many correlated numeric columns and want to compress them before a distance-based or linear method.",
            "For visualizing high-dimensional data in two or three dimensions.",
        ),
        avoid=(
            "Do not use it when interpretability matters; 'component 3 went up' is not something you can act on.",
            "Do not use it to select predictive features — PCA never looks at the target, so the biggest component can be entirely irrelevant to it.",
        ),
        myths=(
            (
                "PCA picks the most predictive directions.",
                "It picks the highest-variance directions. Variance and predictive value are unrelated; the useful signal can sit in the smallest component.",
            ),
            (
                "95% explained variance means 95% of the information is kept.",
                "It means 95% of the *variance* is kept. Information the target depends on may live entirely in the discarded 5%.",
            ),
        ),
        example=(
            "session.scale(strategy='standard')          # required before PCA",
            "session.reduce_dimensions(n_components=0.95)",
            "print(session.metadata()['n_columns'])      # components replace the originals",
        ),
        check=(
            "Did you scale before reducing, and does the answer change if you did not?",
            "Can you explain what your first component means using its loadings?",
        ),
        tools=("reduce_dimensions", "scale", "fit_clusters", "eda"),
        terms=("PCA", "dimensionality reduction", "scaling", "embedding"),
        difficulty=CORE,
    ),
    _layer(
        "normality-screens",
        plain=(
            "A normality test asks whether a column's values look like they came from the classic bell "
            "curve. It is a description of shape. It is not a permission check — machine learning does not "
            "require normally distributed features."
        ),
        analogy=(
            "Measuring whether a room is exactly square. Useful if you are laying tiles that assume right "
            "angles. Irrelevant if you are just walking through."
        ),
        steps=(
            "Run the screen and read both the test result and the shape summary (skew, kurtosis, a histogram).",
            "Note the sample size — with tens of thousands of rows almost everything fails a normality test on a trivially small deviation.",
            "Ask what actually depends on normality in your plan: some statistical tests and confidence intervals do, most estimators do not.",
            "If a transform would help (log for a long right tail), apply it as a deliberate modeling choice with train-only parameters.",
            "Record the decision rather than silently transforming.",
        ),
        use=(
            "Before applying a statistical procedure whose validity assumes normal residuals or normal inputs.",
            "As a quick way to notice heavy tails, hard floors at zero, or bimodal columns worth investigating.",
        ),
        avoid=(
            "Do not transform every skewed column reflexively; tree models do not care and you lose interpretability.",
            "Do not read a failed normality test on 100,000 rows as a serious finding — read the effect size instead.",
        ),
        myths=(
            (
                "Features must be normally distributed for machine learning.",
                "Almost no estimator requires it. Linear regression assumes things about *residuals*, not about feature distributions.",
            ),
            (
                "A significant test result means the deviation matters.",
                "With enough rows, any tiny deviation is significant. Significance is about detectability, not importance.",
            ),
        ),
        example=(
            "report = session.eda()",
            "for row in report.distributions.normality:",
            "    print(row.column, row.statistic, row.p_value, row.skew)",
            "# large n: read skew and the histogram, not just the p-value",
        ),
        check=(
            "Which step in your plan actually assumes normality?",
            "With your row count, would any real column pass this test?",
        ),
        tools=("eda", "scale", "handle_outliers"),
        terms=("distribution", "skew", "p-value", "statistical significance"),
        difficulty=CORE,
    ),
    _layer(
        "outlier-handling",
        plain=(
            "An outlier is a value far from the rest of the column. Handling it means choosing deliberately "
            "between three options: flag it and investigate, cap it at a boundary, or drop the row. The "
            "worst option is doing it automatically without deciding which case you are in."
        ),
        analogy=(
            "One shopper's basket says £48,000. Either the till glitched, or a business bought pallets. "
            "Deleting the row hides a bug; keeping it unexamined skews every average you compute."
        ),
        steps=(
            "Detect first: use the interquartile-range fence or a z-score threshold to list candidates.",
            "Look at the actual rows. Genuine extremes and data errors need opposite treatments.",
            "If it is an error you cannot fix, dropping the row is defensible — record how many you dropped.",
            "If it is real but destabilizing, capping (winsorizing) keeps the row while limiting its pull.",
            "Learn the fences on training rows only, then apply the same boundaries elsewhere.",
        ),
        use=(
            "Before fitting models sensitive to extremes — linear regression, k-means, anything using squared error.",
            "During EDA, as a data-quality check on columns you do not know well.",
        ),
        avoid=(
            "Do not remove outliers when they *are* the thing you want to predict — fraud, failures, and rare events live in the tail.",
            "Do not apply fences to categorical or one-hot columns; the concept does not transfer.",
        ),
        myths=(
            (
                "Outliers are errors.",
                "Some are. Many are the most informative rows in the dataset. The check is domain knowledge, not a statistical rule.",
            ),
            (
                "Removing outliers improves the model.",
                "It improves the training score by removing hard cases. If those cases occur in production, you made the model worse where it matters most.",
            ),
        ),
        example=(
            "session.handle_outliers(strategy='detect', method='iqr')   # look first",
            "session.handle_outliers(strategy='cap', method='iqr', factor=1.5)",
            "# fences are learned on train and reused on validation/test",
        ),
        check=(
            "Have you actually read ten of the rows your detector flagged?",
            "Is the rare tail the thing you are trying to predict?",
        ),
        tools=("handle_outliers", "eda", "fit_anomaly"),
        terms=("outlier", "skew", "anomaly detection", "distribution"),
        difficulty=CORE,
    ),
    _layer(
        "feature-binning",
        plain=(
            "Binning replaces a precise number with the range it falls into — age 34 becomes '30-39'. You "
            "trade resolution for robustness and for the ability to express step changes that a straight "
            "line cannot."
        ),
        analogy=(
            "Age brackets on a form. You lose the exact birthday, but the bracket is stable, easy to read, "
            "and enough for most decisions."
        ),
        steps=(
            "Decide why you are binning: to tame outliers, to express a genuine step effect, or to match an existing business definition.",
            "Choose the boundaries: equal-width, equal-frequency (quantile), or explicit domain cut points.",
            "Fit the boundaries on training rows so the same edges apply everywhere.",
            "Check bin occupancy — a bin with four rows will produce an unstable estimate.",
            "Treat the result as a category from that point on, including how you encode it.",
        ),
        use=(
            "When the relationship really is stepwise — a policy that changes at age 65, a fee that applies above a threshold.",
            "When you need a compact, explainable feature for a rules-based or regulatory context.",
        ),
        avoid=(
            "Do not bin before a gradient-boosting model as a matter of routine; it finds its own split points and you are only discarding resolution.",
            "Do not choose bin edges by looking at the target across all rows — that is leakage dressed up as feature engineering.",
        ),
        myths=(
            (
                "Binning helps models handle non-linearity.",
                "It lets a *linear* model express steps. For trees and boosting it is a pure information loss.",
            ),
            (
                "Equal-width bins are the neutral choice.",
                "On a skewed column, equal-width bins leave most rows in one bucket and a handful spread across the rest. Quantile bins usually behave better.",
            ),
        ),
        example=(
            "session.bin(columns=['age'], strategy='quantile', n_bins=5)",
            "session.encode(strategy='onehot', columns=['age_binned'])",
            "# edges are learned on train and frozen for other partitions",
        ),
        check=(
            "How many rows land in your smallest bin?",
            "Would a tree model have found better split points on its own?",
        ),
        tools=("bin", "encode", "eda"),
        terms=("binning", "categorical", "cardinality"),
        difficulty=CORE,
    ),
    _layer(
        "target-encoding",
        plain=(
            "Target encoding replaces each category with the average target value observed for that "
            "category. It is compact and often very effective for high-cardinality columns — and it is the "
            "single easiest way to leak, because you are literally putting the answer into a feature."
        ),
        analogy=(
            "Grading a student by the average grade of their class. Informative — unless the student's own "
            "grade is one of the ones being averaged, in which case you have handed them the answer."
        ),
        steps=(
            "Confirm the column has enough distinct values that one-hot would be impractical.",
            "Compute the encoding out-of-fold within the training rows: each row's value comes from folds it was not part of.",
            "Smooth toward the global average so a category seen three times is not trusted as much as one seen three thousand times.",
            "Freeze the mapping and apply it to validation and test.",
            "Decide the fallback value for categories that never appeared in training.",
        ),
        use=(
            "For high-cardinality categorical columns — postcodes, product SKUs, merchant IDs — where one-hot would explode.",
            "With gradient-boosting models, where it often outperforms alternatives by a wide margin.",
        ),
        avoid=(
            "Do not use it without out-of-fold computation. In-fold target encoding will look spectacular in training and collapse in production.",
            "Do not use it on tiny datasets or on categories with only a handful of rows each — the smoothing has nothing to fall back on.",
        ),
        myths=(
            (
                "Target encoding is just a compact one-hot.",
                "One-hot uses no information about the target. Target encoding uses the target directly, which is exactly why it needs fold discipline.",
            ),
            (
                "Computing it on training rows only makes it safe.",
                "Within training, a row still influences its own encoded value unless you go out-of-fold. That is enough to overfit badly.",
            ),
        ),
        example=(
            "session.encode(",
            "    strategy='target', columns=['merchant_id'],",
            "    smoothing=20.0, cv=5,   # out-of-fold inside train",
            ")",
            "# unseen categories fall back to the smoothed global mean",
        ),
        check=(
            "Is your encoding for row i computed from folds that exclude row i?",
            "What value does an unseen category get at scoring time?",
        ),
        tools=("encode", "cv_score", "select_features"),
        terms=("target encoding", "leakage", "out-of-fold", "cardinality"),
        difficulty=ADVANCED,
    ),
    _layer(
        "feature-selection",
        plain=(
            "Feature selection keeps a subset of your columns and drops the rest. Fewer columns can mean "
            "less noise, faster training, cheaper data collection, and less overfitting. The catch is that "
            "the selection itself is a decision learned from data, so it has to be made on training rows only."
        ),
        analogy=(
            "Packing for a trip. Taking everything you own is heavy and slows you down; the skill is "
            "deciding what you will actually use — and deciding it before you leave, not at the destination."
        ),
        steps=(
            "Decide the goal: accuracy, speed, cost, or explainability. They favour different subsets.",
            "Choose a method — univariate scores for a cheap screen, model-based importance for something closer to your estimator.",
            "Run the selection using training rows only.",
            "Freeze the chosen column list before evaluating.",
            "Compare against keeping everything; selection does not automatically help.",
        ),
        use=(
            "When you have many more columns than rows, where noise fitting is nearly guaranteed.",
            "When each column has a real collection or maintenance cost in production.",
        ),
        avoid=(
            "Do not select using scores computed on validation or test — that quietly overfits your selection to the evaluation rows.",
            "Do not drop columns a domain expert insists matter without checking why the score disagrees; it is often a data problem, not an irrelevance.",
        ),
        myths=(
            (
                "Fewer features always generalize better.",
                "Removing a genuinely useful column reduces both training and holdout performance. Selection helps against noise, not against signal.",
            ),
            (
                "Selecting features once at the start is enough.",
                "If you select inside a cross-validated search, the selection must happen inside each fold, or the fold scores are contaminated.",
            ),
        ),
        example=(
            "session.select_features(method='model', k=20, random_state=0)",
            "print(session.metadata()['n_columns'])",
            "session.cv_score(cv=5)   # compare against the full-column score",
        ),
        check=(
            "Was your selection score computed on training rows only?",
            "Does the reduced model actually beat the full one on validation?",
        ),
        tools=("select_features", "feature_importance", "cv_score", "reduce_dimensions"),
        terms=("feature selection", "overfitting", "leakage", "cross-validation"),
        difficulty=CORE,
    ),
    _layer(
        "encoding-imputation-scaling",
        plain=(
            "These three transforms — filling blanks, converting categories to numbers, and putting numbers "
            "on a common range — are the standard preparation trio. They share one rule: each learns "
            "something from the data, so each must learn it from training rows and then be replayed "
            "unchanged everywhere else."
        ),
        analogy=(
            "Tailoring a suit. You take the measurements once, from the person, and then cut every piece to "
            "those measurements. Re-measuring halfway through gives you a garment that fits nobody."
        ),
        steps=(
            "Split first. Nothing in this trio may run before the boundary exists.",
            "Impute, so later steps are not confused by blanks.",
            "Encode, so categories become numeric columns.",
            "Scale, once the final numeric columns exist.",
            "Store the fitted plans so scoring and deployment reproduce the same columns and the same values.",
        ),
        use=(
            "As the default preparation order for classical tabular modeling.",
            "Whenever you are about to save a pipeline, since the plans are what make the pipeline replayable.",
        ),
        avoid=(
            "The order is a default, not a law — skip imputation when your estimator handles blanks meaningfully, and skip scaling for trees.",
            "Do not run any of them before splitting, even 'just to look'. Fitting a scaler on all rows is leakage regardless of intent.",
        ),
        myths=(
            (
                "You must always run all three.",
                "A gradient-boosting model on numeric-plus-native-categorical data may need none of them. Each step should earn its place.",
            ),
            (
                "The order does not matter much.",
                "Scaling before encoding leaves category codes unscaled; encoding before imputing can create a 'missing' category you did not intend. The order encodes assumptions.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.impute(strategy='median')",
            "session.encode(strategy='onehot')",
            "session.scale(strategy='standard')",
            "session.save_pipeline('artifacts/pipeline')   # plans travel with the model",
        ),
        check=(
            "Which of these three does your chosen estimator actually need?",
            "If you loaded your pipeline tomorrow, would it rebuild identical columns?",
        ),
        tools=("impute", "encode", "scale", "apply_preprocess_plans", "save_pipeline"),
        terms=("imputation", "encoding", "scaling", "plan", "pipeline"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "text-features",
        plain=(
            "Text feature extraction turns free-text columns into numeric columns by counting words or "
            "short word sequences. The model never sees language — it sees how often each term appears. "
            "That is often enough for classification, and never enough for understanding."
        ),
        analogy=(
            "Summarizing a book by how many times each word appears. You can tell a cookbook from a thriller "
            "that way. You cannot tell whether the thriller is any good."
        ),
        steps=(
            "Pick the text column and decide the unit: single words, or short sequences of two or three words.",
            "Choose a weighting: raw counts, or TF-IDF, which down-weights terms that appear in nearly every document.",
            "Cap the vocabulary size, or use hashing, so a huge corpus does not produce a million columns.",
            "Fit the vocabulary on training rows only — the term list is learned information.",
            "Apply the frozen vocabulary to other partitions; terms that only appear later are simply dropped.",
        ),
        use=(
            "For classification and clustering over short texts — support tickets, product titles, review snippets.",
            "As a strong, cheap, interpretable baseline before reaching for transformer embeddings.",
        ),
        avoid=(
            "Do not expect it to capture meaning, negation, or word order — 'not good' and 'good' share most of their features.",
            "Do not one-hot a free-text column as if it were a category; near-every value is unique and you get nothing.",
        ),
        myths=(
            (
                "TF-IDF understands importance.",
                "It weights terms by rarity across documents. A rare typo scores highly and means nothing.",
            ),
            (
                "More n-grams is better.",
                "Three-word sequences multiply the vocabulary enormously and mostly add sparse noise unless you have a lot of text.",
            ),
        ),
        example=(
            "session.text_features(",
            "    columns=['ticket_body'], method='tfidf',",
            "    max_features=5000, ngram_range=(1, 2),",
            ")",
            "session.fit(LogisticRegression(max_iter=1000))",
        ),
        check=(
            "How many columns did your vocabulary create, relative to your row count?",
            "Does your task depend on word order or negation?",
        ),
        tools=("text_features", "fit", "rag_ingest_corpus", "make_text_torch_loaders"),
        terms=("token", "embedding", "cardinality", "RAG"),
        difficulty=CORE,
    ),
    _layer(
        "custom-transforms",
        plain=(
            "A custom transform is your own function registered with BuildML so it runs as a first-class "
            "step — recorded in history, replayed at scoring time, and bound by the same train-only rule as "
            "every built-in transform."
        ),
        analogy=(
            "Adding your own tool to a workshop's rack. It is welcome, but it has to hang on the same hook, "
            "follow the same safety rules, and be there when the next person opens the cupboard."
        ),
        steps=(
            "Write a function that takes the frame and returns a transformed frame with a stable set of column names.",
            "If it learns anything from the data — a mapping, a threshold, an average — learn it from training rows and store it.",
            "Register it with `register_transform` under a clear name.",
            "Apply it with `apply_custom_transform` so the call lands in history.",
            "Confirm the output schema is identical for every partition.",
        ),
        use=(
            "For domain logic no generic transform can express — a business rule, a unit conversion, a bespoke parse.",
            "When you want a one-off step to be reproducible and auditable instead of a stray notebook cell.",
        ),
        avoid=(
            "Do not use one to reach across partitions or compute a global statistic; you would be building leakage into a recorded step.",
            "Do not let it produce a different set of columns depending on the input data — downstream operations rely on a stable schema.",
        ),
        myths=(
            (
                "Anything in a registered transform is automatically safe.",
                "Registration gives you lineage and replay. It cannot inspect your function for leakage — that responsibility stays with you.",
            ),
            (
                "Custom transforms are for advanced users only.",
                "A three-line unit conversion is a perfectly good custom transform, and registering it is cheaper than remembering to rerun it.",
            ),
        ),
        example=(
            "def add_ratio(frame):",
            "    frame['spend_per_visit'] = frame['spend'] / frame['visits'].clip(lower=1)",
            "    return frame",
            "session.register_transform('add_ratio', add_ratio)",
            "session.apply_custom_transform('add_ratio')",
        ),
        check=(
            "Does your function produce the same columns on a frame with different data?",
            "Does it read anything computed from rows outside the training partition?",
        ),
        tools=("register_transform", "apply_custom_transform", "list_transforms"),
        terms=("schema", "plan", "leakage", "history"),
        difficulty=CORE,
    ),
    _layer(
        "dry-run-plans",
        plain=(
            "A dry run tells you what an operation would do — what it needs, what would block it, what it "
            "would change, and what could go wrong — without actually doing it or recording anything."
        ),
        analogy=(
            "Reading the recipe all the way through before turning the oven on. You find out you are missing "
            "an ingredient while it is still cheap to find out."
        ),
        steps=(
            "Name the operation, a sequence of operations, or nothing at all for a default next-step preview.",
            "BuildML resolves the current prerequisites and reports availability and blockers.",
            "It lists the effects the operation would have and the leakage risks the catalog records for it.",
            "You read the ranked risks and suggested next steps.",
            "Nothing changes: no data is modified and no history entry is written.",
        ),
        use=(
            "Before an expensive or hard-to-undo operation.",
            "When you are unsure why something is blocked and want the prerequisite chain spelled out.",
        ),
        avoid=(
            "Do not treat a clean dry run as approval — it checks API prerequisites, not whether the operation makes sense for your data.",
            "Do not use it as documentation; `explain` and the concept notes are the teaching surfaces.",
        ),
        myths=(
            (
                "Dry run simulates the result.",
                "It resolves prerequisites and describes intended effects. It does not compute what your metrics would be.",
            ),
            (
                "If dry run says available, the operation is a good idea.",
                "Available means the mechanical requirements are met. Whether it suits your data-generating process is a judgement it cannot make.",
            ),
        ),
        example=(
            "report = session.dry_run('fit')",
            "print(report.ranked_risks)",
            "print(report.suggested_next_ops)",
            "# nothing was mutated and nothing was recorded",
        ),
        check=(
            "What is blocking the operation you want, and which operation would unblock it?",
            "Which of the listed risks apply to your specific dataset?",
        ),
        tools=("dry_run", "workflow", "explain", "assert_can_fit"),
        terms=("dry run", "prerequisite", "workflow resolution", "operation"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "operation-history",
        plain=(
            "History is BuildML's ordered log of everything you ran through the Session: which operation, "
            "with which parameters, in what order, and what changed as a result. It is what makes a session "
            "auditable instead of a pile of notebook cells."
        ),
        analogy=(
            "A lab notebook that writes itself. It records what you did and when. It does not vouch for "
            "where your samples came from or whether your method was sound."
        ),
        steps=(
            "Every public operation appends a record automatically — you do not have to do anything.",
            "Each record carries a sequence number, the parameters, a result summary, and the state transition.",
            "`summarize_history` counts operations and surfaces unresolved risks.",
            "`walkthrough` combines history with workflow resolution into a review document you can export.",
            "Checkpoints carry the history along, so a resumed session keeps its lineage.",
        ),
        use=(
            "When handing work over, reviewing your own long session, or writing up what you did.",
            "When debugging: the order of operations frequently explains a surprising result.",
        ),
        avoid=(
            "Do not present history as data provenance — it starts at ingest and knows nothing about the pipeline that produced your file.",
            "Do not treat a complete history as evidence of a valid method; it records choices, it does not grade them.",
        ),
        myths=(
            (
                "History proves the workflow was correct.",
                "It proves what was called. Calling `split` after `scale` is faithfully recorded and still wrong.",
            ),
            (
                "Manual pandas edits show up in history.",
                "Only Session operations are recorded. Reaching into the frame directly leaves no trace.",
            ),
        ),
        example=(
            "summary = session.summarize_history()",
            "print(summary.operation_counts, summary.unresolved_risks)",
            "session.walkthrough(export_html='artifacts/workflow.html')",
        ),
        check=(
            "Does your history show a split before every fitted transform?",
            "Which steps in your analysis happened outside the Session and are therefore invisible?",
        ),
        tools=("summarize_history", "walkthrough", "workflow", "checkpoint_save"),
        terms=("history", "operation", "provenance", "workflow resolution"),
        difficulty=FOUNDATION,
    ),
)

__all__ = ["CLASSICAL_BEGINNER"]
