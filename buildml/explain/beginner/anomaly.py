# ruff: noqa: E501
"""Beginner layers for anomaly and novelty detection."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, FOUNDATION, BeginnerLayer, _index, _layer

ANOMALY_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "anomaly-train-fit-holdout-score",
        plain=(
            "An anomaly detector learns what 'normal' looks like from your training rows, then gives every "
            "other row a strangeness score. The rule is the same as everywhere else in BuildML: learn "
            "normality on train, freeze it, and score held-out rows without updating the notion of normal."
        ),
        analogy=(
            "A night guard learns the usual comings and goings over a few weeks. Later they flag someone "
            "unusual. They do not redefine 'usual' to include the intruder they are trying to spot."
        ),
        steps=(
            "Split first, so there is a partition the detector has never influenced.",
            "Scale numeric columns if your detector is distance-based; most are.",
            "Fit the detector on training rows to learn the shape of normal.",
            "Score validation or test rows with the frozen plan — you get a continuous strangeness score per row.",
            "Apply a threshold to turn scores into flags, and report the resulting alert rate.",
        ),
        use=(
            "When you have very few or no labelled examples of the thing you want to catch.",
            "For data-quality monitoring, fraud screening, and equipment health, where 'unusual' is the best available proxy for 'bad'.",
        ),
        avoid=(
            "Do not use it when you actually have decent labels — a supervised classifier will almost always beat an unsupervised detector.",
            "Do not refit the detector on the rows you are about to score; the anomalies would help define normality.",
        ),
        myths=(
            (
                "No labels means no leakage risk.",
                "Fitting normality on all rows and then scoring 'held-out' rows leaks. The detector already absorbed those rows into its notion of normal.",
            ),
            (
                "Anomalous means bad.",
                "It means unusual. A legitimate bulk order and a fraudulent one both look unusual. Anomaly scores are a triage tool, not a verdict.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.scale(strategy='standard')",
            "session.fit_anomaly(method='isolation_forest', random_state=0)",
            "scores = session.score_anomalies(partition='test')",
        ),
        check=(
            "Which partition did your reported alert rate come from?",
            "Would a supervised model be possible if you labelled 200 rows?",
        ),
        tools=("fit_anomaly", "score_anomalies", "evaluate_anomaly", "scale"),
        terms=("anomaly detection", "leakage", "split", "plan"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "anomaly-threshold-alert-rate",
        plain=(
            "A detector produces scores, not decisions. You choose the cut-off, and that choice sets your "
            "alert rate — the fraction of rows you flag. Alert rate is the number that determines how much "
            "human review work you have just created, so it belongs beside every claim you make."
        ),
        analogy=(
            "A metal detector's sensitivity dial. There is no 'correct' setting; there is the setting that "
            "matches how many bags your staff can actually search per hour."
        ),
        steps=(
            "Score a held-out partition to get the distribution of strangeness scores.",
            "Decide your capacity: how many alerts per day can be reviewed properly?",
            "Set the threshold at the score quantile that produces that volume.",
            "If you have any labels, check precision at that alert rate — of the flagged rows, how many were genuinely bad?",
            "Report the threshold, the alert rate, and the partition together, every time.",
        ),
        use=(
            "Before any anomaly system goes anywhere near a real workflow.",
            "Whenever the volume of data changes, since a fixed threshold produces a different absolute alert count.",
        ),
        avoid=(
            "Do not accept a library's default contamination parameter as your operating point; it is an arbitrary guess about your data.",
            "Do not set the threshold on the same rows you use to report performance.",
        ),
        myths=(
            (
                "The detector decides what is an anomaly.",
                "The detector ranks. You decide where the line is, and that decision is a capacity and cost question.",
            ),
            (
                "A 1% alert rate is standard.",
                "There is no standard. One percent of ten million rows is a hundred thousand alerts, which no team can review.",
            ),
        ),
        example=(
            "scores = session.score_anomalies(partition='validation')",
            "report = session.evaluate_anomaly(partition='validation', alert_rate=0.02)",
            "print(report.threshold, report.alert_rate, report.n_flagged)",
        ),
        check=(
            "How many alerts per day does your threshold produce, and who reviews them?",
            "What fraction of your alerts turn out to be worth acting on?",
        ),
        tools=("score_anomalies", "evaluate_anomaly", "tune_threshold", "fit_decision_policy"),
        terms=("threshold", "alert rate", "precision", "anomaly detection"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "anomaly-novelty-vs-unsupervised",
        plain=(
            "Two modes, two different assumptions. Unsupervised mode fits on all your training rows and "
            "assumes anomalies are rare enough not to distort normality. Novelty mode fits on rows you have "
            "certified as clean, and then flags anything unlike them."
        ),
        analogy=(
            "Unsupervised is learning what a normal street looks like by watching everyone, accepting that "
            "a few odd characters are in the crowd. Novelty is learning from a vetted guest list and "
            "treating everyone else as a stranger."
        ),
        steps=(
            "Ask whether you can identify a clean subset of training rows with confidence.",
            "If yes, use novelty mode and pass that clean subset; BuildML records that you did.",
            "If no, use unsupervised mode over all training rows.",
            "Either way, the holdout scoring contract is identical: frozen plan, held-out rows.",
            "State which mode you used, because it changes what a flag means.",
        ),
        use=(
            "Novelty when you have verified-good historical periods — a known-clean month, a certified batch.",
            "Unsupervised when contamination is unavoidable or unknown, which is the common case.",
        ),
        avoid=(
            "Do not claim novelty mode when your 'clean' set was never actually verified; the guarantee it implies would be false.",
            "Do not use unsupervised mode when contamination is heavy — if 30% of rows are anomalous, they define normality.",
        ),
        myths=(
            (
                "Novelty mode is simply the better option.",
                "It is better only when your clean subset is genuinely clean. A contaminated 'clean' set gives you false confidence on top of the same errors.",
            ),
            (
                "The two modes give similar results.",
                "They learn different notions of normal, so the same row can be ordinary in one mode and extreme in the other.",
            ),
        ),
        example=(
            "session.fit_anomaly(",
            "    method='one_class_svm', mode='novelty',",
            "    normal_only_filter='verified_clean == True',",
            ")",
            "print(session.anomaly_plan.disclosures)   # records the normal-only subset",
        ),
        check=(
            "How was your 'clean' subset verified, and by whom?",
            "Roughly what fraction of your training rows do you believe are anomalous?",
        ),
        tools=("fit_anomaly", "score_anomalies", "evaluate_anomaly"),
        terms=("anomaly detection", "unsupervised", "disclosure", "train"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-imbalance-metrics",
        plain=(
            "When you do have labels for anomalies, they are almost always extremely rare — under 1%. That "
            "wrecks the usual metrics. Accuracy is meaningless and even ROC-AUC can look respectable while "
            "your alerts are almost all false. Precision, recall, and PR-AUC are the honest choices."
        ),
        analogy=(
            "Finding 20 counterfeit notes in a million. Getting 99.998% of notes right is trivial. The only "
            "interesting numbers are how many fakes you caught and how many genuine notes you seized."
        ),
        steps=(
            "Compute PR-AUC as your primary label-aware score.",
            "Report precision@k and recall@k at the alert rate you will actually operate at.",
            "Compare against the trivial baseline: random flagging at the same rate.",
            "Note how many positive rows exist in the partition — with 12 positives, every metric is fragile.",
            "Only then look at ROC-AUC, and read it as a ranking summary rather than a quality claim.",
        ),
        use=(
            "Whenever any labels exist, even a partial or delayed set.",
            "When comparing detectors, since the ranking under PR-AUC often differs from the ranking under ROC-AUC.",
        ),
        avoid=(
            "Do not use accuracy. At 0.1% prevalence, flagging nothing scores 99.9%.",
            "Do not read ROC-AUC as if it reflected precision; with rare positives a 0.95 AUC can still mean 95% of your alerts are false.",
        ),
        myths=(
            (
                "ROC-AUC handles imbalance well.",
                "It is invariant to prevalence, which sounds good and means it hides how bad precision gets when positives are rare.",
            ),
            (
                "A metric above 0.9 means the system is ready.",
                "At an operating threshold you may still be sending your team ninety false alerts for every real one. Precision at the operating point is the number that matters.",
            ),
        ),
        example=(
            "report = session.evaluate_anomaly(",
            "    partition='test', label_column='is_fraud', alert_rate=0.01,",
            ")",
            "print(report.pr_auc, report.precision_at_k, report.recall_at_k)",
        ),
        check=(
            "How many labelled positives are in your evaluation partition?",
            "At your operating alert rate, what fraction of alerts are real?",
        ),
        tools=("evaluate_anomaly", "score_anomalies", "tune_threshold"),
        terms=("PR-AUC", "precision", "recall", "ROC-AUC", "class imbalance"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-eda-boundary",
        plain=(
            "BuildML has two things that both mention isolation forests, and they are not the same. The EDA "
            "screen is a quick descriptive look at odd rows during exploration. `fit_anomaly` builds a real, "
            "leakage-safe, saveable detector with scoring, thresholds, and evaluation."
        ),
        analogy=(
            "Glancing at a room and noticing something out of place, versus installing an alarm system. The "
            "glance is useful; you would not stake operations on it."
        ),
        steps=(
            "During exploration, read the EDA outlier screen to get a feel for which rows are unusual.",
            "Do not build a process on that screen — it is descriptive and partition-agnostic.",
            "When you need an operational detector, call `fit_anomaly` after splitting.",
            "That produces a plan you can score with, threshold, evaluate, and save.",
            "Keep the two mentally separate in your write-up so readers know which one produced a number.",
        ),
        use=(
            "The EDA screen while you are still understanding the data.",
            "The product path as soon as anything downstream depends on the flags.",
        ),
        avoid=(
            "Do not quote EDA outlier counts as detector performance; the screen never held out a partition.",
            "Do not skip EDA either — it often tells you the anomalies are a data-collection bug rather than a phenomenon.",
        ),
        myths=(
            (
                "They are the same algorithm, so the results are equivalent.",
                "Same family, different contract. One is fitted on whatever it was shown; the other is fitted on train and frozen for scoring.",
            ),
            (
                "EDA is for beginners and the product path is for experts.",
                "Both are for everyone. They answer different questions at different stages.",
            ),
        ),
        example=(
            "report = session.eda()",
            "print(report.outliers.isolation_forest)   # descriptive screen",
            "session.split(test_size=0.2, random_state=0)",
            "session.fit_anomaly(method='isolation_forest', random_state=0)   # the product path",
        ),
        check=(
            "Which of the two produced the number you are about to present?",
            "Does anything operational depend on the flags you are looking at?",
        ),
        tools=("eda", "fit_anomaly", "score_anomalies", "handle_outliers"),
        terms=("anomaly detection", "outlier", "plan", "split"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-bundle-boundary",
        plain=(
            "A fitted detector saves as an anomaly bundle: the estimator, the feature contract, the mode, "
            "and the disclosures. It is a separate artifact from a Session checkpoint and from every other "
            "domain's bundle."
        ),
        analogy=(
            "The alarm system's configuration file is not the building's floor plan. You need both, and "
            "they live in different places."
        ),
        steps=(
            "Fit a detector so an anomaly plan exists.",
            "Call `save_anomaly_bundle(path)`.",
            "Reload with `load_anomaly_bundle(path)` in a fresh Session or a scheduled job.",
            "Confirm the feature columns still exist, then score.",
            "Keep your threshold with the bundle — a detector without its operating point is only half a system.",
        ),
        use=(
            "When the detector runs on a schedule outside your notebook.",
            "When the mode and disclosures must travel with the model for audit.",
        ),
        avoid=(
            "Do not assume the threshold is inside the bundle unless you put it there; record it explicitly.",
            "Do not load a detector bundle expecting your training data back.",
        ),
        myths=(
            (
                "Saving the detector saves the whole system.",
                "Scoring also needs the same preprocessing and the chosen threshold. Save or record those too.",
            ),
            (
                "One bundle format would be simpler.",
                "Each domain has a different contract to enforce at load time. A shared format would only defer the failure to prediction time.",
            ),
        ),
        example=(
            "session.save_anomaly_bundle('artifacts/fraud-detector')",
            "job = Session.ingest(today_frame).load_anomaly_bundle('artifacts/fraud-detector')",
            "flags = job.score_anomalies()",
        ),
        check=(
            "Where is your operating threshold recorded?",
            "Does the scheduled job apply the same scaling the detector was fitted with?",
        ),
        tools=("save_anomaly_bundle", "load_anomaly_bundle", "score_anomalies", "checkpoint_save"),
        terms=("bundle", "checkpoint", "plan", "threshold"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-isolation-forest",
        plain=(
            "Isolation Forest is the default tabular anomaly detector in BuildML: an ensemble of "
            "random trees isolates unusual rows quickly. Shorter isolation paths mean higher "
            "anomaly scores once the plan orients them so higher means more anomalous."
        ),
        analogy=(
            "Odd items get separated from the crowd in fewer random cuts than normal ones — "
            "like finding the one red sock in a laundry pile with fewer grabs."
        ),
        steps=(
            "Split first, then scale numeric features when distances matter.",
            "Call fit_anomaly(method='isolation_forest') on train only.",
            "Score holdout rows and read alert_rate with a disclosed threshold.",
            "Compare against LOF or One-Class SVM on validation if ranking differs.",
        ),
        use=(
            "Fast multivariate baseline when labels are scarce or delayed.",
            "First detector to try before heavier PyOD or torch paths.",
        ),
        avoid=(
            "Heavy contamination in train without novelty mode or a verified clean subset.",
            "Reporting alert_rate from train without a holdout partition.",
        ),
        myths=(
            ("Isolation Forest needs labels.", "It is unsupervised; labels are for evaluation only."),
            ("Default contamination is your operating point.", "It is a prior guess — set threshold for capacity."),
        ),
        example=(
            "session.fit_anomaly(method='isolation_forest', contamination=0.05, random_state=0)",
            "session.evaluate_anomaly(partition='validation')",
        ),
        check=(
            "Did you scale before comparing against LOF or One-Class SVM?",
            "Which partition produced the alert_rate you are reporting?",
        ),
        tools=("fit_anomaly", "score_anomalies", "evaluate_anomaly"),
        terms=("anomaly detection", "threshold", "outlier"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-lof",
        plain=(
            "Local Outlier Factor compares how dense a point is relative to its k-nearest "
            "neighbours. Rows that look sparse among dense neighbours score as anomalies — "
            "a local-density view rather than a global distance fence."
        ),
        analogy=(
            "Someone standing alone in a packed room while everyone around them is in tight "
            "groups — locally sparse even if the room overall is crowded."
        ),
        steps=(
            "Scale numeric features so neighbour distances are meaningful.",
            "Choose n_neighbors with your expected cluster size in mind.",
            "fit_anomaly(method='lof') on train only.",
            "Score validation/test and compare alert_rate against Isolation Forest.",
        ),
        use=(
            "When anomalies are locally sparse rather than globally far from everything.",
            "As a second opinion when Isolation Forest flags too many borderline rows.",
        ),
        avoid=(
            "Very high dimensions without feature selection or scaling.",
            "Assuming LOF and Isolation Forest must agree on the same rows.",
        ),
        myths=(
            ("LOF and Isolation Forest always agree.", "They optimize different notions of unusual."),
            ("More neighbours always helps.", "Too large k washes out local structure."),
        ),
        example=(
            "session.fit_anomaly(method='lof', n_neighbors=20, random_state=0)",
            "session.score_anomalies(partition='test')",
        ),
        check=(
            "Is your feature scale meaningful for k-nearest neighbours?",
            "Does validation alert_rate fit review capacity?",
        ),
        tools=("fit_anomaly", "score_anomalies", "evaluate_anomaly"),
        terms=("anomaly detection", "outlier", "scaling"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-one-class-svm",
        plain=(
            "One-Class SVM learns a tight boundary around normal train data in feature space. "
            "Points outside the learned envelope score as anomalies; threshold_policy='decision_zero' "
            "uses the sign of sklearn's decision_function."
        ),
        analogy=(
            "Draw the smallest fence around verified-normal examples; anyone outside the fence "
            "is treated as a stranger even if they are not the strangest person in the city."
        ),
        steps=(
            "Scale numeric columns before fitting the RBF kernel.",
            "fit_anomaly(method='one_class_svm', kernel='rbf') on train or a normal-only subset.",
            "Read disclosures for nu, kernel, and train alert_rate.",
            "Evaluate validation alert_rate before locking an operating threshold.",
        ),
        use=(
            "When the normal class is compact and anomalies lie clearly outside in feature space.",
            "Novelty mode when you can certify a clean train subset.",
        ),
        avoid=(
            "Large n without subsampling — training cost grows quickly.",
            "Unscaled mixed-scale columns with an RBF kernel.",
        ),
        myths=(
            ("One-Class SVM finds the same anomalies as Isolation Forest.", "Different boundaries, different flags."),
            ("nu is the holdout alert rate.", "nu is a training prior; holdout alert_rate can differ."),
        ),
        example=(
            "session.fit_anomaly(method='one_class_svm', nu=0.05, random_state=0)",
            "session.evaluate_anomaly(partition='validation')",
        ),
        check=(
            "Did you disclose novelty vs unsupervised mode?",
            "Is validation alert_rate acceptable for reviewers?",
        ),
        tools=("fit_anomaly", "evaluate_anomaly", "score_anomalies"),
        terms=("anomaly detection", "threshold", "scaling"),
        difficulty=CORE,
    ),
    _layer(
        "anomaly-pyod-hbos-copod-ecod",
        plain=(
            "With buildml[anomaly-industry], PyOD adds histogram- and copula-based detectors "
            "(HBOS, COPOD, ECOD). They score multivariate tails quickly and extend the catalog "
            "beyond the sklearn isolation/LOF/One-Class trio."
        ),
        analogy=(
            "Check how extreme each column looks on its own or via a copula, then combine those "
            "tail signals — like multiple thermometers agreeing something is off."
        ),
        steps=(
            "Read anomaly_capability_matrix()['backends']['pyod']['available'].",
            "fit_anomaly(backend='pyod', method='copod'|'hbos'|'ecod') on train.",
            "Score validation and compare alert_rate against isolation_forest on the same split.",
            "Record that PyOD was the backend in your write-up.",
        ),
        use=(
            "Industry-depth tabular scoring when PyOD is installed.",
            "When sklearn trio rankings plateau but you still lack labels.",
        ),
        avoid=(
            "Requesting pyod methods without buildml[anomaly-industry] installed.",
            "Treating PyOD scores as calibrated probabilities without checking alert_rate.",
        ),
        myths=(
            ("PyOD path is always better.", "It is another honest baseline, not a guaranteed win."),
            ("PyOD removes the need for scaling.", "Distance/tail methods still benefit from sensible feature scale."),
        ),
        example=(
            "# pip install 'buildml[anomaly-industry]'",
            "session.fit_anomaly(backend='pyod', method='ecod', random_state=0)",
        ),
        check=(
            "Does anomaly_capability_matrix show pyod available?",
            "Did you compare validation alert_rate against sklearn baselines?",
        ),
        tools=("fit_anomaly", "anomaly_capability_matrix", "evaluate_anomaly"),
        terms=("anomaly detection", "extra", "outlier"),
        difficulty=ADVANCED,
    ),
    _layer(
        "anomaly-autoencoder",
        plain=(
            "The torch backend trains a small feedforward autoencoder on normal train rows and "
            "uses reconstruction mean-squared error as the anomaly score. High error means the "
            "row did not match patterns the encoder learned as normal."
        ),
        analogy=(
            "If the model cannot rebuild a row well, that row probably did not belong to the "
            "shapes it memorized from training — like a photocopier struggling on a forged note."
        ),
        steps=(
            "Scale features and confirm buildml[torch] is available.",
            "fit_anomaly(backend='torch', method='autoencoder', epochs=...) on train.",
            "Score holdout rows; read reconstruction-error and alert_rate disclosures.",
            "Compare against isolation_forest on validation before claiming uplift.",
        ),
        use=(
            "Nonlinear normal manifold when sklearn or PyOD rankings plateau.",
            "When reconstruction error aligns with domain intuition (sensor profiles, embeddings).",
        ),
        avoid=(
            "Tiny train sets where the AE memorizes instead of generalizing normal.",
            "Scoring without the same scaling pipeline used at fit time.",
        ),
        myths=(
            ("Autoencoder replaces labels.", "It still follows train-fit / holdout-score like every detector."),
            ("Higher epochs always help.", "Under- or over-training shifts reconstruction calibration."),
        ),
        example=(
            "session.fit_anomaly(backend='torch', method='autoencoder', epochs=40, random_state=0)",
            "session.score_anomalies(partition='validation')",
        ),
        check=(
            "Are feature columns identical at score time?",
            "Does validation alert_rate beat a sklearn baseline on the same split?",
        ),
        tools=("fit_anomaly", "score_anomalies", "anomaly_capability_matrix"),
        terms=("anomaly detection", "neural network", "extra"),
        difficulty=ADVANCED,
    ),
)

__all__ = ["ANOMALY_BEGINNER"]
