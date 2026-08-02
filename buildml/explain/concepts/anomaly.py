# ruff: noqa: E501
"""Anomaly / fraud detection concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

ANOMALY_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="anomaly-train-fit-holdout-score",
            title="Anomaly train-fit / holdout-score",
            summary="Fit the detector on train only; score and flag holdout rows with a frozen plan — never refit on evaluation partitions.",
            definition=(
                "Train-fit / holdout-score is the leakage-safe anomaly contract: "
                "estimate a detector (and usually a threshold) on the training "
                "partition, freeze the AnomalyPlan, and score validation/test rows "
                "without updating that detector."
            ),
            intuition=(
                "If you redraw the anomaly boundary using the exam answers, the exam "
                "no longer measures generalization. Fit the ruler on train; measure "
                "holdout points with that frozen ruler."
            ),
            formal_idea=(
                "Let f_train be a detector fit on X_train (or a normal-only subset). "
                "For partition P, scores s_P = score(f_train, X_P) with higher s more "
                "anomalous; flags ŷ_P = 1[s_P ≥ τ] for a disclosed threshold τ."
            ),
            why_it_matters=(
                "Refitting on all rows contaminates holdout alert rates and ranking metrics.",
                "Teaching and model cards need explicit fit-partition and threshold disclosures.",
            ),
            how_buildml_uses=(
                "Session.fit_anomaly requires a SplitPlan and fits on train only.",
                "Session.score_anomalies / evaluate_anomaly reuse the frozen AnomalyPlan.",
                "Novelty mode further restricts fit rows to a normal-only train subset.",
            ),
            interpretation_rules=(
                "Read partition name, threshold policy, and alert_rate beside every metric.",
                "Train-partition metrics are optimistic for threshold selection.",
            ),
            assumptions=(
                "A disjoint SplitPlan exists before fit_anomaly.",
                "Feature columns are numeric and imputed before distance-based methods.",
            ),
            failure_modes=(
                "Fitting detectors on the full frame before splitting.",
                "Treating EDA IsolationForest screens as a fitted AnomalyPlan.",
            ),
            anti_patterns=(
                "Calling sklearn fit_predict on concatenated train+test for 'evaluation'.",
            ),
            worked_example_pattern=(
                "split → scale → fit_anomaly → evaluate_anomaly(partition='validation').",
            ),
            related_concepts=("leakage-boundary", "evaluation-partitions", "anomaly-threshold-alert-rate"),
        ),
        _note(
            key="anomaly-threshold-alert-rate",
            title="Thresholds and alert rates",
            summary="Anomaly flags require an explicit threshold; alert rate is the fraction flagged and must be reported with every operational claim.",
            definition=(
                "A threshold τ converts continuous anomaly scores into binary alerts. "
                "Alert rate is |{i : s_i ≥ τ}| / n on a named partition. Contamination "
                "and quantile policies disclose how τ was chosen from train scores."
            ),
            intuition=(
                "Without a disclosed cut, 'anomaly' is just a ranking. Operations need "
                "to know how many alarms fire — and that holdout rates can drift."
            ),
            formal_idea=(
                "Policies include contamination (τ ≈ Q_{1-c}(s_train)), quantile, "
                "absolute score_threshold, and decision_zero for One-Class SVM. "
                "Scores are oriented so higher means more anomalous."
            ),
            why_it_matters=(
                "Silent thresholding hides trade-offs between precision and recall.",
                "Comparing detectors without alert_rate confuses ranking quality with operating point.",
            ),
            how_buildml_uses=(
                "AnomalyPlan stores threshold_, threshold_policy, and train_alert_rate_.",
                "score_anomalies / evaluate_anomaly always report alert_rate and threshold.",
                "override_threshold can change a single call without mutating the plan.",
            ),
            interpretation_rules=(
                "Prefer validation/test alert rates for operational claims.",
                "Under labels, pair alert_rate with precision/recall@k and PR-AUC.",
            ),
            assumptions=(
                "Train score distribution is a reasonable calibration reference.",
                "Score orientation (higher = more anomalous) is understood by consumers.",
            ),
            failure_modes=(
                "Publishing flags without threshold or alert_rate.",
                "Assuming holdout alert_rate equals the contamination prior.",
            ),
            anti_patterns=(
                "Retuning τ on the test partition after peeking at labels.",
            ),
            worked_example_pattern=(
                "fit_anomaly(contamination=0.05) → score_anomalies → read alert_rate.",
            ),
            related_concepts=("anomaly-train-fit-holdout-score", "anomaly-imbalance-metrics"),
        ),
        _note(
            key="anomaly-novelty-vs-unsupervised",
            title="Novelty vs unsupervised anomaly modes",
            summary="Unsupervised fits on all train rows; novelty fits on a disclosed normal-only train subset — different assumptions, same holdout score contract.",
            definition=(
                "Unsupervised anomaly detection estimates unusualness from a train "
                "mixture that may already contain anomalies (contamination prior). "
                "Novelty / normal-only mode fits exclusively on rows labeled normal, "
                "then scores new rows for departure from that normal class."
            ),
            intuition=(
                "Unsupervised asks 'what looks rare in this bag?'; novelty asks "
                "'does this look unlike the clean examples I trained on?'"
            ),
            formal_idea=(
                "Unsupervised: f ← fit(X_train). Novelty: f ← fit(X_train[y=normal]). "
                "Both freeze f for holdout scoring. Novelty is semi-supervised in the "
                "label-for-normal sense — not Phase-2 representation learning."
            ),
            why_it_matters=(
                "Using novelty without a honest normal-only subset leaks anomalies into the fit.",
                "Contamination priors are meaningless if the fit subset was already filtered clean.",
            ),
            how_buildml_uses=(
                "mode='unsupervised' vs mode='novelty' on Session.fit_anomaly.",
                "novelty requires normal_label_column (or a single target role) and normal_label_value.",
                "Catalog and guides disclose the normal-only contract explicitly.",
            ),
            interpretation_rules=(
                "State which mode was used beside every alert_rate.",
                "Do not call novelty 'unlabeled' when normal labels selected the fit rows.",
            ),
            assumptions=(
                "For novelty, normal labels on train are trusted enough to define the fit subset.",
                "Feature space remains comparable between fit and score partitions.",
            ),
            failure_modes=(
                "Novelty with polluted 'normal' labels.",
                "Confusing novelty with supervised fraud classification.",
            ),
            anti_patterns=(
                "Filtering to normal on train and then claiming a fully unsupervised detector.",
            ),
            worked_example_pattern=(
                "fit_anomaly(mode='novelty', normal_label_column='is_fraud', "
                "normal_label_value=0) → evaluate_anomaly.",
            ),
            related_concepts=("anomaly-train-fit-holdout-score", "anomaly-eda-boundary"),
        ),
        _note(
            key="anomaly-imbalance-metrics",
            title="Imbalance-honest anomaly metrics",
            summary="When labels exist, prefer PR-AUC and precision/recall@k; accuracy and raw ROC-AUC can mislead under rare positives.",
            definition=(
                "Imbalance-honest evaluation reports ranking and operating-point metrics "
                "that remain informative when positives are rare: average precision "
                "(PR-AUC), precision/recall/F1 at the disclosed threshold, and "
                "precision@k / recall@k, alongside alert_rate and positive_rate."
            ),
            intuition=(
                "If fraud is 1%, a model that never alerts is 99% accurate and useless. "
                "Ask how well rare events rank and what you pay in false alarms."
            ),
            formal_idea=(
                "With y ∈ {0,1} and scores s, AP = area under the precision-recall curve. "
                "At threshold τ, precision/recall use ŷ=1[s≥τ]. At budget k, "
                "precision@k uses the top-k scores."
            ),
            why_it_matters=(
                "Fraud-like tasks are almost always imbalanced.",
                "Supervised mode reuses classical binary patterns but must keep the same honesty.",
            ),
            how_buildml_uses=(
                "evaluate_anomaly fills labeled_metrics when a label/target is available.",
                "Disclosures warn when positive_rate is low.",
                "Supervised mode fits HistGradientBoostingClassifier; unsupervised modes "
                "use labels for evaluation only.",
            ),
            interpretation_rules=(
                "Always publish positive_rate and alert_rate with labeled metrics.",
                "Do not equate PR-AUC with causal recovery of fraud mechanisms.",
            ),
            assumptions=(
                "Labels used for evaluation are scoped honestly (no peeking into fit for unsupervised).",
                "positive_label matches the rare/event class of interest.",
            ),
            failure_modes=(
                "Reporting accuracy alone on imbalanced fraud data.",
                "Tuning k or τ on the test partition after seeing labels.",
            ),
            anti_patterns=(
                "Claiming a 'fraud platform' from batch PR-AUC on a static table.",
            ),
            worked_example_pattern=(
                "fit_anomaly → evaluate_anomaly(label_column='is_fraud') → read "
                "average_precision and precision_at_k.",
            ),
            related_concepts=("anomaly-threshold-alert-rate", "evaluation-partitions"),
        ),
        _note(
            key="anomaly-eda-boundary",
            title="Anomaly product vs EDA IsolationForest",
            summary="EDA IsolationForest is a descriptive screen; Session.fit_anomaly produces a leakage-safe AnomalyPlan with score/flag/evaluate and bundles.",
            definition=(
                "The anomaly product boundary separates descriptive EDA multivariate "
                "IsolationForest screens (and preprocess outlier fences) from the "
                "Session anomaly path that fits a train-only AnomalyPlan, scores "
                "holdout partitions, and persists buildml.anomaly_bundle.v1."
            ),
            intuition=(
                "A dashboard spike chart is a flashlight; a fitted AnomalyPlan is a "
                "calibrated ruler you can reload and apply without refitting."
            ),
            formal_idea=(
                "EDA screens may call IsolationForest on the visible table without a "
                "SplitPlan contract. AnomalyPlan requires assert_fit_partition(train) "
                "and freeze-for-score semantics."
            ),
            why_it_matters=(
                "Promoting EDA screens to production detectors hides leakage and threshold policy.",
                "ClusterPlan labels are structure signals, not anomaly flags — keep APIs separate.",
            ),
            how_buildml_uses=(
                "fit_anomaly / score_anomalies / evaluate_anomaly / anomaly bundles.",
                "Docs and overlays explicitly refuse to equate EDA IF with AnomalyPlan.",
                "handle_outliers remains a preprocess fence path.",
            ),
            interpretation_rules=(
                "Use EDA for teaching/exploration; use fit_anomaly for Session product claims.",
                "Cite bundle format when shipping a detector artifact.",
            ),
            assumptions=(
                "Users understand screens vs plans.",
            ),
            failure_modes=(
                "Exporting EDA anomaly_rate as a production alert policy.",
                "Loading an unsupervised cluster bundle as an anomaly detector.",
            ),
            anti_patterns=(
                "Renaming EDA IsolationForest output columns to is_anomaly without a plan.",
            ),
            worked_example_pattern=(
                "eda() for exploration → separately fit_anomaly after split/scale.",
            ),
            related_concepts=("anomaly-train-fit-holdout-score", "unsupervised-train-fit-holdout-assign"),
        ),
        _note(
            key="anomaly-bundle-boundary",
            title="Anomaly bundle boundary",
            summary="Anomaly plans persist as buildml.anomaly_bundle.v1 — complementary to Session checkpoints and Torch/RAG/unsupervised bundles.",
            definition=(
                "The anomaly bundle boundary is the contract that a train-fitted "
                "AnomalyPlan (estimator, features, threshold, alert-rate disclosures) "
                "is stored under buildml.anomaly_bundle.v1, separate from Session "
                "workflow checkpoints and from Torch/RAG/unsupervised/classical artifacts."
            ),
            intuition=(
                "Saving your notebook (checkpoint) does not shelf the detector; "
                "saving the detector does not restore the dataset."
            ),
            formal_idea=(
                "Bundle layout: meta.json + anomaly_plan.joblib. Session checkpoints "
                "may carry classical preprocess plans but do not embed AnomalyPlan."
            ),
            why_it_matters=(
                "Mixing artifact kinds causes failed loads and false resume expectations.",
                "Threshold and mode disclosures must travel with the estimator.",
            ),
            how_buildml_uses=(
                "save_anomaly_bundle / load_anomaly_bundle on Session.",
                "CHECKPOINT_BOUNDARY string documents complementarity.",
            ),
            interpretation_rules=(
                "After load_anomaly_bundle, rebuild or reload the tabular Session separately if needed.",
                "Confirm feature columns still exist before score_anomalies.",
            ),
            assumptions=(
                "joblib can serialize the sklearn estimator in the plan.",
                "Feature schema at score time matches the plan columns.",
            ),
            failure_modes=(
                "Expecting checkpoint_load to restore fit_anomaly state.",
                "Loading a Torch/RAG/unsupervised bundle as anomaly.",
            ),
            anti_patterns=(
                "Hand-copying only scores without threshold/mode disclosures.",
            ),
            worked_example_pattern=(
                "fit_anomaly → save_anomaly_bundle → Session().load_anomaly_bundle → score_anomalies.",
            ),
            related_concepts=("anomaly-train-fit-holdout-score", "anomaly-eda-boundary"),
        ),
    )
}
