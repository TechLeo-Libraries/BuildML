# ruff: noqa: E501
"""Semi-supervised learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

SEMISUPERVISED_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="semisupervised-label-missingness",
            title="Semi-supervised label missingness",
            summary="Unlabeled rows are target missingness (NaN by default), mapped internally to sklearn's -1 convention — not a separate mystery role.",
            definition=(
                "BuildML treats scarce labels via the Session target role: missing "
                "values (NaN/NA/None by default, or an explicit unlabeled_marker) "
                "mark unlabeled rows. Features stay present; only the label is absent."
            ),
            intuition=(
                "Think of a spreadsheet where most answer cells are blank but every "
                "row still has features. Propagation and self-training fill train "
                "blanks carefully; holdout blanks are never fake exam keys."
            ),
            formal_idea=(
                "Let y_i ∈ Y ∪ {∅}. Encode ∅ → -1 for sklearn semi-supervised "
                "estimators; labeled rows keep a LabelEncoder mapping over observed classes."
            ),
            why_it_matters=(
                "Clear missingness prevents silent leakage from inventing holdout labels.",
                "Teaching and audits need an explicit unlabeled count beside every fit/eval.",
            ),
            how_buildml_uses=(
                "Session.fit_semisupervised reads the target role and counts labeled/unlabeled train rows.",
                "evaluate_semisupervised scores only labeled partition rows.",
            ),
            interpretation_rules=(
                "Always read n_labeled_train / n_unlabeled_train and n_labeled_eval.",
                "Empty metrics with n_labeled_eval=0 means the partition had no ground truth.",
            ),
            assumptions=(
                "At least two labeled train rows spanning ≥2 classes.",
                "Unlabeled marker policy is stable between fit and evaluate.",
            ),
            failure_modes=(
                "Stratifying a split on a mostly-missing target without a labeled-first recipe.",
                "Using -1 as a real class while also using -1 as unlabeled without disclosure.",
            ),
            anti_patterns=(
                "Silently writing pseudo-labels into the Session target role.",
            ),
            worked_example_pattern=(
                "After split, set a fraction of train targets to NaN, then fit_semisupervised.",
            ),
            related_concepts=("semisupervised-train-only-fit", "ssl-pretext-then-head"),
        ),
        _note(
            key="semisupervised-train-only-fit",
            title="Semi-supervised train-only fit",
            summary="Propagate or self-train using train rows only; freeze the plan; evaluate labeled holdout without inventing selection labels.",
            definition=(
                "Semi-supervised fit learns from labeled + unlabeled *train* rows, "
                "freezes a SemiSupervisedPlan, and scores validation/test without "
                "updating that plan or treating unlabeled holdout as truth."
            ),
            intuition=(
                "Practice with the worksheet you were given — including blank answers "
                "on that worksheet — then take the exam. Do not peek at exam blanks "
                "to rewrite the worksheet."
            ),
            formal_idea=(
                "Fit f on (X_train, y_train^{(partial)}); for partition P predict "
                "ŷ_P = f(X_P). Metrics use {(x,y) ∈ P : y ≠ ∅} only."
            ),
            why_it_matters=(
                "Inventing labels on validation for model selection is silent leakage.",
                "Pseudo-label accuracy on train is not holdout performance.",
            ),
            how_buildml_uses=(
                "Session.fit_semisupervised asserts a train SplitPlan.",
                "predict_semisupervised / evaluate_semisupervised reuse the frozen plan.",
            ),
            interpretation_rules=(
                "Prefer validation/test labeled metrics over train pseudo-label accuracy.",
                "Read disclosures about n_unlabeled_train every time.",
            ),
            assumptions=("A disjoint SplitPlan exists before fit.",),
            failure_modes=(
                "Fitting on the full frame before splitting.",
                "Using unlabeled test rows as if they were labeled for selection.",
            ),
            anti_patterns=(
                "Reporting train pseudo-label accuracy as holdout performance.",
            ),
            worked_example_pattern=(
                "fit_semisupervised(method='self_training') → evaluate_semisupervised(partition='test').",
            ),
            related_concepts=("semisupervised-label-missingness", "leakage-boundary"),
        ),
        _note(
            key="semisupervised-vs-novelty",
            title="Semi-supervised vs anomaly novelty",
            summary="Label propagation/self-training is scarce-label classification; anomaly novelty is normal-only detector fit — different Session paths.",
            definition=(
                "Phase-2 semi-supervised learning expands scarce class labels across "
                "unlabeled train features. Anomaly novelty mode fits a detector on a "
                "normal-only train subset and scores anomalies — it is not label spreading."
            ),
            intuition=(
                "One path fills blank class stickers; the other learns 'what normal looks "
                "like' and flags strangers. Do not swap the metaphors."
            ),
            formal_idea=(
                "Semi-supervised: y ∈ {1..K, ∅}. Novelty: fit on {i : y_i = normal}, "
                "score unusualness on all partitions."
            ),
            why_it_matters=(
                "Conflating the APIs produces wrong metrics and wrong product claims.",
            ),
            how_buildml_uses=(
                "Session.fit_semisupervised vs Session.fit_anomaly(mode='novelty').",
            ),
            interpretation_rules=(
                "If the goal is class labels under scarce annotation, use semi-supervised.",
                "If the goal is alert rates / anomaly scores, use the anomaly path.",
            ),
            assumptions=("Problem framing is chosen before API selection.",),
            failure_modes=("Calling novelty 'semi-supervised representation learning'.",),
            anti_patterns=("Reusing anomaly novelty APIs for scarce multiclass labels.",),
            worked_example_pattern=(
                "Fraud class labels scarce → fit_semisupervised; normal-only stream → fit_anomaly novelty.",
            ),
            related_concepts=("anomaly-novelty-vs-unsupervised", "semisupervised-train-only-fit"),
        ),
        _note(
            key="semisupervised-bundle-boundary",
            title="Semi-supervised bundle boundary",
            summary="buildml.semisupervised_bundle.v1 stores the plan; Session checkpoints do not embed it.",
            definition=(
                "A semi-supervised bundle persists the train-fitted estimator and label "
                "contract. Session checkpoints persist data/roles/splits/history/plans "
                "for preprocess — not the SemiSupervisedPlan weights."
            ),
            intuition=(
                "The notebook resume is not the trained teacher. Save the teacher bundle "
                "when the estimator must travel."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ semisupervised estimator; "
                "load_semisupervised_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-estimator failures.",),
            how_buildml_uses=(
                "save_semisupervised_bundle / load_semisupervised_bundle.",
            ),
            interpretation_rules=("Read meta.json format buildml.semisupervised_bundle.v1.",),
            assumptions=("Feature contract still matches at load time.",),
            failure_modes=("Expecting checkpoint_load to restore SemiSupervisedPlan.",),
            anti_patterns=("Treating unsupervised or anomaly bundles as semi-supervised plans.",),
            worked_example_pattern=(
                "session.save_semisupervised_bundle(path); other.load_semisupervised_bundle(path).",
            ),
            related_concepts=("semisupervised-train-only-fit",),
        ),
    )
}
