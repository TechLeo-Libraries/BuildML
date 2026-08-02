# ruff: noqa: E501
"""Active learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

ACTIVELEARNING_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="activelearning-train-pool",
            title="Active learning train-only pool",
            summary="The unlabeled query pool is train-partition target missingness (NaN by default) — never validation/test.",
            definition=(
                "Pool-based active learning selects unlabeled *train* rows for a "
                "human to label. BuildML reuses the semi-supervised missingness "
                "contract: NaN/NA/None (or unlabeled_marker) marks pool rows."
            ),
            intuition=(
                "Ask the human about the practice worksheet blanks — never peek at "
                "the exam paper to decide which practice questions to ask."
            ),
            formal_idea=(
                "Pool U subset of train; query q subset of U; holdout H is disjoint "
                "from U for selection — never query validation/test to pick labels."
            ),
            why_it_matters=(
                "Querying test is silent leakage and invalidates holdout claims.",
                "Audits need an explicit pool partition disclosure every round.",
            ),
            how_buildml_uses=(
                "Session.suggest_query scores only train unlabeled indices.",
                "label_rows refuses validation/test indices.",
            ),
            interpretation_rules=(
                "Always read n_unlabeled_pool and that pool=train.",
                "Empty suggestions with budget remaining usually means the pool is empty.",
            ),
            assumptions=("A SplitPlan exists; seed labels exist on train.",),
            failure_modes=(
                "Blanking holdout targets and treating them as the pool.",
                "Using test indices returned by a custom heuristic as queries.",
            ),
            anti_patterns=("Calling the test set 'the unlabeled pool'.",),
            worked_example_pattern=(
                "After split, blank a fraction of train targets, fit_active_learner, suggest_query.",
            ),
            related_concepts=("activelearning-human-labels", "semisupervised-label-missingness"),
        ),
        _note(
            key="activelearning-human-labels",
            title="Human-in-the-loop labels (no oracle in core)",
            summary="BuildML suggests indices; humans (or test harnesses) supply labels. Core never invents an oracle.",
            definition=(
                "suggest_query returns informative indices and uncertainty scores. "
                "label_rows writes user-provided labels onto Session targets and "
                "optionally refits. The library does not look up hidden truths."
            ),
            intuition=(
                "The tutor points at a blank cell; the student (or a test double) "
                "fills it in. The library is not a cheating answer key."
            ),
            formal_idea=(
                "Oracle O: X → Y exists outside the library; AL core only exposes q and accepts O(q)."
            ),
            why_it_matters=(
                "Product honesty: annotation cost is real.",
                "Tests may simulate an oracle; docs must disclose that simulation.",
            ),
            how_buildml_uses=(
                "Session.label_rows requires concrete labels aligned to indices.",
                "Disclosures on fit/query/label state that labels are user-supplied.",
            ),
            interpretation_rules=(
                "Read n_queries_used and label_budget every round.",
                "Budget exhaustion returns empty suggestions rather than inventing labels.",
            ),
            assumptions=("A labeling process exists outside BuildML core.",),
            failure_modes=("Expecting suggest_query to return labels.",),
            anti_patterns=("Shipping a silent oracle that reads holdout truths in production code.",),
            worked_example_pattern=(
                "q = session.suggest_query(batch_size=5); session.label_rows(indices=q.indices, labels=human_labels).",
            ),
            related_concepts=("activelearning-train-pool", "activelearning-uncertainty"),
        ),
        _note(
            key="activelearning-uncertainty",
            title="Uncertainty and committee query strategies",
            summary="least_confidence / margin / entropy use predict_proba; committee uses bagged vote entropy; expected_model_change_lite is a gradient-magnitude proxy.",
            definition=(
                "Uncertainty sampling ranks unlabeled points by model doubt. "
                "Query-by-committee ranks by disagreement among bagged clones. "
                "expected_model_change_lite scores ||x||(1-p_max) as a lite proxy."
            ),
            intuition=(
                "Ask about the examples the model is least sure of — or where a "
                "committee of clones argue the loudest."
            ),
            formal_idea=(
                "LC: 1-max p; margin: -(p_(1)-p_(2)); entropy: -∑ p log p; "
                "QBC: vote entropy; EMC-lite: ||x||(1-p_max)."
            ),
            why_it_matters=(
                "Strategy choice changes which rows burn the annotation budget.",
            ),
            how_buildml_uses=(
                "Session.fit_active_learner(strategy=...); Session.suggest_query(...).",
            ),
            interpretation_rules=(
                "Higher returned scores mean higher priority under the chosen strategy.",
                "Committee requires strategy='committee' so a bagged committee is fitted.",
            ),
            assumptions=("Base estimator supports predict_proba for uncertainty strategies.",),
            failure_modes=("Using committee strategy without a fitted committee_.",),
            anti_patterns=("Random sampling while claiming uncertainty sampling.",),
            worked_example_pattern=(
                "fit_active_learner(strategy='entropy') → suggest_query(batch_size=10).",
            ),
            related_concepts=("activelearning-human-labels", "activelearning-bundle-boundary"),
        ),
        _note(
            key="activelearning-bundle-boundary",
            title="Active-learning bundle boundary",
            summary="buildml.activelearning_bundle.v1 stores the plan (model + pool indices + query history); Session checkpoints do not embed it.",
            definition=(
                "An active-learning bundle persists the estimator, label encoder, "
                "labeled/pool index contract, query history, and budget. Session "
                "checkpoints persist data/roles/splits/history/preprocess plans — "
                "not the ActiveLearningPlan weights."
            ),
            intuition=(
                "Saving the notebook resume is not the same as saving the learner "
                "and its labeling ledger."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ active learner; "
                "load_active_learning_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-learner failures.",),
            how_buildml_uses=(
                "save_active_learning_bundle / load_active_learning_bundle.",
            ),
            interpretation_rules=("Read meta.json format buildml.activelearning_bundle.v1.",),
            assumptions=("Feature contract and target role still match at load time.",),
            failure_modes=("Expecting checkpoint_load to restore ActiveLearningPlan.",),
            anti_patterns=("Treating semisupervised bundles as active-learning plans.",),
            worked_example_pattern=(
                "session.save_active_learning_bundle(path); other.load_active_learning_bundle(path).",
            ),
            related_concepts=("activelearning-train-pool", "semisupervised-bundle-boundary"),
        ),
    )
}
