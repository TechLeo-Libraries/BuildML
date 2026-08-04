# ruff: noqa: E501
"""Multi-task / multi-output learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

MULTITASK_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="multitask-multi-output",
            title="Multi-output / multi-task on shared features",
            summary="Sklearn MultiOutput/Chain, industry GBDT multi-target, or torch shared-trunk multi-head: not a deep MTL research platform.",
            definition=(
                "Multi-task learning in BuildML fits multiple targets that share "
                "the same feature matrix. Backends: sklearn MultiOutput/Chain "
                "(core), industry XGBoost/LightGBM/CatBoost multi-target "
                "(multitask-industry extra), torch shared-trunk multi-head "
                "(torch extra; mixed cls+reg via separate heads). Classical "
                "Session.fit remains single-target."
            ),
            intuition=(
                "One practice binder with several answer columns graded by "
                "related but separate rubrics: not a research lab inventing "
                "new shared neural heads."
            ),
            formal_idea=(
                "Ŷ = f_θ(X) with Y ∈ R^{n×T} (or label codes); θ from "
                "MultiOutput / Chain; train only; holdout H for metrics."
            ),
            why_it_matters=(
                "Treating multi-target as accidental single-target fit drops tasks silently.",
                "Using holdout rows to fit is leakage.",
            ),
            how_buildml_uses=(
                "session.multitask.fit(backend=..., method=...) → predict/evaluate.",
                "See session.multitask.capability_matrix() for honest defaults.",
            ),
            interpretation_rules=(
                "Read n_tasks, target_columns, method, task, and disclosures.",
                "session.multitask.evaluate reports per_task_metrics plus unweighted means.",
            ),
            assumptions=("At least two same-type targets; numeric non-null features.",),
            failure_modes=(
                "Passing a single target and expecting classical fit semantics.",
                "Mixing continuous regression with categorical classification targets.",
            ),
            anti_patterns=(
                "Claiming deep multi-head MTL from sklearn MultiOutput wrappers.",
            ),
            worked_example_pattern=(
                "session.multitask.fit(method='multi_output') → session.multitask.evaluate('validation').",
            ),
            related_concepts=("multitask-chain", "multitask-target-roles", "leakage-boundary"),
        ),
        _note(
            key="multitask-chain",
            title="ClassifierChain / RegressorChain ordering",
            summary="Chains model task dependence by feeding earlier predictions into later estimators.",
            definition=(
                "ClassifierChain and RegressorChain fit one estimator per target "
                "in a declared order, appending previous-task predictions as "
                "features for later tasks. order= must permute the target columns."
            ),
            intuition=(
                "Grade question 1 first, then let that grade inform question 2: "
                "order matters when tasks are dependent."
            ),
            formal_idea=(
                "ŷ₁ = f₁(X); ŷ₂ = f₂(X, ŷ₁); … along a permutation π of tasks."
            ),
            why_it_matters=(
                "Independent MultiOutput ignores label dependence that chains can capture.",
            ),
            how_buildml_uses=(
                "session.multitask.fit(method='classifier_chain'|'regressor_chain', order=...).",
            ),
            interpretation_rules=(
                "Read the chain-order disclosure on the plan; compare to multi_output.",
            ),
            assumptions=("Targets are all classification or all regression.",),
            failure_modes=("Passing order= that is not a permutation of targets.",),
            anti_patterns=("Using chains for independent targets without checking whether dependence exists.",),
            worked_example_pattern=(
                "session.multitask.fit(method='classifier_chain', order=['t1', 't2']).",
            ),
            related_concepts=("multitask-multi-output", "multitask-target-roles"),
        ),
        _note(
            key="multitask-target-roles",
            title="Multiple target roles vs classical require_target",
            summary="Multi-task needs ≥2 targets; classical Session.fit still requires exactly one.",
            definition=(
                "Assign multiple columns role='target' (or pass targets=) for "
                "session.multitask.fit. Classical require_target() still enforces a "
                "single target for Session.fit: the paths are distinct."
            ),
            intuition=(
                "Two graded columns need the multi-task desk; the single-target "
                "desk still rejects a stack of answer sheets."
            ),
            formal_idea=(
                "|targets| ≥ 2 for multitask; |targets| = 1 for classical fit."
            ),
            why_it_matters=(
                "Collapsing multiple targets into one silently drops tasks.",
            ),
            how_buildml_uses=(
                "set_roles({..., 't1': 'target', 't2': 'target'}) then session.multitask.fit.",
            ),
            interpretation_rules=(
                "If classical fit raises 'Expected exactly one target', use session.multitask.fit or drop extras.",
            ),
            assumptions=("Target columns exist on the Session frame.",),
            failure_modes=("Calling session.multitask.fit with only one target role.",),
            anti_patterns=("Expecting Session.fit to auto-switch into multi-output mode.",),
            worked_example_pattern=(
                "set_roles({'x': 'feature', 't1': 'target', 't2': 'target'})",
            ),
            related_concepts=("multitask-multi-output", "multitask-bundle-boundary"),
        ),
        _note(
            key="multitask-bundle-boundary",
            title="Multi-task bundle boundary",
            summary="buildml.multitask_bundle.v1 stores MultiTaskPlan; Session checkpoints do not embed it.",
            definition=(
                "A multi-task bundle persists the multi-output / chain estimator, "
                "target contract, and per-task label encoders. Session "
                "checkpoints persist data/roles/splits/history: not MultiTaskPlan."
            ),
            intuition=(
                "Saving the binder resume is not the same as saving the joint "
                "multi-target model."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ multitask learner; "
                "session.multitask.load_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-learner failures.",),
            how_buildml_uses=("session.multitask.save_bundle / session.multitask.load_bundle.",),
            interpretation_rules=("Read meta.json format buildml.multitask_bundle.v1.",),
            assumptions=("Feature contract and target columns still match at load time.",),
            failure_modes=("Expecting checkpoint_load to restore MultiTaskPlan.",),
            anti_patterns=("Treating online or classical pipeline bundles as multi-task plans.",),
            worked_example_pattern=(
                "session.multitask.save_bundle(path); other.multitask.load_bundle(path).",
            ),
            related_concepts=("multitask-multi-output", "online-bundle-boundary"),
        ),
    )
}
