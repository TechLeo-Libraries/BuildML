# ruff: noqa: E501, F401
"""Meta-learning Session operation overlays (human teaching prose)."""

from __future__ import annotations

from buildml.explain.overlays._common import (
    DATASET,
    ROLES,
    SPLIT,
    OperationKind,
    _operation,
    _p,
)
from buildml.explain.schemas import OperationSpec, Prerequisite

METALEARNING_PLAN = Prerequisite(
    "metalearning-plan",
    "A train-fitted MetaLearningPlan is attached to the Session.",
    check_hint="Session.metalearning_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_metalearning",
        OperationKind.MODEL,
        "Meta-train a tabular few-shot / episodic learner on train tasks only.",
        "Resolve task/group column + target; run episodic meta-train (prototypical or warm_start).",
        "Meta-learning fit step.",
        (
            "Require a SplitPlan, exactly one target, and a task/group column.",
            "Carve episodic tasks from the train partition only.",
            "Optionally hold out a fraction of train task ids for internal checks.",
            "Never use validation/test for meta-training.",
        ),
        parameters=(
            _p(
                "method",
                "prototypical | warm_start",
                "Few-shot algorithm path.",
                "prototypical",
            ),
            _p(
                "task_column",
                "str | None",
                "Episodic task id column (else single role='group').",
            ),
            _p("columns", "list[str] | None", "Optional explicit numeric feature columns."),
            _p("n_way", "int | None", "Classes per episode (default = n_classes)."),
            _p("k_shot", "int", "Labeled support examples per class.", 5),
            _p("n_query", "int", "Query examples budget per episode.", 10),
            _p("n_episodes", "int", "Meta-train episodes for disclosure metrics.", 20),
            _p(
                "base_estimator",
                "logistic_regression | sgd_classifier",
                "Warm-start base estimator (ignored for prototypical).",
                "logistic_regression",
            ),
            _p("random_state", "int | None", "RNG seed.", 0),
            _p(
                "prefer_reduce_components",
                "bool",
                "Prefer Session.reduce_dimensions component columns when available.",
                True,
            ),
            _p(
                "task_holdout_fraction",
                "float",
                "Fraction of train task ids held out internally (when >=3 tasks).",
                0.25,
            ),
        ),
        inputs=(
            "Split Session with numeric features, one target, and a task/group column.",
        ),
        outputs=("MetaLearningFitResult; MetaLearningPlan stored on the Session.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After split and usually after impute/scale.",),
        alternatives=(
            "Session.fit for classical single-task learning.",
            "Session.fit_multitask when targets are multiple columns, not task groups.",
        ),
        rationale=(
            "Use when many related tasks share a feature space and few labels per task.",
        ),
        assumptions=(
            "Tasks are identified by a column; labels share a global class set.",
            "Features are numeric and non-null.",
        ),
        failures=(
            "Missing task column, <2 train tasks, insufficient k-shot rows, "
            "unknown method/base_estimator.",
        ),
        leakage=(
            "Meta-training before split contaminates holdout metrics.",
            "Using validation/test rows during meta-train.",
        ),
        anti_patterns=(
            "Claiming MAML-at-scale or foundation-model meta-learning.",
            "Using the task id column as a feature.",
        ),
        state_changes=(
            "Stores metalearning_plan and fit result; clears prior adapt/eval slots.",
        ),
        result_reading=(
            "Read n_meta_train_tasks, meta_train_accuracy, method, disclosures.",
        ),
        next_steps=(
            "adapt_to_task / evaluate_metalearning; optionally save_metalearning_bundle.",
        ),
        concepts=(
            "metalearning-episodic",
            "metalearning-prototypical",
            "metalearning-warm-start",
            "leakage-boundary",
        ),
    ),
    _operation(
        "adapt_to_task",
        OperationKind.MODEL,
        "Fast-adapt the meta-learner to one task's labeled support set.",
        "Build prototypes or refit warm-start init on support rows only.",
        "Meta-learning adapt step.",
        (
            "Require an attached MetaLearningPlan.",
            "Adapt on support only; do not refit the global meta-train plan.",
        ),
        parameters=(
            _p("task_id", "Any | None", "Task id to pull from partition (if no support_frame)."),
            _p(
                "partition",
                "train | validation | test",
                "Partition to pull support rows from when task_id is set.",
                "train",
            ),
            _p(
                "support_frame",
                "DataFrame | None",
                "Optional explicit support rows.",
            ),
            _p(
                "max_support_per_class",
                "int | None",
                "Optional per-class support cap.",
            ),
            _p("random_state", "int | None", "RNG for support capping.", 0),
        ),
        inputs=("Active MetaLearningPlan and labeled support rows.",),
        outputs=("MetaAdaptResult stored on the Session.",),
        prerequisites=(DATASET, METALEARNING_PLAN),
        ordering=("After fit_metalearning or load_metalearning_bundle.",),
        alternatives=("evaluate_metalearning for automatic episodic support/query scoring.",),
        rationale=("Use when a novel task arrives with a small labeled support set.",),
        assumptions=("Support labels are in the plan's known class set.",),
        failures=("No plan; empty support; missing columns; unseen labels.",),
        leakage=("Do not pull support from rows you will also score as query without disclosure.",),
        anti_patterns=("Treating adapt as permission to meta-train on holdout tasks silently.",),
        state_changes=("Stores metalearning_adapt_result.",),
        result_reading=("Read task_id, n_support, n_classes_adapted, disclosures.",),
        next_steps=("evaluate_metalearning or save_metalearning_bundle.",),
        concepts=("metalearning-episodic", "metalearning-prototypical", "metalearning-warm-start"),
    ),
    _operation(
        "evaluate_metalearning",
        OperationKind.DIAGNOSTIC,
        "Evaluate episodic few-shot performance on a holdout partition.",
        "Per-task support/query episodes; prefer novel task ids; aggregate metrics.",
        "Meta-learning evaluation step.",
        (
            "Build episodic support/query splits per selected task.",
            "Prefer novel tasks absent from meta-train.",
            "Disclose when only overlapping task ids are available.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Evaluation partition (validation falls back to test if absent).",
                "validation",
            ),
            _p("k_shot", "int | None", "Override plan k_shot."),
            _p("n_query", "int | None", "Override plan n_query."),
            _p("n_way", "int | None", "Override plan n_way."),
            _p(
                "prefer_novel_tasks",
                "bool",
                "Prefer task ids absent from meta-train.",
                True,
            ),
            _p("random_state", "int | None", "Episode RNG seed.", 0),
        ),
        inputs=("Active MetaLearningPlan and a labeled evaluation partition.",),
        outputs=("MetaLearningEvalResult.",),
        prerequisites=(DATASET, METALEARNING_PLAN),
        ordering=("After fit_metalearning.",),
        alternatives=("Classical Session.evaluate for a single non-episodic model.",),
        rationale=("Use to quantify few-shot generalization across tasks.",),
        assumptions=("Enough per-class rows for support + query on scored tasks.",),
        failures=("No plan; empty partition; all tasks skipped for insufficient rows.",),
        leakage=(
            "Using evaluate_metalearning metrics to choose which holdout rows to meta-train on.",
        ),
        anti_patterns=("Reporting meta_train_accuracy as holdout few-shot performance.",),
        state_changes=("Stores metalearning_eval_result.",),
        result_reading=(
            "Read metrics, per_task_metrics, novel_task_ids, overlapping_task_ids.",
        ),
        next_steps=(
            "save_metalearning_bundle; or compare prototypical vs warm_start.",
        ),
        concepts=(
            "metalearning-episodic",
            "evaluation-partitions",
        ),
    ),
    _operation(
        "save_metalearning_bundle",
        OperationKind.PERSIST,
        "Persist the active MetaLearningPlan as buildml.metalearning_bundle.v1.",
        "Write a domain bundle distinct from Session checkpoints.",
        "Meta-learning bundle save.",
        ("Require an attached plan.", "Write meta.json + metalearning_plan.joblib."),
        parameters=(_p("path", "str | Path", "Destination directory.", required=True),),
        inputs=("Active MetaLearningPlan.",),
        outputs=("Bundle directory path.",),
        prerequisites=(METALEARNING_PLAN,),
        ordering=("After a successful fit_metalearning.",),
        alternatives=("Session.checkpoint_save for workflow resume without the learner.",),
        rationale=("Use when the few-shot meta-learner must travel separately.",),
        assumptions=("Destination is writable.",),
        failures=("No plan attached.",),
        leakage=("Bundles do not embed holdout rows.",),
        anti_patterns=("Assuming a Session checkpoint embeds the MetaLearningPlan.",),
        state_changes=("History records save_metalearning_bundle.",),
        result_reading=("Confirm meta.json format buildml.metalearning_bundle.v1.",),
        next_steps=("load_metalearning_bundle in another Session.",),
        concepts=("metalearning-bundle-boundary",),
    ),
    _operation(
        "load_metalearning_bundle",
        OperationKind.PERSIST,
        "Load a buildml.metalearning_bundle.v1 MetaLearningPlan into the Session.",
        "Restore a meta-learner without re-running meta-train.",
        "Meta-learning bundle load.",
        ("Validate bundle format.", "Attach MetaLearningPlan; clear fit/adapt/eval slots."),
        parameters=(_p("path", "str | Path", "Bundle directory.", required=True),),
        inputs=("Bundle directory with meta.json + metalearning_plan.joblib.",),
        outputs=("Session with metalearning_plan attached.",),
        prerequisites=(DATASET,),
        ordering=("After ingest/roles/split aligned with the plan feature contract.",),
        alternatives=("fit_metalearning to learn a new plan.",),
        rationale=("Use to resume a few-shot meta-learner.",),
        assumptions=("Feature/task/target columns still match the plan contract.",),
        failures=("Incomplete or wrong-format bundle.",),
        leakage=("Do not treat load as permission to meta-train on holdout rows.",),
        anti_patterns=("Loading into a Session whose features drifted from the plan.",),
        state_changes=(
            "Sets metalearning_plan; clears fit/adapt/eval result slots.",
        ),
        result_reading=("Inspect Session.metalearning_plan.to_dict().",),
        next_steps=("adapt_to_task / evaluate_metalearning.",),
        concepts=("metalearning-bundle-boundary",),
    ),
)
