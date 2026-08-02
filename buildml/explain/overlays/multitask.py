# ruff: noqa: E501, F401
"""Multi-task / multi-output Session operation overlays (human teaching prose)."""

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

MULTITASK_PLAN = Prerequisite(
    "multitask-plan",
    "A train-fitted MultiTaskPlan is attached to the Session.",
    check_hint="Session.multitask_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_multitask",
        OperationKind.MODEL,
        "Fit a multi-target estimator on train only (sklearn / industry / torch).",
        "Resolve ≥2 targets, route backend=, infer task kinds, fit on train.",
        "Multi-task learning fit step.",
        (
            "Require a SplitPlan and at least two target columns (roles or targets=).",
            "Sklearn/industry: refuse mixed classification+regression targets.",
            "Torch shared_trunk_multihead: joint training with per-task heads.",
            "Never use validation/test for fitting.",
            "Leave classical Session.fit single-target semantics unchanged.",
        ),
        parameters=(
            _p(
                "backend",
                "sklearn | industry | torch | None",
                "Backend router; defaults to industry/torch/sklearn when installed.",
            ),
            _p(
                "method",
                "multi_output | classifier_chain | regressor_chain | "
                "multi_output_xgb | multi_output_lgbm | multi_output_catboost | "
                "shared_trunk_multihead",
                "Method within the selected backend.",
                "multi_output",
            ),
            _p(
                "task",
                "classification | regression | auto | mixed",
                "Task type; mixed only on torch shared_trunk_multihead.",
                "auto",
            ),
            _p(
                "targets",
                "list[str] | None",
                "Optional explicit target columns (else all role='target').",
            ),
            _p("columns", "list[str] | None", "Optional explicit numeric feature columns."),
            _p(
                "base_estimator",
                "logistic_regression | hist_gradient_boosting | ridge | "
                "hist_gradient_boosting_regressor",
                "Base estimator inside the multi-output / chain wrapper.",
                "logistic_regression",
            ),
            _p("random_state", "int | None", "RNG seed where applicable.", 0),
            _p(
                "order",
                "list[str] | None",
                "Optional chain order as a permutation of target column names.",
            ),
            _p(
                "prefer_reduce_components",
                "bool",
                "Prefer Session.reduce_dimensions component columns when available.",
                True,
            ),
            _p(
                "prediction_prefix",
                "str",
                "Prefix for attachable prediction columns.",
                "multitask_pred",
            ),
            _p("epochs", "int", "Torch training epochs.", 60),
            _p("batch_size", "int", "Torch mini-batch size.", 64),
            _p("learning_rate", "float", "Torch AdamW learning rate.", 1e-3),
            _p("device", "str", "Torch device string.", "cpu"),
        ),
        inputs=(
            "Split Session with numeric features and ≥2 same-type target columns.",
        ),
        outputs=("MultiTaskFitResult; MultiTaskPlan stored on the Session.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After split and usually after impute/scale.",),
        alternatives=(
            "Session.fit for a single classical target.",
            "Session.fit_voting / fit_stacking when combining models, not targets.",
        ),
        rationale=(
            "Use when several related targets share features and should be learned jointly.",
        ),
        assumptions=(
            "Sklearn/industry: all classification or all regression.",
            "Torch mixed: per-task heads with honest joint loss.",
            "Features are numeric and non-null.",
        ),
        failures=(
            "Fewer than 2 targets, mixed task kinds, null features/targets, "
            "unknown method/base_estimator, chain order mismatch.",
        ),
        leakage=(
            "Fitting before split contaminates holdout metrics.",
            "Using validation/test rows during fit.",
        ),
        anti_patterns=(
            "Claiming deep multi-head MTL from MultiOutput wrappers.",
            "Expecting Session.fit to auto-enable multi-output.",
        ),
        state_changes=(
            "Stores multitask_plan and fit result; clears prior predict/eval slots.",
        ),
        result_reading=(
            "Read n_tasks, target_columns, method, task, disclosures.",
        ),
        next_steps=(
            "predict_multitask / evaluate_multitask; optionally save_multitask_bundle.",
        ),
        concepts=(
            "multitask-multi-output",
            "multitask-target-roles",
            "multitask-chain",
            "leakage-boundary",
        ),
    ),
    _operation(
        "predict_multitask",
        OperationKind.DIAGNOSTIC,
        "Predict per-task outputs with the frozen multi-task plan (no refit).",
        "Score a partition; optionally attach {prefix}_{task} feature columns.",
        "Multi-task predict step.",
        (
            "Require an attached MultiTaskPlan.",
            "Predict without refitting.",
            "attach=True only with partition='all'.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Prediction partition.",
                "test",
            ),
            _p("attach", "bool", "Attach prediction columns to the Session frame.", False),
            _p(
                "prediction_prefix",
                "str | None",
                "Override plan prediction_prefix for attached columns.",
            ),
        ),
        inputs=("Active MultiTaskPlan.",),
        outputs=("MultiTaskPredictResult (and updated dataset when attach=True).",),
        prerequisites=(DATASET, MULTITASK_PLAN),
        ordering=("After fit_multitask or load_multitask_bundle.",),
        alternatives=("evaluate_multitask when you also need metrics.",),
        rationale=("Use for per-task inference snapshots without mutating the plan.",),
        assumptions=("Feature columns match the plan contract.",),
        failures=("No plan; missing columns; null features; attach without partition='all'.",),
        leakage=("None inherent — still do not train on predictions.",),
        anti_patterns=("Writing predictions back into target roles without disclosure.",),
        state_changes=("Stores multitask_predict_result; may replace dataset when attach=True.",),
        result_reading=("Read predictions dict keyed by target column; n_rows; disclosures.",),
        next_steps=("evaluate_multitask or save_multitask_bundle.",),
        concepts=("multitask-multi-output",),
    ),
    _operation(
        "evaluate_multitask",
        OperationKind.DIAGNOSTIC,
        "Evaluate multi-task predictions with per-task and aggregate metrics.",
        "Score holdout accuracy/F1 or MAE/RMSE/R² per task; mean aggregates.",
        "Multi-task evaluation step.",
        (
            "Predict with the frozen plan.",
            "Score a holdout partition never used for fitting.",
            "Report per_task_metrics and unweighted mean aggregates.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Evaluation partition (validation falls back to test if absent).",
                "validation",
            ),
        ),
        inputs=("Active MultiTaskPlan and labeled evaluation targets.",),
        outputs=("MultiTaskEvalResult.",),
        prerequisites=(DATASET, MULTITASK_PLAN),
        ordering=("After fit_multitask.",),
        alternatives=("Classical Session.evaluate for a single-target FitResult.",),
        rationale=("Use to quantify holdout quality across all tasks.",),
        assumptions=("Holdout has labeled rows for every target column.",),
        failures=("No plan; empty/null evaluation partition.",),
        leakage=(
            "Using evaluate_multitask metrics to choose which holdout rows to refit on.",
        ),
        anti_patterns=("Reporting train accuracy as holdout multi-task performance.",),
        state_changes=("Stores multitask_eval_result.",),
        result_reading=(
            "Read per_task_metrics, metrics (means), n_rows, disclosures.",
        ),
        next_steps=(
            "save_multitask_bundle; or compare method='multi_output' vs chains.",
        ),
        concepts=(
            "multitask-multi-output",
            "evaluation-partitions",
        ),
    ),
    _operation(
        "save_multitask_bundle",
        OperationKind.PERSIST,
        "Persist the active MultiTaskPlan as buildml.multitask_bundle.v1.",
        "Write a domain bundle distinct from Session checkpoints.",
        "Multi-task bundle save.",
        ("Require an attached plan.", "Write meta.json + multitask_plan.joblib."),
        parameters=(_p("path", "str | Path", "Destination directory.", required=True),),
        inputs=("Active MultiTaskPlan.",),
        outputs=("Bundle directory path.",),
        prerequisites=(MULTITASK_PLAN,),
        ordering=("After a successful fit_multitask.",),
        alternatives=("Session.checkpoint_save for workflow resume without the learner.",),
        rationale=("Use when the joint multi-target model must travel separately.",),
        assumptions=("Destination is writable.",),
        failures=("No plan attached.",),
        leakage=("Bundles do not embed holdout rows.",),
        anti_patterns=("Assuming a Session checkpoint embeds the MultiTaskPlan.",),
        state_changes=("History records save_multitask_bundle.",),
        result_reading=("Confirm meta.json format buildml.multitask_bundle.v1.",),
        next_steps=("load_multitask_bundle in another Session.",),
        concepts=("multitask-bundle-boundary",),
    ),
    _operation(
        "load_multitask_bundle",
        OperationKind.PERSIST,
        "Load a buildml.multitask_bundle.v1 MultiTaskPlan into the Session.",
        "Restore a multi-output / chain learner without refitting.",
        "Multi-task bundle load.",
        ("Validate bundle format.", "Attach MultiTaskPlan; clear fit/predict/eval slots."),
        parameters=(_p("path", "str | Path", "Bundle directory.", required=True),),
        inputs=("Bundle directory with meta.json + multitask_plan.joblib.",),
        outputs=("Session with multitask_plan attached.",),
        prerequisites=(DATASET,),
        ordering=("After ingest/roles/split aligned with the plan feature contract.",),
        alternatives=("fit_multitask to learn a new plan.",),
        rationale=("Use to resume a multi-target learner.",),
        assumptions=("Feature columns and target columns still match the plan contract.",),
        failures=("Incomplete or wrong-format bundle.",),
        leakage=("Do not treat load as permission to fit on holdout rows.",),
        anti_patterns=("Loading into a Session whose features drifted from the plan.",),
        state_changes=(
            "Sets multitask_plan; clears fit/predict/eval result slots.",
        ),
        result_reading=("Inspect Session.multitask_plan.to_dict().",),
        next_steps=("predict_multitask / evaluate_multitask.",),
        concepts=("multitask-bundle-boundary",),
    ),
)
