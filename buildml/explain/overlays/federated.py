# ruff: noqa: E501, F401
"""Federated learning Session operation overlays (human teaching prose)."""

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

FEDERATED_PLAN = Prerequisite(
    "federated-plan",
    "A train-fitted FederatedPlan is attached to the Session.",
    check_hint="Session.federated_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_federated",
        OperationKind.MODEL,
        "Simulate federated averaging on Session train clients.",
        "Partition train by client/group column; local updates; aggregate coef_/intercept_.",
        "Federated fit step (local simulation).",
        (
            "Require a SplitPlan, exactly one target, and a client/group column.",
            "Run local updates on each selected client's train rows only.",
            "Aggregate coefficients with sample-size weights (FedAvg) or FedProx pull.",
            "Never use validation/test for local training.",
        ),
        parameters=(
            _p(
                "backend",
                "native | flower | None",
                "Federation backend (defaults to flower when flwr installed).",
                None,
            ),
            _p(
                "method",
                "fedavg | fedprox",
                "Aggregation path (fedprox requires mu > 0).",
                "fedavg",
            ),
            _p(
                "estimator",
                "sgd_classifier | sgd_regressor | logistic_regression | ridge | linear_regression",
                "Linear/SGD family with coef_/intercept_.",
                "sgd_classifier",
            ),
            _p(
                "task",
                "classification | regression | None",
                "Inferred from estimator when omitted.",
            ),
            _p(
                "client_column",
                "str | None",
                "Client id column (else single role='group').",
            ),
            _p("columns", "list[str] | None", "Optional explicit numeric feature columns."),
            _p("n_rounds", "int", "Server aggregation rounds.", 5),
            _p("local_epochs", "int", "Local passes per selected client per round.", 1),
            _p("client_fraction", "float", "Fraction of clients sampled each round.", 1.0),
            _p("mu", "float", "FedProx proximal strength (required > 0 for fedprox).", 0.0),
            _p("random_state", "int | None", "RNG seed for client sampling.", 0),
            _p(
                "prefer_reduce_components",
                "bool",
                "Prefer reduce components when a ReducePlan is attached.",
                True,
            ),
            _p(
                "min_client_rows",
                "int",
                "Minimum train rows for a client to participate.",
                2,
            ),
        ),
        inputs=("Session dataset with split, target, and client/group column.",),
        outputs=("FederatedFitResult; Session.federated_plan attached.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=("After ingest → set_roles → split → optional scale/reduce.",),
        alternatives=(
            "fit_online for single-stream partial_fit; fit for centralized pooling.",
        ),
        rationale=(
            "Use to study client-partitioned training and FedAvg aggregation "
            "without a network stack.",
        ),
        assumptions=(
            "Compatible linear/SGD estimators; numeric non-null features; "
            ">= 2 eligible clients.",
        ),
        failures=(
            "No split; missing client/group column; fewer than two eligible clients.",
        ),
        leakage=(
            "Using validation/test rows in local client updates.",
            "Feeding the client id column as a model feature.",
        ),
        anti_patterns=(
            "Silent centralized .fit on all clients while claiming federated learning.",
            "Advertising secure aggregation without implementing it.",
        ),
        state_changes=(
            "Stores federated_plan and fit result; clears prior eval/predict slots.",
        ),
        result_reading=(
            "Read n_clients, round_history, final_train_metric, disclosures.",
        ),
        next_steps=(
            "evaluate_federated → predict_federated; optionally save_federated_bundle.",
        ),
        concepts=(
            "federated-simulation",
            "federated-fedavg",
            "federated-fedprox",
            "federated-flower-backend",
            "leakage-boundary",
        ),
    ),
    _operation(
        "evaluate_federated",
        OperationKind.DIAGNOSTIC,
        "Evaluate the global federated model on a holdout partition.",
        "Score global (+ optional per-client) metrics; never for training.",
        "Federated holdout evaluation.",
        (
            "Require an attached FederatedPlan.",
            "Score the chosen partition with the global estimator.",
            "Optionally slice metrics by client id on the holdout frame.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Evaluation partition (validation falls back to test if absent).",
                "validation",
            ),
            _p(
                "per_client",
                "bool",
                "Also report per-client holdout metrics.",
                True,
            ),
        ),
        inputs=("FederatedPlan + holdout partition.",),
        outputs=("FederatedEvalResult with global and optional per-client metrics.",),
        prerequisites=(DATASET, FEDERATED_PLAN),
        ordering=("After fit_federated or load_federated_bundle.",),
        alternatives=("predict_federated when you only need labels.",),
        rationale=("Use to quantify global (+ client-sliced) holdout performance.",),
        assumptions=("Feature/target columns match the plan contract.",),
        failures=("No plan; empty partition; missing columns.",),
        leakage=(
            "Using evaluate_federated metrics to choose which holdout clients to retrain on.",
        ),
        anti_patterns=(
            "Reporting per-client train metrics as privacy-preserving FL results.",
        ),
        state_changes=("Stores federated_eval_result.",),
        result_reading=(
            "Read metrics, per_client_metrics, n_clients_evaluated, disclosures.",
        ),
        next_steps=("save_federated_bundle; or predict_federated.",),
        concepts=(
            "federated-simulation",
            "federated-fedavg",
            "evaluation-partitions",
        ),
    ),
    _operation(
        "predict_federated",
        OperationKind.DIAGNOSTIC,
        "Predict with the global federated estimator (no update).",
        "Score a partition without advancing federation state.",
        "Federated predict step.",
        (
            "Require an attached FederatedPlan.",
            "Run estimator.predict on the partition features.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Partition to score.",
                "test",
            ),
        ),
        inputs=("Active FederatedPlan.",),
        outputs=("FederatedPredictResult.",),
        prerequisites=(DATASET, FEDERATED_PLAN),
        ordering=("After fit_federated / evaluate_federated.",),
        alternatives=("evaluate_federated when you also need metrics.",),
        rationale=("Use for inference snapshots without mutating federation state.",),
        assumptions=("Feature columns match the plan contract.",),
        failures=("No plan; missing columns; null features.",),
        leakage=("None inherent — still do not train on predictions.",),
        anti_patterns=("Calling predict a federated round.",),
        state_changes=("Stores federated_predict_result.",),
        result_reading=("Read n_rows / n_predictions and disclosures.",),
        next_steps=("evaluate_federated or save_federated_bundle.",),
        concepts=("federated-simulation",),
    ),
    _operation(
        "save_federated_bundle",
        OperationKind.PERSIST,
        "Persist the active FederatedPlan as buildml.federated_bundle.v1.",
        "Write a domain bundle distinct from Session checkpoints.",
        "Federated bundle save.",
        ("Require an attached plan.", "Write meta.json + federated_plan.joblib."),
        parameters=(_p("path", "str | Path", "Destination directory.", required=True),),
        inputs=("Active FederatedPlan.",),
        outputs=("Bundle directory path.",),
        prerequisites=(FEDERATED_PLAN,),
        ordering=("After a successful fit_federated.",),
        alternatives=("Session.checkpoint_save for workflow resume without the learner.",),
        rationale=("Use when the global federated model must travel separately.",),
        assumptions=("Destination is writable.",),
        failures=("No plan attached.",),
        leakage=("Bundles do not embed holdout rows.",),
        anti_patterns=("Assuming a Session checkpoint embeds the FederatedPlan.",),
        state_changes=("History records save_federated_bundle.",),
        result_reading=("Confirm meta.json format buildml.federated_bundle.v1.",),
        next_steps=("load_federated_bundle in another Session.",),
        concepts=("federated-bundle-boundary",),
    ),
    _operation(
        "load_federated_bundle",
        OperationKind.PERSIST,
        "Load a buildml.federated_bundle.v1 FederatedPlan into the Session.",
        "Restore a global federated model without re-running rounds.",
        "Federated bundle load.",
        (
            "Validate bundle format.",
            "Attach FederatedPlan; clear fit/eval/predict slots.",
        ),
        parameters=(_p("path", "str | Path", "Bundle directory.", required=True),),
        inputs=("Bundle directory with meta.json + federated_plan.joblib.",),
        outputs=("Session with federated_plan attached.",),
        prerequisites=(DATASET,),
        ordering=("After ingest/roles/split aligned with the plan feature contract.",),
        alternatives=("fit_federated to learn a new plan.",),
        rationale=("Use to resume a federated global model.",),
        assumptions=("Feature/client/target columns still match the plan contract.",),
        failures=("Incomplete or wrong-format bundle.",),
        leakage=("Do not treat load as permission to train on holdout rows.",),
        anti_patterns=("Loading into a Session whose features drifted from the plan.",),
        state_changes=(
            "Sets federated_plan; clears fit/eval/predict result slots.",
        ),
        result_reading=("Inspect Session.federated_plan.to_dict().",),
        next_steps=("evaluate_federated / predict_federated.",),
        concepts=("federated-bundle-boundary",),
    ),
)
