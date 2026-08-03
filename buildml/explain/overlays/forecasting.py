# ruff: noqa: E501, F401
"""Forecasting Session operation overlays (human teaching prose)."""

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

FORECAST_PLAN = Prerequisite(
    "forecast-plan",
    "A train-fitted ForecastPlan is attached to the Session.",
    check_hint="Session.forecast_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_forecast",
        OperationKind.MODEL,
        "Fit a classical forecaster on the train partition and store a ForecastPlan.",
        "Learn lag/baseline structure without using future holdout targets as features.",
        "Classical forecasting fit step.",
        (
            "Require a temporal SplitPlan (time_split or chronological inject_split).",
            "Refuse random/stratified/group splits as leakage.",
            "Order train by the time-role column; build lag features from past targets only.",
            "Fit naive/seasonal_naive/drift/mean baselines or lag_ridge/lag_hgb models.",
        ),
        parameters=(
            _p(
                "method",
                "naive | seasonal_naive | drift | mean | lag_ridge | lag_hgb",
                "Forecast algorithm.",
                "lag_ridge",
            ),
            _p("horizon", "int", "Default generate horizon stored on the plan.", 1),
            _p("lags", "list[int] | None", "Positive lag orders.", [1, 2, 3, 7]),
            _p("seasonal_period", "int | None", "Season length for seasonal_naive."),
            _p("exog_columns", "list[str] | None", "Optional numeric exogenous columns."),
            _p("target_column", "str | None", "Override; defaults to the target role."),
            _p("time_column", "str | None", "Override; defaults to the time role."),
            _p("random_state", "int | None", "RNG seed for lag models.", 0),
            _p("alpha", "float", "Ridge alpha for lag_ridge.", 1.0),
            _p("max_iter", "int", "HGB max_iter for lag_hgb.", 100),
            _p("max_depth", "int | None", "HGB max_depth for lag_hgb.", 3),
            _p("learning_rate", "float", "HGB learning_rate for lag_hgb.", 0.1),
        ),
        inputs=(
            "Session with target + time roles and a chronological SplitPlan.",
        ),
        outputs=("ForecastFitResult; ForecastPlan stored on the Session.",),
        prerequisites=(DATASET, ROLES, SPLIT),
        ordering=(
            "After set_roles (target + time) and time_split; optional impute for null targets/exog.",
        ),
        alternatives=(
            "Use supervised Session.fit only when rows are exchangeable — not for temporal forecasts.",
            "Compare naive / seasonal_naive baselines before claiming lag-model value.",
        ),
        rationale=(
            "Use when the goal is leakage-safe univariate (or light exogenous) forecasting with honest metrics.",
        ),
        assumptions=(
            "Clock order is meaningful; partitions are chronological.",
            "Target is numeric and non-null after prep.",
            "Exogenous columns, when used, are numeric and known at prediction time.",
        ),
        failures=(
            "Missing time role, random split, unparseable timestamps, insufficient rows for lags, null targets.",
        ),
        leakage=(
            "Shuffled splits mix future rows into train features and optimistic metrics.",
            "Building lag features with future targets is silent temporal leakage.",
        ),
        anti_patterns=(
            "Calling this a digital twin or full econometrics suite.",
            "Using Session.split(random) then fit_forecast.",
            "Treating MAPE near zero targets as a stable primary metric.",
        ),
        state_changes=(
            "Stores forecast_plan and forecast_fit_result; clears prior generate/eval slots.",
        ),
        result_reading=(
            "Read method, n_train_rows, n_fit_rows, lags, univariate, disclosures.",
        ),
        next_steps=(
            "generate_forecast and/or evaluate_forecast; optionally save_forecast_bundle.",
        ),
        concepts=(
            "forecast-temporal-leakage",
            "forecast-lag-features",
            "forecast-univariate-vs-exog",
            "leakage-boundary",
        ),
    ),
    _operation(
        "generate_forecast",
        OperationKind.TRANSFORM,
        "Generate an H-step forecast from a train-fitted ForecastPlan (no refit).",
        "Produce future values from a frozen origin using recursive one-step predictions.",
        "Forecast horizon generate step.",
        (
            "Require an attached ForecastPlan.",
            "Start from train_end (default) or later partition ends.",
            "Recursive multi-step: predictions may feed later lags.",
            "Require future_exog when the plan uses exogenous columns.",
        ),
        parameters=(
            _p("horizon", "int | None", "Steps ahead; defaults to plan.horizon."),
            _p(
                "origin",
                "train_end | validation_end | test_end",
                "History origin for generation.",
                "train_end",
            ),
            _p(
                "future_exog",
                "array | DataFrame | None",
                "Future exogenous rows when the plan is not univariate.",
            ),
        ),
        inputs=("Active ForecastPlan; optional future_exog.",),
        outputs=("ForecastGenerateResult with prediction tuple.",),
        prerequisites=(FORECAST_PLAN,),
        ordering=("After fit_forecast or load_forecast_bundle.",),
        alternatives=("evaluate_forecast when holdout actuals exist for scoring.",),
        rationale=("Use to materialize a horizon path for planning or inspection.",),
        assumptions=(
            "Origin history is long enough for max(lags).",
            "Future exog, when required, is honestly supplied by the caller.",
        ),
        failures=(
            "Missing plan, short history, missing future_exog for exog plans.",
        ),
        leakage=(
            "Supplying future target values as if they were history invents leakage — do not.",
        ),
        anti_patterns=(
            "Interpreting recursive multi-step accuracy as one-step skill.",
        ),
        state_changes=("Stores forecast_generate_result.",),
        result_reading=("Read predictions, horizon, origin, disclosures.",),
        next_steps=("evaluate_forecast on holdout; save_forecast_bundle if deploying the plan.",),
        concepts=(
            "forecast-horizon-generate",
            "forecast-lag-features",
            "forecast-univariate-vs-exog",
        ),
    ),
    _operation(
        "evaluate_forecast",
        OperationKind.DIAGNOSTIC,
        "Evaluate a train-fitted ForecastPlan on a holdout partition.",
        "Score MAE/RMSE/MAPE with rolling one-step or fixed-origin strategies.",
        "Forecast evaluation step.",
        (
            "Require an attached ForecastPlan and a holdout partition.",
            "rolling_one_step walks chronologically using prior actuals only.",
            "origin issues a fixed multi-step recursive forecast from prior end.",
            "Report MAE/RMSE/MAPE with MAPE limitations disclosed.",
        ),
        parameters=(
            _p("partition", "validation | test", "Holdout partition to score.", "test"),
            _p(
                "strategy",
                "rolling_one_step | origin",
                "Evaluation protocol.",
                "rolling_one_step",
            ),
        ),
        inputs=("Active ForecastPlan and chronological SplitPlan.",),
        outputs=("ForecastEvalResult with metrics and disclosures.",),
        prerequisites=(FORECAST_PLAN, SPLIT),
        ordering=("After fit_forecast; typically on validation then test.",),
        alternatives=("generate_forecast when you need future values without actuals.",),
        rationale=("Use to quantify holdout forecast error under an explicit protocol.",),
        assumptions=(
            "Holdout actuals exist and are numeric.",
            "Partition order remains chronological vs train.",
        ),
        failures=("Empty partition, missing plan, null holdout targets/exog.",),
        leakage=(
            "Evaluating after a shuffled split is refused at fit time; do not bypass with inject_split that mixes clocks.",
        ),
        anti_patterns=(
            "Reporting only MAPE near zero.",
            "Calling origin multi-step error 'accuracy' without naming the strategy.",
        ),
        state_changes=("Stores forecast_eval_result.",),
        result_reading=("Read mae/rmse/mape beside partition and strategy.",),
        next_steps=("Compare baselines; save_forecast_bundle for the chosen plan.",),
        concepts=(
            "forecast-eval-protocols",
            "forecast-temporal-leakage",
            "forecast-metric-limits",
        ),
    ),
    _operation(
        "save_forecast_bundle",
        OperationKind.PERSIST,
        "Persist the active ForecastPlan as buildml.forecast_bundle.v1.",
        "Write a domain bundle distinct from Session checkpoints and Torch/RAG bundles.",
        "Forecast bundle save.",
        (
            "Require an attached ForecastPlan.",
            "Write meta.json + forecast_plan.joblib.",
        ),
        parameters=(_p("path", "str | Path", "Destination directory.", required=True),),
        inputs=("Active ForecastPlan.",),
        outputs=("Bundle directory path.",),
        prerequisites=(FORECAST_PLAN,),
        ordering=("After fit_forecast (optionally after evaluate_forecast).",),
        alternatives=("checkpoint_save for workflow resume without the forecaster.",),
        rationale=("Use when the fitted forecast plan must travel independently of Session data.",),
        assumptions=("Destination is writable.",),
        failures=("Missing plan; incomplete write permissions.",),
        leakage=(
            "Persistence only — the plan is written as fitted, so a plan fitted with temporal leakage stays leaky after reload.",
        ),
        anti_patterns=(
            "Treating a forecast bundle as a Session checkpoint or digital-twin state dump.",
        ),
        state_changes=("No Session mutation beyond history.",),
        result_reading=("Confirm meta.json format == buildml.forecast_bundle.v1.",),
        next_steps=("load_forecast_bundle in a fresh Session with matching roles/splits.",),
        concepts=("forecast-bundle-boundary",),
    ),
    _operation(
        "load_forecast_bundle",
        OperationKind.PERSIST,
        "Load a buildml.forecast_bundle.v1 ForecastPlan into the Session.",
        "Restore a train-fitted forecast plan without refitting.",
        "Forecast bundle load.",
        (
            "Validate bundle format.",
            "Attach ForecastPlan; clear prior fit/generate/eval result slots.",
        ),
        parameters=(
            _p("path", "str | Path", "Bundle directory.", required=True),
            _p(
                "trusted",
                "bool",
                "Must be True to deserialize pickle/joblib/torch payloads (default False).",
                False,
            ),
        ),
        inputs=("forecast_bundle directory.",),
        outputs=("Session with forecast_plan attached.",),
        prerequisites=(DATASET,),
        ordering=("After ingest/roles/time_split for subsequent generate/evaluate.",),
        alternatives=("fit_forecast to train a new plan.",),
        rationale=("Use to reuse a previously fitted forecast plan.",),
        assumptions=("Bundle format matches buildml.forecast_bundle.v1.",),
        failures=("Missing files; wrong format tag.",),
        leakage=(
            "Loading fits nothing, but it does not re-validate the plan against this Session's split; a shuffled split still makes evaluate_forecast dishonest.",
        ),
        anti_patterns=("Loading into a shuffled-split Session and trusting evaluate_forecast.",),
        state_changes=("Sets forecast_plan; clears fit/generate/eval result caches.",),
        result_reading=("Inspect Session.forecast_plan.to_dict().",),
        next_steps=("generate_forecast / evaluate_forecast.",),
        concepts=("forecast-bundle-boundary",),
    ),
)
