# ruff: noqa: E501
"""Forecasting concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

FORECASTING_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="forecast-temporal-leakage",
            title="Temporal leakage in forecasting",
            summary="Shuffled splits and future-into-past features invent optimistic forecast scores; forecasting APIs require chronological order.",
            definition=(
                "Temporal leakage occurs when information from a later clock time "
                "influences training features, model selection, or evaluation labels "
                "for an earlier decision time."
            ),
            intuition=(
                "If tomorrow's sales help you 'predict' today's sales in the training "
                "table, the exam was graded with the answer key."
            ),
            formal_idea=(
                "For ordered times t1 < t2 < … < tn, features for predicting y(ti) may "
                "use only information available at times ≤ ti (or < ti for strict "
                "causal lags). Random row splits violate that filtration."
            ),
            why_it_matters=(
                "Shuffled holdouts systematically overstate forecast skill.",
                "Production clocks never shuffle the future into the past.",
            ),
            how_buildml_uses=(
                "Session.fit_forecast refuses random/stratified/group SplitPlan kinds.",
                "time_split (or chronological inject_split) is required; partition order is checked.",
                "Lag features at time t use only y[t-lag].",
            ),
            interpretation_rules=(
                "Always report the split kind beside forecast metrics.",
                "Prefer time_split for any operational claim.",
            ),
            assumptions=(
                "A parseable time-role column exists.",
                "Partitions are disjoint in clock time (train ends before holdout).",
            ),
            failure_modes=(
                "Using Session.split then fit_forecast.",
                "Building lags with centered windows that peek ahead.",
            ),
            anti_patterns=(
                "Random K-fold CV on a single series for the primary forecast claim.",
            ),
            worked_example_pattern=(
                "set_roles(time+target) → time_split → fit_forecast → evaluate_forecast.",
            ),
            related_concepts=("leakage-boundary", "evaluation-partitions", "forecast-lag-features"),
        ),
        _note(
            key="forecast-lag-features",
            title="Lag / window features for classical forecasting",
            summary="Tabularize recent target history into supervised rows so sklearn models can forecast without a sequence neural net.",
            definition=(
                "Lag features are columns y(t-1), y(t-2), … used as inputs to predict "
                "y(t). Window aggregates are summaries of recent history. Together they "
                "turn a series into a supervised table under a strict past-only rule."
            ),
            intuition=(
                "You are teaching a regressor to answer: given the last week of values, "
                "what happens next?"
            ),
            formal_idea=(
                "Given lags L={l1,…,lk}, the supervised row for time t is "
                "x_t = (y_{t-l1},…,y_{t-lk}[, exog_t]) with label y_t, for t large enough "
                "that all lags exist."
            ),
            why_it_matters=(
                "Enables strong classical baselines with core sklearn.",
                "Makes leakage rules explicit at feature construction time.",
            ),
            how_buildml_uses=(
                "lag_ridge and lag_hgb build train-only lag matrices.",
                "generate_forecast recurses: predictions may feed later lags.",
                "rolling_one_step eval appends holdout actuals after each prediction.",
            ),
            interpretation_rules=(
                "n_fit_rows < n_train_rows when early rows lack full lag history.",
                "Recursive multi-step error compounds; compare to rolling one-step.",
            ),
            assumptions=(
                "Series is regularly enough sampled for fixed integer lags to be meaningful.",
                "Target is numeric.",
            ),
            failure_modes=(
                "Too few train rows relative to max(lags).",
                "Irregular sampling treated as fixed-lag without resampling.",
            ),
            anti_patterns=(
                "Calling lag_hgb a 'sequence model' or Transformer forecaster.",
            ),
            worked_example_pattern=(
                "fit_forecast(method='lag_ridge', lags=[1,2,3,7]) → generate_forecast(horizon=7).",
            ),
            related_concepts=("forecast-horizon-generate", "forecast-temporal-leakage"),
        ),
        _note(
            key="forecast-classical-ets-arima",
            title="ETS / ARIMA / SARIMAX industry paths (buildml[timeseries])",
            summary="Statsmodels Holt-Winters and ARIMA-family models fit univariate history; exog only on arima/sarimax/lag paths.",
            definition=(
                "ETS (Exponential Smoothing) captures level/trend/seasonality in the target "
                "series alone. ARIMA/SARIMAX model autoregressive integrated moving-average "
                "structure; SARIMAX adds seasonal orders and optional contemporaneous exog. "
                "auto_arima here is a lightweight in-tree AIC grid — not pmdarima."
            ),
            intuition=(
                "ETS smooths the past forward; ARIMA learns how today relates to recent "
                "errors and levels. Neither uses promo calendars unless you pick "
                "sarimax/lag models with exog_columns."
            ),
            formal_idea=(
                "ETS: level + optional trend + seasonal components with additive errors. "
                "SARIMAX: φ(B)Φ(B^s)(1-B)^d(1-B^s)^D y_t = θ(B)Θ(B^s)ε_t + βX_t."
            ),
            why_it_matters=(
                "Default method='auto' selects ETS when statsmodels is installed.",
                "exog_columns with ETS/Prophet/N-BEATS/auto_arima is refused at fit time.",
            ),
            how_buildml_uses=(
                "fit_forecast(method='ets'|'arima'|'auto_arima'|'sarimax').",
                "Prophet/N-BEATS require separate extras; synthetic calendar ds disclosed.",
            ),
            interpretation_rules=(
                "Read plan.backend, method, disclosures, and univariate flag.",
            ),
            assumptions=("time_split in place; seasonal_period meaningful when seasonal.",),
            failure_modes=("Short train for seasonal ETS; expecting exog on univariate methods.",),
            anti_patterns=("Labeling auto_arima as full pmdarima without reading warnings.",),
            worked_example_pattern=(
                "time_split → fit_forecast(method='ets') → evaluate_forecast('validation').",
            ),
            related_concepts=("forecast-temporal-leakage", "forecast-univariate-vs-exog"),
        ),
        _note(
            key="forecast-univariate-vs-exog",
            title="Univariate vs exogenous forecasting",
            summary="Default path is univariate (target history only); optional numeric exog columns require known future values for horizon generate.",
            definition=(
                "Univariate forecasting uses only the target's own history. Exogenous "
                "(exog) forecasting adds external drivers known at prediction time."
            ),
            intuition=(
                "Temperature next week may help predict ice-cream sales — but only if "
                "you actually have a next-week temperature forecast to feed the model."
            ),
            formal_idea=(
                "Univariate: ŷ(t+h) = f(y≤t). With exog: ŷ(t+h) = f(y≤t, x_{t+h}) when "
                "x is available; otherwise generation is under-specified."
            ),
            why_it_matters=(
                "Silent exog assumptions create undeployable generate paths.",
                "Univariate honesty avoids fake multivariate claims.",
            ),
            how_buildml_uses=(
                "exog_columns empty ⇒ univariate disclosures on the plan.",
                "generate_forecast requires future_exog when exog_columns are set.",
                "evaluate_forecast may use holdout exog at each scored timestamp.",
            ),
            interpretation_rules=(
                "Read plan.univariate before interpreting generate failures.",
                "Document who supplies future exog in production.",
            ),
            assumptions=(
                "Exog columns are numeric and non-null after prep.",
                "Holdout exog used in eval is honestly available at that clock time.",
            ),
            failure_modes=(
                "Fitting with exog then generating without future_exog.",
                "Using target-derived columns as 'exog' that leak the label.",
            ),
            anti_patterns=(
                "Claiming multivariate econometric identification from lag_ridge + exog.",
            ),
            worked_example_pattern=(
                "fit_forecast(exog_columns=['promo']) → generate_forecast(future_exog=...).",
            ),
            related_concepts=("forecast-lag-features", "forecast-horizon-generate"),
        ),
        _note(
            key="forecast-horizon-generate",
            title="Horizon generation (recursive multi-step)",
            summary="H-step generate frezes the plan and recursively predicts ahead; it is not the same protocol as rolling one-step evaluation.",
            definition=(
                "Horizon generation produces ŷ(t+1),…,ŷ(t+H) from a fixed origin without "
                "observing intermediate actuals. Recursive strategies feed predictions "
                "into subsequent lag features."
            ),
            intuition=(
                "You leave the dock with only today's instruments and narrate the next "
                "H days — each day your story depends on yesterday's story."
            ),
            formal_idea=(
                "Let ĥ0 be history at origin. For h=1..H: ŷ_h = f(ĥ_{h-1}[, x_h]); "
                "ĥ_h = ĥ_{h-1} ∪ {ŷ_h}."
            ),
            why_it_matters=(
                "Operational planning needs H-step paths.",
                "Error compounds; do not equate with one-step skill.",
            ),
            how_buildml_uses=(
                "Session.generate_forecast implements recursive generate.",
                "evaluate_forecast(strategy='origin') scores that protocol on holdout length.",
                "Baselines (naive/seasonal/drift/mean) remain available for reference.",
            ),
            interpretation_rules=(
                "Name origin and horizon beside every generated path.",
                "Compare origin vs rolling_one_step metrics.",
            ),
            assumptions=("History length ≥ max(lags) at the chosen origin.",),
            failure_modes=(
                "Short history at validation_end/test_end origins.",
                "Missing future_exog for exog plans.",
            ),
            anti_patterns=(
                "Publishing only generate plots without holdout metrics.",
            ),
            worked_example_pattern=(
                "fit_forecast(horizon=7) → generate_forecast(horizon=7) → inspect predictions.",
            ),
            related_concepts=("forecast-eval-protocols", "forecast-lag-features"),
        ),
        _note(
            key="forecast-eval-protocols",
            title="Rolling one-step vs origin evaluation",
            summary="Rolling one-step uses prior actuals as history; origin scores a fixed recursive multi-step path — report which protocol you used.",
            definition=(
                "Rolling one-step evaluation predicts each holdout point using all prior "
                "actuals, then appends the actual. Origin evaluation freezes history at "
                "the prior partition end and scores a recursive H-step path."
            ),
            intuition=(
                "Rolling is 'tomorrow's forecast with today's truth already known for "
                "yesterday'. Origin is 'forecast the whole exam week on Monday morning'."
            ),
            formal_idea=(
                "Rolling: ŷ_i = f(history ∪ actuals_<i). Origin: "
                "(ŷ_1..ŷ_n) = generate(history, n) vs actuals."
            ),
            why_it_matters=(
                "Protocols answer different operational questions.",
                "Mixing them silently makes model comparisons meaningless.",
            ),
            how_buildml_uses=(
                "evaluate_forecast(strategy='rolling_one_step'|'origin').",
                "Metrics always carry strategy and partition in the result.",
            ),
            interpretation_rules=(
                "Default claims should name the strategy.",
                "Large origin−rolling gaps often mean weak multi-step recursion.",
            ),
            assumptions=("Holdout partition is non-empty and chronological.",),
            failure_modes=("Scoring train as if it were holdout.",),
            anti_patterns=(
                "Cherry-picking the friendlier protocol without disclosure.",
            ),
            worked_example_pattern=(
                "evaluate_forecast(partition='validation', strategy='rolling_one_step').",
            ),
            related_concepts=("forecast-horizon-generate", "forecast-metric-limits"),
        ),
        _note(
            key="forecast-metric-limits",
            title="Forecast metric limitations (MAE/RMSE/MAPE)",
            summary="MAE and RMSE are primary; MAPE is scale-sensitive and unstable near zero — disclosed, not a universal accuracy.",
            definition=(
                "MAE is mean absolute error, RMSE is root mean squared error, and MAPE "
                "is mean absolute percentage error. None certify causal correctness or "
                "business value alone."
            ),
            intuition=(
                "MAPE blows up when actuals are near zero — like dividing by a whisper."
            ),
            formal_idea=(
                "MAE = mean(|ŷ−y|), RMSE = sqrt(mean((ŷ−y)^2)), "
                "MAPE = 100·mean(|ŷ−y|/|y|) on |y|>ε."
            ),
            why_it_matters=(
                "Metric choice changes model ranking.",
                "MAPE-only dashboards mislead on intermittent or near-zero series.",
            ),
            how_buildml_uses=(
                "evaluate_forecast reports mae, rmse, mape with disclosures.",
                "MAPE may be NaN when all |actual|≈0.",
            ),
            interpretation_rules=(
                "Lead with MAE/RMSE; treat MAPE as secondary.",
                "Compare against naive/seasonal_naive on the same protocol.",
            ),
            assumptions=("Errors are computed on the named partition and strategy.",),
            failure_modes=("Publishing MAPE without units or zero-actual warnings.",),
            anti_patterns=("Calling MAPE 'accuracy percent' without caveats.",),
            worked_example_pattern=(
                "Read metrics['mae'], metrics['rmse']; check warnings for MAPE.",
            ),
            related_concepts=("forecast-eval-protocols", "evaluation-partitions"),
        ),
        _note(
            key="forecast-bundle-boundary",
            title="Forecast bundle vs Session checkpoint",
            summary="buildml.forecast_bundle.v1 stores ForecastPlan weights/contract; Session checkpoints store data/roles/splits/history — they are not interchangeable.",
            definition=(
                "A forecast bundle persists a train-fitted ForecastPlan (baseline or lag "
                "estimator + lag/exog contract + disclosures). A Session checkpoint "
                "persists workflow state without embedding the forecaster."
            ),
            intuition=(
                "The recipe card is not the kitchen, and the kitchen is not the recipe card."
            ),
            formal_idea=(
                "Bundle format tag buildml.forecast_bundle.v1 ⇒ meta.json + "
                "forecast_plan.joblib. Checkpoint load does not restore ForecastPlan."
            ),
            why_it_matters=(
                "Operators otherwise expect checkpoint_load to restore forecasts.",
                "Domain bundles stay complementary to Torch/RAG/classical pipelines.",
            ),
            how_buildml_uses=(
                "save_forecast_bundle / load_forecast_bundle.",
                "CHECKPOINT_BOUNDARY disclosure on meta.json.",
            ),
            interpretation_rules=(
                "Reload data/splits separately before evaluate_forecast claims.",
            ),
            assumptions=("joblib can serialize the sklearn estimator when present.",),
            failure_modes=("Wrong format tag; incomplete directory.",),
            anti_patterns=(
                "Calling a forecast bundle a digital-twin state snapshot.",
            ),
            worked_example_pattern=(
                "save_forecast_bundle(path) → load_forecast_bundle(path) → generate_forecast.",
            ),
            related_concepts=("forecast-temporal-leakage", "leakage-boundary"),
        ),
    )
}
