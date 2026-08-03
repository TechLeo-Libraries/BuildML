# ruff: noqa: E501
"""Beginner layers for time-series forecasting."""

from __future__ import annotations

from buildml.explain.beginner._builder import (
    ADVANCED,
    CORE,
    FOUNDATION,
    BeginnerLayer,
    _index,
    _layer,
)

FORECASTING_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "forecast-temporal-leakage",
        plain=(
            "In forecasting, the split boundary is time. If any training row comes from after any "
            "evaluation row, the model has seen the future and your score is fiction. A random split — "
            "perfectly fine for ordinary tabular work — destroys a forecasting evaluation."
        ),
        analogy=(
            "Predicting Monday's weather after being shown Tuesday's. Impressive on paper, useless on "
            "Sunday night."
        ),
        steps=(
            "Sort your rows by the time column and confirm the order is genuinely chronological.",
            "Split by time: everything before a cut-off trains, everything after evaluates.",
            "Check every engineered feature for future information — a 'monthly total' that includes the target month is a future leak.",
            "Make sure rolling statistics only look backwards.",
            "Verify the maximum training timestamp is earlier than the minimum evaluation timestamp.",
        ),
        use=(
            "On every forecasting problem, without exception. BuildML's forecasting APIs require chronological order for this reason.",
            "Also on ordinary classification when the model will be used to predict future cases from past ones.",
        ),
        avoid=(
            "Do not use `split(shuffle=True)` or stratified splitting on a time series.",
            "Do not shuffle rows for 'better mixing' before a forecasting fit — you are shuffling the future into the past.",
        ),
        myths=(
            (
                "Random splits are more statistically rigorous.",
                "They assume rows are exchangeable. Time-ordered data is the textbook case where that assumption is false.",
            ),
            (
                "A feature computed only from other columns cannot leak.",
                "It can, if those columns were themselves recorded later. A 'customer lifetime value' column usually contains the whole future.",
            ),
        ),
        example=(
            "session.set_roles({'sales': 'target', 'order_date': 'time'})",
            "session.time_split(time_column='order_date', test_size=0.2)",
            "session.fit_forecast(method='lag_ridge', lags=[1, 7, 28])",
            "session.evaluate_forecast(partition='test')",
        ),
        check=(
            "What is the latest timestamp in your training rows and the earliest in your test rows?",
            "Does any feature aggregate over a window that includes the prediction point?",
        ),
        tools=("time_split", "fit_forecast", "evaluate_forecast", "extract_dates"),
        terms=("forecasting", "leakage", "time split", "lag feature"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "forecast-lag-features",
        plain=(
            "A lag feature is simply 'what the value was N steps ago', turned into its own column. Once you "
            "have a few of those, an ordinary regression model can forecast — no specialized sequence model "
            "required, because the history is now sitting in the row."
        ),
        analogy=(
            "Guessing today's temperature. You would want yesterday's, last week's same day, and maybe last "
            "month's average written down in front of you. Lag features write them down."
        ),
        steps=(
            "Choose lags that match the rhythm of your data: 1 for yesterday, 7 for the same weekday, 12 or 365 for yearly cycles.",
            "Optionally add rolling summaries — a 7-day mean, a 28-day maximum — always computed strictly backwards.",
            "BuildML builds the supervised table where each row's features are past values and its target is the current one.",
            "Rows at the very start have no history and are dropped; note how many.",
            "Fit any ordinary regression model on that table.",
        ),
        use=(
            "As the first thing to try on almost any forecasting problem — it is fast, interpretable, and often hard to beat.",
            "When you want to reuse the gradient-boosting model you already trust rather than learn a new framework.",
        ),
        avoid=(
            "Do not add dozens of lags on a short series; you will run out of rows and start fitting noise.",
            "Do not use a lag longer than your forecast horizon can support in recursive generation without understanding the error compounding.",
        ),
        myths=(
            (
                "Forecasting needs a specialized model like ARIMA or an LSTM.",
                "Lag features plus a good tabular model is a genuinely competitive approach and is far easier to debug.",
            ),
            (
                "More lags capture more of the pattern.",
                "Each lag costs a row of history and a column of width. Lags matched to the actual seasonality beat a long undifferentiated list.",
            ),
        ),
        example=(
            "session.fit_forecast(",
            "    method='lag_ridge', lags=[1, 2, 7, 14, 28],",
            "    rolling_windows=[7, 28],",
            ")",
            "print(session.forecast_plan.n_dropped_warmup_rows)",
        ),
        check=(
            "Do your lag choices match a real cycle in the data — weekly, monthly, yearly?",
            "How many rows did you lose to the warm-up period?",
        ),
        tools=("fit_forecast", "generate_forecast", "evaluate_forecast", "eda"),
        terms=("lag feature", "seasonality", "forecasting", "horizon"),
        difficulty=CORE,
    ),
    _layer(
        "forecast-univariate-vs-exog",
        plain=(
            "A univariate forecast uses only the target's own history. An exogenous forecast also uses "
            "outside drivers such as price or promotion. The catch with exogenous drivers is brutal: to "
            "forecast H steps ahead you need to already know those drivers H steps ahead."
        ),
        analogy=(
            "Predicting ice-cream sales from temperature works beautifully — until you realize forecasting "
            "next month's sales now requires forecasting next month's weather."
        ),
        steps=(
            "Start univariate. It is the honest baseline and often surprisingly strong.",
            "If you add exogenous columns, list them explicitly and confirm they are numeric.",
            "For horizon generation, supply the future values of every exogenous column.",
            "Known-in-advance drivers (holidays, planned promotions, scheduled prices) are the safe ones.",
            "For unknown drivers, either forecast them separately and accept the compounded error, or leave them out.",
        ),
        use=(
            "When a driver is genuinely known ahead of time and materially moves the target.",
            "When domain experts can name a cause your target's own history cannot express.",
        ),
        avoid=(
            "Do not add an exogenous column whose future you cannot obtain — you will build a model you cannot run.",
            "Do not add many exogenous columns on a short series; each one costs degrees of freedom you do not have.",
        ),
        myths=(
            (
                "Adding more drivers always improves a forecast.",
                "It improves the fit on history. Out-of-sample it often makes things worse, because you have swapped one uncertainty for two.",
            ),
            (
                "A driver that correlates strongly with the target is a good exogenous feature.",
                "Only if you will know its value at forecast time. Otherwise the correlation is unusable.",
            ),
        ),
        example=(
            "session.fit_forecast(",
            "    method='lag_ridge', lags=[1, 7],",
            "    exog_columns=['is_holiday', 'planned_discount'],",
            ")",
            "session.generate_forecast(horizon=14, future_exog=future_frame)",
        ),
        check=(
            "For each exogenous column: will you actually know its value for the whole horizon?",
            "How much does the univariate baseline lose compared with the exogenous version?",
        ),
        tools=("fit_forecast", "generate_forecast", "evaluate_forecast"),
        terms=("exogenous", "horizon", "forecasting", "seasonality"),
        difficulty=CORE,
    ),
    _layer(
        "forecast-horizon-generate",
        plain=(
            "Generating an H-step forecast means predicting one step, then feeding that prediction back in "
            "as if it were an observed value to predict the next, and so on. Errors compound: step 14 is "
            "built on thirteen guesses."
        ),
        analogy=(
            "Photocopying a photocopy. The first copy is nearly perfect. The fourteenth is visibly degraded, "
            "and every flaw you introduced early is still there, magnified."
        ),
        steps=(
            "Freeze the fitted plan — generation never refits.",
            "Predict the next step from the real observed history.",
            "Append that prediction to the history as if it were actual.",
            "Repeat until you reach the horizon.",
            "Report the horizon length alongside the numbers, and expect accuracy to decay with distance.",
        ),
        use=(
            "When you genuinely need multiple future periods — a quarter of demand, a month of capacity.",
            "For planning scenarios where the trajectory matters more than any single point.",
        ),
        avoid=(
            "Do not compare a 14-step generated forecast against a rolling one-step evaluation and call them the same accuracy; they measure different tasks.",
            "Do not push the horizon far beyond what you validated — the error growth is not linear and not guessable.",
        ),
        myths=(
            (
                "A model with good one-step accuracy has good multi-step accuracy.",
                "One-step accuracy uses real history at every point. Multi-step accumulates its own mistakes, and the two can diverge dramatically.",
            ),
            (
                "The forecast for step 14 is as trustworthy as the forecast for step 1.",
                "It is built on thirteen previous predictions. Uncertainty grows with every step, which is why forecast intervals widen.",
            ),
        ),
        example=(
            "result = session.generate_forecast(horizon=14)",
            "print(result.predictions)          # 14 values",
            "print(result.protocol)             # 'recursive multi-step'",
        ),
        check=(
            "What is your accuracy at step 1 versus step H?",
            "Does your planning process need the trajectory or just the total?",
        ),
        tools=("generate_forecast", "evaluate_forecast", "fit_forecast", "predict_interval"),
        terms=("horizon", "forecasting", "plan", "prediction interval"),
        difficulty=CORE,
    ),
    _layer(
        "forecast-eval-protocols",
        plain=(
            "There are two honest ways to score a forecast and they answer different questions. Rolling "
            "one-step re-supplies the real value after each prediction, measuring 'how good is the next-step "
            "model?'. Origin evaluation fixes a starting point and scores the whole recursive path, "
            "measuring 'how good is the plan I would actually ship?'."
        ),
        analogy=(
            "Testing a sat-nav by correcting the driver at every junction, versus letting them drive the "
            "whole route on the original instructions. Both are fair tests; they are not the same test."
        ),
        steps=(
            "Decide which question you need answered — next-step quality, or full-horizon planning quality.",
            "For rolling one-step, the evaluator walks forward, predicting one step and then revealing the actual.",
            "For origin evaluation, it fixes an origin and scores the recursive multi-step path from there.",
            "Read the metric together with the protocol name; a number without a protocol is uninterpretable.",
            "Report both when the audience might assume the more favourable one.",
        ),
        use=(
            "Rolling one-step when you will retrain or re-observe every period anyway.",
            "Origin evaluation when a plan is committed for the whole horizon before any actuals arrive.",
        ),
        avoid=(
            "Do not quote rolling one-step accuracy for a system that must forecast a quarter ahead unaided.",
            "Do not switch protocols between model comparisons; the ranking can genuinely reverse.",
        ),
        myths=(
            (
                "There is one correct way to score a forecast.",
                "The protocol has to match how the forecast is used. That is a business fact, not a statistical one.",
            ),
            (
                "Rolling evaluation is more rigorous because it uses more data.",
                "It is more favourable, because it hands the model a real value after every step. Rigour comes from matching the deployment, not from the higher number.",
            ),
        ),
        example=(
            "rolling = session.evaluate_forecast(partition='test', protocol='rolling_one_step')",
            "origin = session.evaluate_forecast(partition='test', protocol='origin', horizon=14)",
            "print(rolling.mae, origin.mae)   # expect origin to be worse",
        ),
        check=(
            "In production, will your model see actuals between predictions?",
            "Which protocol produced the number in your slide deck?",
        ),
        tools=("evaluate_forecast", "generate_forecast", "fit_forecast"),
        terms=("forecasting", "horizon", "metric", "MAE"),
        difficulty=ADVANCED,
    ),
    _layer(
        "forecast-metric-limits",
        plain=(
            "MAE is the average size of your miss in the target's own units. RMSE is similar but punishes "
            "large misses much harder. MAPE turns misses into percentages, which reads nicely and breaks "
            "badly whenever the actual value is near zero."
        ),
        analogy=(
            "Being 5 units off matters differently if the true value is 1000 or 2. MAPE tries to fix that "
            "and creates a new problem: being 5 off when the truth is 0.1 is a 5000% error."
        ),
        steps=(
            "Start with MAE — it is in your units and easy to explain to anyone.",
            "Add RMSE when a few large misses are disproportionately costly.",
            "Use MAPE only when your series stays comfortably away from zero and stakeholders need percentages.",
            "Always report the metric alongside a naive baseline, such as 'predict the last observed value'.",
            "Note the horizon and the protocol; the same model produces very different numbers under different ones.",
        ),
        use=(
            "MAE and RMSE for essentially all internal model comparison.",
            "MAPE for communication, with an explicit caveat about small denominators.",
        ),
        avoid=(
            "Do not use MAPE on intermittent demand — the zero days will dominate the average and hide everything else.",
            "Do not compare MAE across series with different scales; a MAE of 100 is excellent for revenue and catastrophic for a percentage.",
        ),
        myths=(
            (
                "MAPE is a universal accuracy percentage.",
                "It is scale-sensitive, asymmetric between over- and under-prediction, and undefined at zero. It is a communication tool, not a measure of truth.",
            ),
            (
                "A low RMSE means a good forecast.",
                "It means small squared errors on the rows you scored. Against a flat series, 'predict the previous value' often scores well and forecasts nothing.",
            ),
        ),
        example=(
            "report = session.evaluate_forecast(partition='test')",
            "print(report.mae, report.rmse, report.mape)",
            "print(report.disclosures)   # includes MAPE instability notes",
        ),
        check=(
            "Does your series ever approach zero?",
            "How does your model compare with 'repeat the last value'?",
        ),
        tools=("evaluate_forecast", "fit_forecast", "generate_forecast"),
        terms=("MAE", "RMSE", "MAPE", "baseline", "metric"),
        difficulty=CORE,
    ),
    _layer(
        "forecast-bundle-boundary",
        plain=(
            "Your fitted forecasting plan — the model, the lag configuration, the feature contract — saves "
            "as a forecast bundle. A Session checkpoint stores your data, roles, split, and history. They "
            "are different files answering different questions."
        ),
        analogy=(
            "The forecasting model is the instrument; the checkpoint is the logbook. Both are worth keeping "
            "and neither replaces the other."
        ),
        steps=(
            "Fit a forecast so a plan exists.",
            "Call `save_forecast_bundle(path)` to store the fitted weights and the lag/exog contract.",
            "Reload with `load_forecast_bundle(path)` on a Session with a compatible time series.",
            "Call `generate_forecast` or `evaluate_forecast` against the restored plan.",
            "Save a checkpoint separately if you also need the historical frame.",
        ),
        use=(
            "When the forecasting model goes into a scheduled job that runs without your notebook.",
            "When the lag and exogenous configuration must be reproduced exactly rather than remembered.",
        ),
        avoid=(
            "Do not expect the bundle to carry your historical data; generation needs history you supply.",
            "Do not load a bundle against a series with a different frequency or different exogenous columns.",
        ),
        myths=(
            (
                "The bundle contains the forecast.",
                "It contains the fitted plan that produces forecasts. The numbers are generated when you call for them, against whatever history you provide.",
            ),
            (
                "A checkpoint is a superset of the bundle.",
                "Checkpoints deliberately do not embed domain plans, so that each artifact has exactly one meaning.",
            ),
        ),
        example=(
            "session.save_forecast_bundle('artifacts/demand-forecast')",
            "job = Session.ingest(latest_history).load_forecast_bundle('artifacts/demand-forecast')",
            "job.generate_forecast(horizon=14)",
        ),
        check=(
            "Does the reloaded plan's expected column list match your fresh data?",
            "Where does the history come from when the scheduled job runs?",
        ),
        tools=("save_forecast_bundle", "load_forecast_bundle", "generate_forecast", "checkpoint_save"),
        terms=("bundle", "checkpoint", "plan", "forecasting"),
        difficulty=CORE,
    ),
)

__all__ = ["FORECASTING_BEGINNER"]
