# ruff: noqa: E501
"""Beginner layers for time-series analysis (not forecasting)."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

TIMESERIES_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "ts-decomposition",
        plain=(
            "Before you forecast, split the series into three parts: the slow trend, the repeating "
            "seasonal pattern, and the leftover noise. That picture tells you whether you need "
            "seasonal lags, differencing, or a simpler model."
        ),
        analogy=(
            "Listening to a song: the melody repeats (seasonal), the volume drifts (trend), and "
            "there is hiss on the recording (residual)."
        ),
        steps=(
            "Put rows in time order and assign time + target roles.",
            "Use a chronological split so diagnostics see only past data by default.",
            "Run decomposition on the training partition.",
            "Read seasonal amplitude and trend slope.",
            "Carry the seasonal period into fit_forecast if a strong cycle appears.",
        ),
        use=(
            "Whenever you are about to forecast and have not looked at seasonality yet.",
            "When stakeholders ask 'is there a weekly pattern?' before you commit to a model.",
        ),
        avoid=(
            "Do not decompose the full dataset including holdout and then tune on what you saw.",
            "Do not assume a pretty decomposition guarantees accurate forecasts.",
        ),
        myths=(
            (
                "Decomposition replaces forecasting.",
                "It describes structure; fit_forecast still predicts the future.",
            ),
        ),
        example=(
            "session.set_roles({'date': 'time', 'sales': 'target'})",
            "session.time_split(time_column='date', test_size=0.2)",
            "session.analyze_timeseries(scope='train', include_decompose=True)",
        ),
        check=(
            "Does the seasonal curve repeat at the period you expect (7, 12, 365)?",
            "Is the residual much noisier after removing trend and season?",
        ),
        tools=("analyze_timeseries", "ts_decompose", "timeseries_capability_matrix"),
        terms=("seasonality", "forecasting", "stationarity"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "ts-stationarity-diagnostics",
        plain=(
            "Stationarity means the statistical behaviour of the series does not drift over time. "
            "BuildML runs ADF/KPSS tests and ACF/PACF plots so you can see whether differencing "
            "or seasonal adjustment is needed before classical forecasting."
        ),
        analogy=(
            "A roulette wheel that slowly gains a bias is not fair over time: differencing or "
            "detrending tries to restore 'fairness' for models that assume it."
        ),
        steps=(
            "Ensure chronological order and a train-only scope.",
            "Run ts_diagnostics or analyze_timeseries with diagnostics enabled.",
            "Read ADF and KPSS together: they test opposite nulls.",
            "Inspect ACF/PACF for slow decay or sharp spikes at seasonal lags.",
            "Decide on differencing or lag features before fit_forecast.",
        ),
        use=(
            "Before lag-based or ARIMA-style classical forecasts.",
            "When the series looks like it has a strong upward trend.",
        ),
        avoid=(
            "Do not pick forecast lags using holdout ACF peaks without saying so.",
            "Do not treat a single p-value as proof the series is ready to model.",
        ),
        myths=(
            (
                "Non-stationary series cannot be forecast.",
                "You transform or model the structure: diagnostics tell you which path is honest.",
            ),
        ),
        example=(
            "session.ts_diagnostics(scope='train', include_diagnostics=True)",
            "session.fit_forecast(method='lag_ridge', lags=[1, 7, 14])",
        ),
        check=(
            "Do ADF and KPSS tell a consistent story about trends?",
            "Does PACF show spikes at your business cycle length?",
        ),
        tools=("ts_diagnostics", "analyze_timeseries", "timeseries_capability_matrix"),
        terms=("stationarity", "forecasting", "leakage"),
        difficulty=CORE,
    ),
    _layer(
        "ts-changepoint-detection",
        plain=(
            "Changepoint detection finds moments when the series behaviour shifts: a policy change, "
            "a product launch, or a broken sensor. Spotting breaks keeps you from training one model "
            "across two different regimes."
        ),
        analogy=(
            "A speed camera that suddenly starts ticketing on a road where limits changed last month: "
            "the 'normal' before and after are not the same distribution."
        ),
        steps=(
            "Run changepoint detection on train scope first.",
            "Inspect proposed break times against domain events.",
            "Document whether to split models, add exogenous flags, or trim history.",
            "Only then fit_forecast on the regime you intend to deploy.",
            "Disclose any regime decision in the model card or report.",
        ),
        use=(
            "When visual inspection shows an obvious level shift.",
            "Before pooling years of data that span major business changes.",
        ),
        avoid=(
            "Do not tune changepoint penalty on holdout forecast error silently.",
            "Do not treat every seasonal peak as a structural break.",
        ),
        myths=(
            (
                "More changepoints are always better.",
                "Each extra break adds variance; weak penalties over-segment noise.",
            ),
        ),
        example=(
            "session.time_split(time_column='date', test_size=0.2)",
            "session.analyze_timeseries(scope='train', include_changepoints=True)",
        ),
        check=(
            "Can you name a real-world event near each detected break?",
            "Would a separate model per segment be feasible in production?",
        ),
        tools=("analyze_timeseries", "timeseries_capability_matrix"),
        terms=("drift", "forecasting", "time split"),
        difficulty=CORE,
    ),
    _layer(
        "ts-analysis-before-forecast",
        plain=(
            "Time-series analysis describes the past: seasonality, stationarity, breaks, and spectra. "
            "Forecasting predicts the future. BuildML keeps them separate so diagnostics on train "
            "do not become hidden tuning on holdout."
        ),
        analogy=(
            "Weather radar shows current storm structure; the forecast tells you where it will go. "
            "You need both, in that order."
        ),
        steps=(
            "Assign time + target roles and create a chronological split.",
            "Call analyze_timeseries on train scope.",
            "Choose forecast method and lags informed by decomposition and diagnostics.",
            "fit_forecast on train, evaluate_forecast on holdout.",
        ),
        use=(
            "On every new temporal dataset before the first fit_forecast call.",
            "When stakeholders ask why a forecast method was chosen: analysis is the evidence trail.",
        ),
        avoid=(
            "Skipping straight to fit_forecast on a series you have not inspected.",
            "Using scope='all' analysis to pick hyperparameters without disclosure.",
        ),
        myths=(
            (
                "analyze_timeseries produces forecast accuracy.",
                "It produces structure reports: evaluate_forecast measures predictive skill.",
            ),
        ),
        example=(
            "session.analyze_timeseries(scope='train')",
            "session.fit_forecast(method='lag_hgb', lags=[1, 7, 28])",
            "session.evaluate_forecast(partition='test')",
        ),
        check=(
            "Did you run analysis before the first forecast fit?",
            "Is your split temporal, not random?",
        ),
        tools=(
            "analyze_timeseries",
            "timeseries_capability_matrix",
            "fit_forecast",
            "evaluate_forecast",
        ),
        terms=("forecasting", "leakage", "time split"),
        difficulty=FOUNDATION,
    ),
)
