# ruff: noqa: E501
"""Time-series analysis concept notes (decomposition, diagnostics, changepoints)."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

TIMESERIES_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="ts-decomposition",
            title="Seasonal decomposition before forecasting",
            summary=(
                "STL or classical decomposition splits a series into trend, seasonal, "
                "and residual components on train-only scope so you can see structure "
                "before choosing a forecaster."
            ),
            definition=(
                "Decomposition models an observation y(t) as trend(t) + seasonal(t) + "
                "residual(t) (additive) or trend(t) × seasonal(t) × residual(t) "
                "(multiplicative). BuildML runs STL when statsmodels is installed, "
                "classical moving-average seasonal when not, or a lightweight "
                "moving-average fallback."
            ),
            intuition=(
                "Separate the slow drift, the repeating pattern, and the leftover "
                "noise before you commit to a forecast model."
            ),
            formal_idea=(
                "Additive STL: y_t = T_t + S_t + R_t with seasonal period m; "
                "residuals should look closer to stationary after removing T and S."
            ),
            why_it_matters=(
                "Seasonality and trend shape which forecast method is honest to try.",
                "Residual diagnostics after decomposition reveal whether simple "
                "classical models are plausible.",
            ),
            how_buildml_uses=(
                "Session.analyze_timeseries(include_decompose=True) or ts_decompose.",
                "Default scope='train' under a temporal SplitPlan.",
            ),
            interpretation_rules=(
                "Inspect seasonal amplitude vs trend slope before picking horizon.",
                "A dominant seasonal period suggests exogenous season length for fit_forecast.",
            ),
            assumptions=(
                "Numeric target with parseable time index.",
                "Enough history to estimate at least one full seasonal cycle when seasonal_period is set.",
            ),
            failure_modes=(
                "Series too short for STL seasonal window.",
                "Irregular sampling treated as fixed seasonal period without resampling.",
            ),
            anti_patterns=(
                "Using scope='all' decomposition to tune forecast hyperparameters silently.",
                "Assuming decomposition quality equals forecast skill.",
            ),
            worked_example_pattern=(
                "time_split → analyze_timeseries(scope='train', include_decompose=True) "
                "→ fit_forecast informed by seasonal_period.",
            ),
            related_concepts=(
                "forecast-temporal-leakage",
                "forecast-univariate-vs-exog",
                "ts-stationarity-diagnostics",
            ),
        ),
        _note(
            key="ts-stationarity-diagnostics",
            title="Stationarity tests and ACF/PACF diagnostics",
            summary=(
                "ADF/KPSS and autocorrelation plots describe whether differencing or "
                "seasonal adjustment is needed before classical forecasting."
            ),
            definition=(
                "A weakly stationary series has constant mean and autocovariance that "
                "depends only on lag. Augmented Dickey–Fuller tests the unit-root "
                "null; KPSS tests stationarity around a trend. ACF/PACF summarize "
                "linear dependence at each lag."
            ),
            intuition=(
                "If today's value strongly predicts tomorrow, the series is not "
                "stationary: and many classical models assume you handled that first."
            ),
            formal_idea=(
                "ADF H0: unit root; reject → evidence against unit root. "
                "KPSS H0: stationarity; reject → evidence against stationarity."
            ),
            why_it_matters=(
                "Misreading stationarity leads to wrong differencing and optimistic scores.",
                "PACF spikes suggest autoregressive order; ACF decay suggests MA structure.",
            ),
            how_buildml_uses=(
                "Session.analyze_timeseries(include_diagnostics=True) or ts_diagnostics.",
                "statsmodels paths when buildml[timeseries] installed; honest refusal otherwise.",
            ),
            interpretation_rules=(
                "Report both ADF and KPSS: they test opposite nulls.",
                "Diagnostics describe the series given; they do not certify a model.",
            ),
            assumptions=("Chronological order; numeric target.",),
            failure_modes=(
                "Structural breaks make single-series stationarity tests misleading.",
                "Very short series produce unstable ACF/PACF.",
            ),
            anti_patterns=(
                "Tuning forecast lags using holdout ACF peaks without disclosure.",
                "Treating p-values as proof the series is forecast-ready.",
            ),
            worked_example_pattern=(
                "ts_diagnostics(scope='train') → choose lag_ridge lags or seasonal diff.",
            ),
            related_concepts=("ts-decomposition", "forecast-lag-features"),
        ),
        _note(
            key="ts-changepoint-detection",
            title="Changepoint detection on temporal series",
            summary=(
                "PELT/binseg (ruptures) or CUSUM fallback locate mean/variance shifts "
                "so you do not train one stationary model across a regime break."
            ),
            definition=(
                "Changepoint detection finds times τ where the generating distribution "
                "changes. BuildML prefers ruptures PELT/binseg when installed and falls "
                "back to a lightweight CUSUM core otherwise."
            ),
            intuition=(
                "If your product launch changed customer behaviour mid-series, a single "
                "forecast model across the whole timeline is averaging two different worlds."
            ),
            formal_idea=(
                "Segment the series at changepoints τ_1,…,τ_k that minimize a penalized "
                "segment cost Σ_i C(y_{τ_{i-1}:τ_i}) + β·k."
            ),
            why_it_matters=(
                "Undetected breaks inflate forecast error and hide non-stationarity.",
                "Regime labels can motivate separate models or exogenous flags.",
            ),
            how_buildml_uses=(
                "Session.analyze_timeseries(include_changepoints=True) on train scope.",
            ),
            interpretation_rules=(
                "Treat changepoints as hypotheses: validate with domain knowledge.",
                "More changepoints with weak penalty often mean over-segmentation.",
            ),
            assumptions=("Regular enough sampling for segment costs to be meaningful.",),
            failure_modes=(
                "Gradual drift mistaken for a single changepoint.",
                "Seasonal spikes flagged as breaks when seasonal_period is wrong.",
            ),
            anti_patterns=(
                "Refitting changepoint penalty on holdout to minimize forecast error.",
            ),
            worked_example_pattern=(
                "analyze_timeseries(include_changepoints=True) → document breaks before fit_forecast.",
            ),
            related_concepts=("ts-decomposition", "forecast-temporal-leakage"),
        ),
        _note(
            key="ts-analysis-before-forecast",
            title="Time-series analysis vs forecasting",
            summary=(
                "analyze_timeseries is diagnostic EDA on train scope; fit_forecast is "
                "predictive modelling: run analysis first so method choice is informed."
            ),
            definition=(
                "Analysis surfaces structure (seasonality, stationarity, breaks, spectral "
                "features) without producing holdout forecasts. Forecasting fits a model "
                "and generates future values under temporal split rules."
            ),
            intuition=(
                "Look at the shape of the river before you pick a boat: analysis is "
                "the map, forecasting is the voyage."
            ),
            formal_idea=(
                "Analysis operators are functionals of past observations only under "
                "scope='train'; forecast fit additionally optimizes parameters for "
                "predictive loss on chronological holdout."
            ),
            why_it_matters=(
                "Skipping analysis hides season length and break points until forecast fails.",
                "Keeps diagnostic leakage separate from predictive evaluation.",
            ),
            how_buildml_uses=(
                "time_split → analyze_timeseries → fit_forecast → evaluate_forecast.",
                "timeseries_capability_matrix() lists installed diagnostic backends.",
            ),
            interpretation_rules=(
                "Analysis results do not include forecast metrics.",
                "Use train scope unless explicitly documenting exploratory scope='all'.",
            ),
            assumptions=("Temporal SplitPlan present for honest defaults.",),
            failure_modes=(
                "Random split then analyze_timeseries: refused or misleading.",
            ),
            anti_patterns=(
                "Using analyze_timeseries on holdout to pick a forecaster silently.",
                "Expecting changepoint detection to replace a forecast model.",
            ),
            worked_example_pattern=(
                "set_roles(time+target) → time_split → analyze_timeseries() → fit_forecast().",
            ),
            related_concepts=(
                "forecast-temporal-leakage",
                "ts-decomposition",
                "ts-stationarity-diagnostics",
            ),
        ),
    )
}
