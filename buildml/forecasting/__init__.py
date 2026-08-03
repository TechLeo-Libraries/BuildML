"""Classical time-series forecasting domain (lag/baseline Session path).

Phase coverage (internal tracker: depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**):
  1. Unsupervised learning: done (``buildml.unsupervised``).
  2. Ensemble learning: done (``buildml.ensemble``).
  3. AutoML: done (``buildml.automl``).
  4. Time-series forecasting: real forecast API. **This module.**
  5. Anomaly / fraud detection: done (see ``buildml.anomaly``).

Later phases (do not start until Phase 1 items hit the Torch/RAG bar):
  Phase 2: semi/self/active/online done; **next=multi-task**; then
  meta-learning, federated, Bayesian/probabilistic (done), causal (separate assumption
  objects; EDA stays associational), graph ML, evolutionary search/NAS-lite,
  symbolic/neuro-symbolic, CBR, imitation + RL, allowlisted LLM tool agents,
  TDA, recommenders / LTR / KG / optimisation / synthetic-data / NLP-CV deepenings.
  Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): neuromorphic/SNN, swarm zoo,
digital twins, AV stack, multi-agent world sims, TTS, robotics/control product,
full COCO detection/segmentation suite.

Honesty
-------
Industry defaults when ``buildml[timeseries]`` installed (ETS/ARIMA/SARIMAX).
Core lag/baseline fallback with clear MissingExtraError when industry methods
requested without extras. Prophet/N-BEATS behind separate extras.
Refuses shuffled random splits. Not a digital twin.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Optional:
``buildml[timeseries]``, ``buildml[timeseries-prophet]``, ``buildml[timeseries-ml]``.

Lazy imports: core never grows heavy forecast stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ForecastConfig",
    "ForecastEvalResult",
    "ForecastEvalStrategy",
    "ForecastFitResult",
    "ForecastGenerateResult",
    "ForecastMethod",
    "ForecastPlan",
    "evaluate_forecast",
    "fit_forecaster",
    "forecast_capability_matrix",
    "forecasting_status",
    "forecasting_status_for_session",
    "generate_forecast",
    "load_forecast_bundle",
    "save_forecast_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"ForecastMethod", "ForecastEvalStrategy", "ForecastConfig"}:
        from buildml.forecasting import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ForecastPlan",
        "ForecastFitResult",
        "ForecastGenerateResult",
        "ForecastEvalResult",
    }:
        from buildml.forecasting import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_forecaster":
        from buildml.forecasting.fit import fit_forecaster

        return fit_forecaster
    if name == "generate_forecast":
        from buildml.forecasting.predict import generate_forecast

        return generate_forecast
    if name == "evaluate_forecast":
        from buildml.forecasting.evaluate import evaluate_forecast

        return evaluate_forecast
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_forecast_bundle",
        "load_forecast_bundle",
    }:
        from buildml.forecasting import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"forecasting_status", "forecasting_status_for_session"}:
        from buildml.forecasting import explain_hooks as hooks

        return getattr(hooks, name)
    if name == "forecast_capability_matrix":
        from buildml.forecasting.catalog import forecast_capability_matrix

        return forecast_capability_matrix
    raise AttributeError(f"module 'buildml.forecasting' has no attribute {name!r}")
