"""Forecast bundle persistence (distinct from Session checkpoints / Torch / RAG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.forecasting.results import (
    ForecastEvalResult,
    ForecastFitResult,
    ForecastGenerateResult,
    ForecastPlan,
)

BUNDLE_FORMAT = "buildml.forecast_bundle.v2"
BUNDLE_FORMAT_V1 = "buildml.forecast_bundle.v1"
BUNDLE_FORMAT_V2 = "buildml.forecast_bundle.v2"
SUPPORTED_FORMATS = frozenset({BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2})
CHECKPOINT_BOUNDARY = (
    "Forecast bundles, classical pipeline bundles, unsupervised/ensemble/AutoML "
    "bundles, Torch trainer bundles, RAG bundles, and Session checkpoints are "
    "complementary, not interchangeable. "
    "A forecast bundle (buildml.forecast_bundle.v2) stores a train-fitted "
    "ForecastPlan (baseline, lag, statsmodels, Prophet, or neural estimator + contract). "
    "A Session checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the forecaster. "
    "Reload tabular workflow via checkpoint_load; reload forecasting via "
    "load_forecast_bundle."
)


def save_forecast_bundle(
    path: str | Path,
    plan: ForecastPlan,
    *,
    fit_result: ForecastFitResult | None = None,
    eval_result: ForecastEvalResult | None = None,
    generate_result: ForecastGenerateResult | None = None,
) -> Path:
    """Write a forecast bundle directory (``buildml.forecast_bundle.v2``).

    Layout
    ------
    ``meta.json``, ``forecast_plan.joblib``.
    v1 bundles remain loadable via :func:`load_forecast_bundle`.
    """
    if plan is None:
        raise ValidationError("No ForecastPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "last_train_values": list(plan.last_train_values_),
        "seasonal_history": list(plan.seasonal_history_),
    }
    joblib.dump(payload, destination / "forecast_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "generate": None if generate_result is None else generate_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_forecast_bundle(path: str | Path) -> ForecastPlan:
    """Load a forecast bundle into a :class:`ForecastPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "forecast_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete forecast bundle at {root}. "
            f"Expected meta.json and forecast_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise ValidationError(
            f"Unsupported forecast bundle format {fmt!r}; "
            f"expected one of {sorted(SUPPORTED_FORMATS)}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, ForecastPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "forecast_plan.joblib must contain a ForecastPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, ForecastPlan):
        raise ValidationError("Loaded plan object is not a ForecastPlan")
    if loaded.get("last_train_values") and not plan.last_train_values_:
        plan.last_train_values_ = tuple(float(v) for v in loaded["last_train_values"])
    if loaded.get("seasonal_history") and not plan.seasonal_history_:
        plan.seasonal_history_ = tuple(float(v) for v in loaded["seasonal_history"])
    return plan
