"""Ranker bundle persistence (distinct from Session / RAG / recommender)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.ranking.results import (
    RankerEvalResult,
    RankerFitResult,
    RankerPlan,
    RankResult,
)

BUNDLE_FORMAT = "buildml.ranker_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Ranker bundles, recommender bundles, RAG bundles, classical pipeline "
    "bundles, and Session checkpoints are complementary, not interchangeable. "
    "A ranker bundle (buildml.ranker_bundle.v1) stores a train-fitted "
    "RankerPlan (feature contract, standardization, pointwise or pairwise "
    "estimator). A Session checkpoint stores data, roles, splits, history, and "
    "optional classical preprocess plans; it does not embed the ranker. Reload "
    "tabular workflow via checkpoint_load; reload the ranker via "
    "load_ranker_bundle. Honesty: Session tabular LTR — not a search-engine "
    "product; distinct from RAG retrieve/generate and from recommender CF."
)


def save_ranker_bundle(
    path: str | Path,
    plan: RankerPlan,
    *,
    fit_result: RankerFitResult | None = None,
    eval_result: RankerEvalResult | None = None,
    rank_result: RankResult | None = None,
) -> Path:
    """Write a ranker bundle directory (``buildml.ranker_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No RankerPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "estimator": plan.estimator_,
        "feature_mean": np.asarray(plan.feature_mean_),
        "feature_scale": np.asarray(plan.feature_scale_),
        "coef": None if plan.coef_ is None else np.asarray(plan.coef_),
        "intercept": plan.intercept_,
    }
    joblib.dump(payload, destination / "ranker_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "rank": None if rank_result is None else rank_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_ranker_bundle(path: str | Path) -> RankerPlan:
    """Load a ranker bundle into a :class:`RankerPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "ranker_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete ranker bundle at {root}. "
            f"Expected meta.json and ranker_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported ranker bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, RankerPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "ranker_plan.joblib must contain a RankerPlan or a payload "
            "with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, RankerPlan):
        raise ValidationError("Loaded plan object is not a RankerPlan")
    if loaded.get("estimator") is not None and plan.estimator_ is None:
        plan.estimator_ = loaded["estimator"]
    if loaded.get("feature_mean") is not None:
        plan.feature_mean_ = np.asarray(loaded["feature_mean"], dtype=float)
    if loaded.get("feature_scale") is not None:
        plan.feature_scale_ = np.asarray(loaded["feature_scale"], dtype=float)
    if loaded.get("coef") is not None and plan.coef_ is None:
        plan.coef_ = np.asarray(loaded["coef"], dtype=float)
    if "intercept" in loaded:
        plan.intercept_ = float(loaded["intercept"])
    return plan
