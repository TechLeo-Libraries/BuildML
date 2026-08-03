"""Recommender bundle persistence (distinct from Session checkpoints / RAG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.recommenders.results import (
    RecommendResult,
    RecommenderEvalResult,
    RecommenderFitResult,
    RecommenderPlan,
)

BUNDLE_FORMAT = "buildml.recommender_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Recommender bundles, classical pipeline bundles, Torch trainer bundles, "
    "RAG bundles, TDA bundles, and Session checkpoints are complementary, not "
    "interchangeable. A recommender bundle (buildml.recommender_bundle.v1) "
    "stores a train-fitted RecommenderPlan (user/item catalog, interaction "
    "matrix, similarities or factors). A Session checkpoint stores data, roles, "
    "splits, history, and optional classical preprocess plans; it does not "
    "embed the recommender. Reload tabular workflow via checkpoint_load; reload "
    "the recommender via load_recommender_bundle. "
    "Honesty: Session CF + optional content — not a Netflix-scale platform; "
    "distinct from RAG retrieve/generate and from EDA Recommendation Findings."
)


def save_recommender_bundle(
    path: str | Path,
    plan: RecommenderPlan,
    *,
    fit_result: RecommenderFitResult | None = None,
    eval_result: RecommenderEvalResult | None = None,
    recommend_result: RecommendResult | None = None,
) -> Path:
    """Write a recommender bundle directory (``buildml.recommender_bundle.v1``).

    Persists the fitted :class:`~buildml.recommenders.results.RecommenderPlan`
    and optional fit/eval/recommend summaries. Distinct from Session
    checkpoints — reload via :func:`load_recommender_bundle`.

    Parameters
    ----------
    path:
        Destination directory; created if missing.
    plan:
        Train-fitted plan containing matrix, factors, and backend state.
    fit_result:
        Optional fit summary written into ``meta.json``.
    eval_result:
        Optional holdout eval summary written into ``meta.json``.
    recommend_result:
        Optional recommendation summary written into ``meta.json``.

    Returns
    -------
    Path
        The bundle directory containing ``meta.json`` and
        ``recommender_plan.joblib``.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No RecommenderPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "matrix": np.asarray(plan.matrix_),
        "similarity": None if plan.similarity_ is None else np.asarray(plan.similarity_),
        "user_factors": (
            None if plan.user_factors_ is None else np.asarray(plan.user_factors_)
        ),
        "item_factors": (
            None if plan.item_factors_ is None else np.asarray(plan.item_factors_)
        ),
        "item_popularity": np.asarray(plan.item_popularity_),
        "item_features": (
            None if plan.item_features_ is None else np.asarray(plan.item_features_)
        ),
        "item_feature_mean": (
            None
            if plan.item_feature_mean_ is None
            else np.asarray(plan.item_feature_mean_)
        ),
        "item_feature_scale": (
            None
            if plan.item_feature_scale_ is None
            else np.asarray(plan.item_feature_scale_)
        ),
        "backend_model": plan.backend_model_,
        "user_item_csr": plan.user_item_csr_,
        "lightfm_user_features": plan.lightfm_user_features_,
        "lightfm_item_features": plan.lightfm_item_features_,
        "user_ids": list(plan.user_ids),
        "item_ids": list(plan.item_ids),
    }
    joblib.dump(payload, destination / "recommender_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "recommend": None if recommend_result is None else recommend_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_recommender_bundle(path: str | Path, *, trusted: bool = False) -> RecommenderPlan:
    """Load a recommender bundle into a :class:`RecommenderPlan`.

    Reads ``meta.json`` for format validation and ``recommender_plan.joblib``
    for the plan payload. Rehydrates numpy arrays and backend objects when
    stored separately in the joblib dict.

    Parameters
    ----------
    path:
        Bundle directory written by :func:`save_recommender_bundle`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    RecommenderPlan
        Train-fitted plan ready for :func:`recommend_for_users` or
        :func:`evaluate_recommender`.

    Raises
    ------
    ValidationError
        When the bundle is incomplete, the format is unsupported, or the
        loaded object is not a valid plan.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "recommender_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete recommender bundle at {root}. "
            f"Expected meta.json and recommender_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported recommender bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, RecommenderPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "recommender_plan.joblib must contain a RecommenderPlan or a payload "
            "with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, RecommenderPlan):
        raise ValidationError("Loaded plan object is not a RecommenderPlan")
    if loaded.get("matrix") is not None and (
        plan.matrix_ is None or plan.matrix_.size == 0
    ):
        plan.matrix_ = np.asarray(loaded["matrix"], dtype=float)
    if loaded.get("similarity") is not None and plan.similarity_ is None:
        plan.similarity_ = np.asarray(loaded["similarity"], dtype=float)
    if loaded.get("user_factors") is not None and plan.user_factors_ is None:
        plan.user_factors_ = np.asarray(loaded["user_factors"], dtype=float)
    if loaded.get("item_factors") is not None and plan.item_factors_ is None:
        plan.item_factors_ = np.asarray(loaded["item_factors"], dtype=float)
    if loaded.get("item_popularity") is not None and (
        plan.item_popularity_ is None or plan.item_popularity_.size == 0
    ):
        plan.item_popularity_ = np.asarray(loaded["item_popularity"], dtype=float)
    if loaded.get("item_features") is not None and plan.item_features_ is None:
        plan.item_features_ = np.asarray(loaded["item_features"], dtype=float)
    if loaded.get("item_feature_mean") is not None and plan.item_feature_mean_ is None:
        plan.item_feature_mean_ = np.asarray(loaded["item_feature_mean"], dtype=float)
    if loaded.get("item_feature_scale") is not None and plan.item_feature_scale_ is None:
        plan.item_feature_scale_ = np.asarray(loaded["item_feature_scale"], dtype=float)
    if loaded.get("backend_model") is not None and plan.backend_model_ is None:
        plan.backend_model_ = loaded["backend_model"]
    if loaded.get("user_item_csr") is not None and plan.user_item_csr_ is None:
        plan.user_item_csr_ = loaded["user_item_csr"]
    if loaded.get("lightfm_user_features") is not None and plan.lightfm_user_features_ is None:
        plan.lightfm_user_features_ = loaded["lightfm_user_features"]
    if loaded.get("lightfm_item_features") is not None and plan.lightfm_item_features_ is None:
        plan.lightfm_item_features_ = loaded["lightfm_item_features"]
    if loaded.get("user_ids") and not plan.user_ids:
        plan.user_ids = tuple(loaded["user_ids"])
        plan.user_index_ = {u: i for i, u in enumerate(plan.user_ids)}
    if loaded.get("item_ids") and not plan.item_ids:
        plan.item_ids = tuple(loaded["item_ids"])
        plan.item_index_ = {it: i for i, it in enumerate(plan.item_ids)}
    # Rebuild indexes if missing after load
    if not plan.user_index_ and plan.user_ids:
        plan.user_index_ = {u: i for i, u in enumerate(plan.user_ids)}
    if not plan.item_index_ and plan.item_ids:
        plan.item_index_ = {it: i for i, it in enumerate(plan.item_ids)}
    return plan
