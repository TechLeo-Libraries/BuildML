"""TDA bundle persistence (distinct from Session checkpoints / Torch / RAG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.tda.results import TdaEvalResult, TdaFitResult, TdaPlan

BUNDLE_FORMAT = "buildml.tda_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "TDA bundles, classical pipeline bundles, Torch trainer bundles, RAG "
    "bundles, and Session checkpoints are complementary, not interchangeable. "
    "A TDA bundle (buildml.tda_bundle.v1) stores a train-fitted TdaPlan "
    "(ripser/persim vectorizer state + train NN index + optional sklearn head). "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the TDA transformer. "
    "Reload tabular workflow via checkpoint_load; reload TDA via load_tda_bundle. "
    "Honesty: persistent homology + vectorization → sklearn — not a Mapper "
    "research suite or every TDA paper."
)


def save_tda_bundle(
    path: str | Path,
    plan: TdaPlan,
    *,
    fit_result: TdaFitResult | None = None,
    eval_result: TdaEvalResult | None = None,
) -> Path:
    """Write a TDA bundle directory (``buildml.tda_bundle.v1``)."""
    if plan is None:
        raise ValidationError("No TdaPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "train_x": np.asarray(plan.train_x_),
        "mean": None if plan.mean_ is None else np.asarray(plan.mean_),
        "scale": None if plan.scale_ is None else np.asarray(plan.scale_),
        "vectorizer_state": dict(plan.vectorizer_state_),
        "feature_names": list(plan.feature_names),
        "classes": list(plan.classes_),
    }
    joblib.dump(payload, destination / "tda_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_tda_bundle(path: str | Path) -> TdaPlan:
    """Load a TDA bundle into a :class:`TdaPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "tda_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete TDA bundle at {root}. "
            f"Expected meta.json and tda_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported TDA bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, TdaPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "tda_plan.joblib must contain a TdaPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, TdaPlan):
        raise ValidationError("Loaded plan object is not a TdaPlan")
    if loaded.get("train_x") is not None and (
        plan.train_x_ is None or plan.train_x_.size == 0
    ):
        plan.train_x_ = np.asarray(loaded["train_x"], dtype=float)
    if loaded.get("mean") is not None and plan.mean_ is None:
        plan.mean_ = np.asarray(loaded["mean"], dtype=float)
    if loaded.get("scale") is not None and plan.scale_ is None:
        plan.scale_ = np.asarray(loaded["scale"], dtype=float)
    if loaded.get("vectorizer_state") and not plan.vectorizer_state_:
        plan.vectorizer_state_ = dict(loaded["vectorizer_state"])
    if loaded.get("feature_names") and not plan.feature_names:
        plan.feature_names = tuple(str(v) for v in loaded["feature_names"])
    if loaded.get("classes") and not plan.classes_:
        plan.classes_ = tuple(loaded["classes"])
    return plan
