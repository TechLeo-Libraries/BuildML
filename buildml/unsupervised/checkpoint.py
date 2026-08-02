"""Unsupervised bundle persistence (v2 + v1 migration)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.unsupervised.results import ClusterEvalResult, ClusterFitResult, ClusterPlan

BUNDLE_FORMAT_V1 = "buildml.unsupervised_bundle.v1"
BUNDLE_FORMAT_V2 = "buildml.unsupervised_bundle.v2"
BUNDLE_FORMAT = BUNDLE_FORMAT_V2
CHECKPOINT_BOUNDARY = (
    "Unsupervised bundles, classical pipeline bundles, Torch trainer bundles, RAG "
    "bundles, and Session checkpoints are complementary, not interchangeable. "
    f"A unsupervised bundle ({BUNDLE_FORMAT_V2}) stores a train-fitted "
    "ClusterPlan (estimator + feature contract + assign strategy disclosures). "
    "Legacy v1 bundles remain loadable. "
    "A Session checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the clusterer. "
    "Reload tabular workflow via checkpoint_load; reload clustering via "
    "load_unsupervised_bundle."
)


def save_unsupervised_bundle(
    path: str | Path,
    plan: ClusterPlan,
    *,
    fit_result: ClusterFitResult | None = None,
    eval_result: ClusterEvalResult | None = None,
) -> Path:
    """Write an unsupervised bundle directory (``buildml.unsupervised_bundle.v2``)."""
    if plan is None:
        raise ValidationError("No ClusterPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "centroids": None if plan.centroids_ is None else np.asarray(plan.centroids_),
        "centroid_label_ids": list(plan.centroid_label_ids_),
        "core_sample_indices": list(plan.core_sample_indices_),
    }
    joblib.dump(payload, destination / "cluster_plan.joblib")
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


def load_unsupervised_bundle(path: str | Path) -> ClusterPlan:
    """Load a unsupervised bundle into a :class:`ClusterPlan`."""
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "cluster_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete unsupervised bundle at {root}. "
            f"Expected meta.json and cluster_plan.joblib ({BUNDLE_FORMAT_V2} or {BUNDLE_FORMAT_V1})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt not in {BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2}:
        raise ValidationError(
            f"Unsupported unsupervised bundle format {fmt!r}; "
            f"expected {BUNDLE_FORMAT_V2} or {BUNDLE_FORMAT_V1}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, ClusterPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "cluster_plan.joblib must contain a ClusterPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, ClusterPlan):
        raise ValidationError("Loaded plan object is not a ClusterPlan")
    if loaded.get("centroids") is not None and plan.centroids_ is None:
        plan.centroids_ = np.asarray(loaded["centroids"], dtype=float)
    if loaded.get("centroid_label_ids") and not plan.centroid_label_ids_:
        plan.centroid_label_ids_ = tuple(int(v) for v in loaded["centroid_label_ids"])
    if loaded.get("core_sample_indices") and not plan.core_sample_indices_:
        plan.core_sample_indices_ = tuple(int(v) for v in loaded["core_sample_indices"])
    return plan
