"""Anomaly bundle persistence (distinct from Session checkpoints / Torch / RAG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.anomaly.results import AnomalyEvalResult, AnomalyFitResult, AnomalyPlan
from buildml.core.serialization import (
    assert_local_load_path,
    attach_payload_sha256,
    joblib_load_trusted,
    read_json_sidecar,
)
from buildml.core.errors import ValidationError

BUNDLE_FORMAT = "buildml.anomaly_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Anomaly bundles, unsupervised bundles, classical pipeline bundles, Torch "
    "trainer bundles, RAG bundles, and Session checkpoints are complementary, "
    "not interchangeable. "
    "An anomaly bundle (buildml.anomaly_bundle.v1) stores a train-fitted "
    "AnomalyPlan (estimator + feature contract + threshold/alert-rate disclosures). "
    "A Session checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the anomaly detector. "
    "Reload tabular workflow via checkpoint_load; reload anomaly detection via "
    "load_anomaly_bundle. "
    "EDA IsolationForest screens are not AnomalyPlan artifacts."
)


def save_anomaly_bundle(
    path: str | Path,
    plan: AnomalyPlan,
    *,
    fit_result: AnomalyFitResult | None = None,
    eval_result: AnomalyEvalResult | None = None,
) -> Path:
    """Write an anomaly bundle directory (``buildml.anomaly_bundle.v1``).

Layout
------
``meta.json``, ``anomaly_plan.joblib``.

Parameters
----------
path:
    Filesystem path to the bundle directory.
plan:
    Fitted plan object carrying model state and feature contract.
fit_result:
    Optional fit summary to embed in bundle metadata or history.
eval_result:
    Optional evaluation summary for bundle metadata or history.

Returns
-------
Path
    Resolved filesystem path to the written bundle directory.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if plan is None:
        raise ValidationError("No AnomalyPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    plan_path = destination / "anomaly_plan.joblib"
    joblib.dump(payload, plan_path)
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    meta = attach_payload_sha256(meta, plan_path)
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_anomaly_bundle(path: str | Path, *, trusted: bool = False) -> AnomalyPlan:
    """Load an anomaly bundle into an :class:`AnomalyPlan`.

    Persists or restores plan state as joblib plus JSON metadata. Distinct from Session checkpoints — reload workflow via checkpoint_load separately.

    Parameters
    ----------
    path:
        Filesystem path to the bundle directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    AnomalyPlan
        Fitted plan object (AnomalyPlan) with private estimators attached.

    Raises
    ------
    ValidationError
        When preconditions for this operation are not met.
        
    """
    root = assert_local_load_path(path, artifact="anomaly bundle")
    meta_path = root / "meta.json"
    plan_path = root / "anomaly_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete anomaly bundle at {root}. "
            f"Expected meta.json and anomaly_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = read_json_sidecar(meta_path, artifact="anomaly meta.json")
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported anomaly bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(
        plan_path,
        trusted=trusted,
        artifact="joblib plan",
        expected_sha256=meta.get("payload_sha256"),
    )
    if isinstance(loaded, AnomalyPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "anomaly_plan.joblib must contain an AnomalyPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, AnomalyPlan):
        raise ValidationError("Loaded plan object is not an AnomalyPlan")
    if not getattr(plan, "backend", None):
        plan.backend = _infer_backend_from_method(plan.method)
    return plan


def _infer_backend_from_method(method: str) -> str:
    if method in {"hbos", "copod", "ecod", "deepsvdd"}:
        return "pyod"
    if method == "autoencoder":
        return "torch"
    return "sklearn"
