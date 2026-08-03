"""Federated-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.federated.results import (
    FederatedEvalResult,
    FederatedFitResult,
    FederatedPlan,
)

BUNDLE_FORMAT = "buildml.federated_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Federated bundles, meta-learning bundles, multi-task bundles, online "
    "bundles, active-learning bundles, semi-supervised bundles, "
    "self-supervised bundles, anomaly bundles, unsupervised bundles, "
    "classical pipeline bundles, Torch trainer bundles, RAG bundles, and "
    "Session checkpoints are complementary, not interchangeable. A federated "
    "bundle (buildml.federated_bundle.v1) stores a FederatedPlan (global "
    "linear/SGD model + client partition contract + round history). A Session "
    "checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the federated global model. Reload "
    "tabular workflow via checkpoint_load; reload the learner via "
    "load_federated_bundle. This is a local FedAvg-style simulation — not a "
    "distributed FL network stack and not cryptographic secure aggregation."
)


def save_federated_bundle(
    path: str | Path,
    plan: FederatedPlan,
    *,
    fit_result: FederatedFitResult | None = None,
    eval_result: FederatedEvalResult | None = None,
) -> Path:
    """Write a federated-learning bundle directory (``buildml.federated_bundle.v1``).

    Persists the plan and optional fit/eval summaries as joblib + JSON metadata.
    Distinct from Session checkpoints.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Fitted :class:`~buildml.federated.results.FederatedPlan`.
    fit_result:
        Optional fit summary to embed in ``meta.json``.
    eval_result:
        Optional evaluation summary to embed in ``meta.json``.

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No FederatedPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "federated_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return destination


def load_federated_bundle(path: str | Path, *, trusted: bool = False) -> FederatedPlan:
    """Load a federated-learning bundle into a :class:`FederatedPlan`.

    Validates bundle format version and plan object type before returning.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``federated_plan.joblib``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    FederatedPlan
        Deserialized plan with global estimator and label encoder attached.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or plan type is wrong.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "federated_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete federated-learning bundle at {root}. "
            f"Expected meta.json and federated_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported federated-learning bundle format {fmt!r}; "
            f"expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, FederatedPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "federated_plan.joblib must contain a FederatedPlan or a "
            "payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, FederatedPlan):
        raise ValidationError("Loaded plan object is not a FederatedPlan")
    return plan
