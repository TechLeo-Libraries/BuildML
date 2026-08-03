"""Synthesizer bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.synthetic.results import (
    SyntheticEvalResult,
    SyntheticSampleResult,
    SynthesizerFitResult,
    SynthesizerPlan,
)
from buildml.synthetic.types import ColumnSchemaSpec

BUNDLE_FORMAT = "buildml.synthetic_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Synthetic bundles and Session checkpoints are complementary, not "
    "interchangeable. A synthetic bundle (buildml.synthetic_bundle.v1) stores "
    "a fitted SynthesizerPlan (schema + generator state). A Session checkpoint "
    "stores data, roles, splits, history, and optionally a classical FitResult; "
    "it does not embed the SynthesizerPlan. Reload tabular workflow via "
    "checkpoint_load; reload the generator via load_synthetic_bundle. "
    "Honesty: train-fitted generators only — not differential privacy. "
    "Cross-link: Session.resample remains class-balance preprocessing "
    "(buildml[imbalanced]); this bundle is the reusable synthetic-data path."
)


def save_synthetic_bundle(
    path: str | Path,
    plan: SynthesizerPlan,
    *,
    fit_result: SynthesizerFitResult | None = None,
    eval_result: SyntheticEvalResult | None = None,
    sample_result: SyntheticSampleResult | None = None,
) -> Path:
    """Write a synthesizer bundle directory (``buildml.synthetic_bundle.v1``).

Persists or restores plan state as joblib plus JSON metadata. Distinct from Session checkpoints — reload workflow via checkpoint_load separately.

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
sample_result:
    sample result (SyntheticSampleResult | None).

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
        raise ValidationError("No SynthesizerPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"plan": plan}
    joblib.dump(payload, destination / "synthetic_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
        "sample": None if sample_result is None else sample_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_synthetic_bundle(path: str | Path, *, trusted: bool = False) -> SynthesizerPlan:
    """Load a synthesizer bundle into a :class:`SynthesizerPlan`.

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
    SynthesizerPlan
        Fitted plan object (SynthesizerPlan) with private estimators attached.

    Raises
    ------
    ValidationError
        When preconditions for this operation are not met.
        
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "synthetic_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete synthetic bundle at {root}. "
            f"Expected meta.json and synthetic_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported synthetic bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, SynthesizerPlan):
        plan = loaded
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        if not isinstance(plan, SynthesizerPlan):
            raise ValidationError("Loaded plan object is not a SynthesizerPlan")
    else:
        raise ValidationError(
            "synthetic_plan.joblib must contain a SynthesizerPlan or a payload "
            "with key 'plan'."
        )

    # Ensure column_specs are proper dataclasses if joblib restored dicts
    if plan.column_specs and isinstance(plan.column_specs[0], dict):
        plan.column_specs = tuple(
            ColumnSchemaSpec(
                name=str(s["name"]),
                kind=s["kind"],
                n_unique=int(s.get("n_unique") or 0),
                n_null=int(s.get("n_null") or 0),
                categories=tuple(s.get("categories") or ()),
                extras=dict(s.get("extras") or {}),
            )
            for s in plan.column_specs  # type: ignore[assignment]
        )
    if plan.generator_ is None:
        raise ValidationError(
            "Loaded SynthesizerPlan is missing generator_ state; "
            "bundle may be incomplete."
        )
    return plan
