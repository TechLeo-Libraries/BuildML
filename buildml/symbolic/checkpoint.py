"""Symbolic / neuro-symbolic bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.symbolic.results import (
    NeuroSymbolicFitResult,
    NeuroSymbolicPlan,
    SymbolicEvalResult,
    SymbolicFitResult,
    SymbolicPlan,
)

BUNDLE_FORMAT = "buildml.symbolic_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Symbolic bundles, classical pipeline bundles, Torch trainer bundles, "
    "RAG bundles, and Session checkpoints are complementary, not "
    "interchangeable. A symbolic bundle (buildml.symbolic_bundle.v1) stores a "
    "SymbolicPlan (rule knowledge base) and/or NeuroSymbolicPlan (sklearn "
    "base estimator + rules). A Session checkpoint stores data, roles, "
    "splits, history, and optional classical preprocess plans; it does not "
    "embed the symbolic / neuro-symbolic learner. Reload tabular workflow via "
    "checkpoint_load; reload the learner via load_symbolic_bundle. Honesty: "
    "structured tabular rules + optional sklearn hybrid: not Prolog/Z3/AGI."
)


def save_symbolic_bundle(
    path: str | Path,
    plan: SymbolicPlan | NeuroSymbolicPlan,
    *,
    fit_result: SymbolicFitResult | NeuroSymbolicFitResult | None = None,
    eval_result: SymbolicEvalResult | None = None,
) -> Path:
    """Write a train-fitted symbolic plan to a ``buildml.symbolic_bundle.v1`` directory.

    Persists either a :class:`SymbolicPlan` or :class:`NeuroSymbolicPlan` plus
    optional fit/eval summaries separate from Session checkpoints.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Train-fitted symbolic or neuro-symbolic plan.
    fit_result:
        Optional fit report embedded in ``meta.json``.
    eval_result:
        Optional evaluation report embedded in ``meta.json``.

    Returns
    -------
    pathlib.Path
        Bundle directory containing ``symbolic_plan.joblib`` and ``meta.json``.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No SymbolicPlan / NeuroSymbolicPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "symbolic_plan.joblib")
    kind = "neuro_symbolic" if isinstance(plan, NeuroSymbolicPlan) else "symbolic"
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "kind": kind,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return destination


def load_symbolic_bundle(path: str | Path, *, trusted: bool = False) -> SymbolicPlan | NeuroSymbolicPlan:
    """Load a symbolic bundle from disk into a plan object.

    Supports bundles written by :func:`save_symbolic_bundle` or Session
    :meth:`~buildml.session.session.Session.load_symbolic_bundle`.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``symbolic_plan.joblib``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    SymbolicPlan or NeuroSymbolicPlan
        Train-fitted plan ready for predict and evaluate calls.

    Raises
    ------
    ValidationError
        When files are missing, the format is unsupported, or the payload is invalid.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "symbolic_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete symbolic bundle at {root}. "
            f"Expected meta.json and symbolic_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported symbolic bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, (SymbolicPlan, NeuroSymbolicPlan)):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "symbolic_plan.joblib must contain a plan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, (SymbolicPlan, NeuroSymbolicPlan)):
        raise ValidationError(
            "Loaded plan object is not a SymbolicPlan or NeuroSymbolicPlan."
        )
    return plan
