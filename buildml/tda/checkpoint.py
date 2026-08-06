"""TDA bundle persistence (distinct from Session checkpoints / Torch / RAG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.tda.results import TdaEvalResult, TdaFitResult, TdaPlan

BUNDLE_FORMAT = "buildml.tda_bundle.v2"
BUNDLE_FORMAT_V1 = "buildml.tda_bundle.v1"
SUPPORTED_BUNDLE_FORMATS = (BUNDLE_FORMAT, BUNDLE_FORMAT_V1)
CHECKPOINT_BOUNDARY = (
    "TDA bundles, classical pipeline bundles, Torch trainer bundles, RAG "
    "bundles, and Session checkpoints are complementary, not interchangeable. "
    "A TDA bundle (buildml.tda_bundle.v2) stores a train-fitted TdaPlan "
    "(backend + PH vectorizer state + train NN index + optional sklearn head). "
    "v1 bundles (native ripser/persim only) remain loadable. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the TDA transformer. "
    "Reload tabular workflow via checkpoint_load; reload TDA via load_tda_bundle. "
    "Honesty: persistent homology + vectorization → sklearn: not a Mapper "
    "research suite or every TDA paper."
)


def save_tda_bundle(
    path: str | Path,
    plan: TdaPlan,
    *,
    fit_result: TdaFitResult | None = None,
    eval_result: TdaEvalResult | None = None,
) -> Path:
    """Write a train-fitted TDA plan to a ``buildml.tda_bundle.v2`` directory.

    Persists the frozen plan, vectorizer state, optional sklearn head metadata,
    and summary JSON separate from Session checkpoints. Reload with
    :func:`load_tda_bundle` or Session :meth:`~buildml.session.session.Session.load_tda_bundle`.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Train-fitted :class:`~buildml.tda.results.TdaPlan` to persist.
    fit_result:
        Optional fit report embedded in ``meta.json`` for audit trails.
    eval_result:
        Optional evaluation report embedded in ``meta.json``.

    Returns
    -------
    pathlib.Path
        The bundle directory containing ``tda_plan.joblib`` and ``meta.json``.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
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
        "mapper_summary": None if plan.mapper_summary_ is None else dict(plan.mapper_summary_),
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


def load_tda_bundle(path: str | Path, *, trusted: bool = False) -> TdaPlan:
    """Load a TDA bundle from disk into a :class:`~buildml.tda.results.TdaPlan`.

    Supports v1 (native ripser/persim only) and v2 bundles. Rehydrates train
    arrays and vectorizer state from ``tda_plan.joblib`` when the plan object
    alone is incomplete.

    Parameters
    ----------
    path:
        Bundle directory written by :func:`save_tda_bundle`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    TdaPlan
        Train-fitted plan ready for transform, predict, or evaluate calls.

    Raises
    ------
    ValidationError
        When files are missing, the format is unsupported, or the payload is
        not a valid :class:`TdaPlan`.
    """
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
    if fmt not in SUPPORTED_BUNDLE_FORMATS:
        raise ValidationError(
            f"Unsupported TDA bundle format {fmt!r}; expected one of {SUPPORTED_BUNDLE_FORMATS}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, TdaPlan):
        plan = loaded
    elif isinstance(loaded, dict) and "plan" in loaded:
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
        if loaded.get("mapper_summary") and plan.mapper_summary_ is None:
            plan.mapper_summary_ = dict(loaded["mapper_summary"])
    else:
        raise ValidationError(
            "tda_plan.joblib must contain a TdaPlan or a payload with key 'plan'."
        )
    if fmt == BUNDLE_FORMAT_V1 and not getattr(plan, "backend", None):
        plan.backend = "native"
    return plan
