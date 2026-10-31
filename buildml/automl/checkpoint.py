"""AutoML bundle persistence (distinct from Session checkpoints / pipelines)."""

from __future__ import annotations

import json
from copy import copy
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib

from buildml._version import __version__
from buildml.automl.results import AutoMLPlan, AutoMLResult, fit_result_from_plan
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.model.supervised import FitResult

BUNDLE_FORMAT = "buildml.automl_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "AutoML bundles, classical pipeline bundles, ensemble bundles, unsupervised "
    "bundles, Torch trainer bundles, RAG bundles, and Session checkpoints are "
    "complementary, not interchangeable. "
    "An AutoML bundle (buildml.automl_bundle.v1) stores a train-selected "
    "AutoMLPlan (family/recipe disclosures + fitted estimator, often a "
    "fold-local preprocess Pipeline) and the classical FitResult feature "
    "contract. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the AutoML plan. "
    "Prefer save_pipeline when Session-global preprocess plans must travel with "
    "the estimator; use save_automl_bundle when AutoML search disclosures and "
    "AutoMLPlan matter."
    " AutoGluon bundles include the backend's saved model files and cached "
    "training artifacts in a relocatable asset directory."
)

def save_automl_bundle(
    path: str | Path,
    plan: AutoMLPlan,
    *,
    fit_result: FitResult | None = None,
    automl_result: AutoMLResult | None = None,
) -> Path:
    """Write an AutoML bundle directory (``buildml.automl_bundle.v1``).

    Persists the plan and optional classical fit / search summaries as joblib +
    JSON metadata. AutoGluon model files are cloned into a relative asset
    directory so the original training directory is not required. These files
    can include the backend's cached training artifacts. Distinct from Session
    checkpoints.

    Parameters
    ----------
    path:
        Destination directory (created if missing).
    plan:
        Fitted :class:`~buildml.automl.results.AutoMLPlan`.
    fit_result:
        Optional classical :class:`~buildml.model.supervised.FitResult` to
        embed alongside the plan.
    automl_result:
        Optional search summary to embed in ``meta.json``.

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
        raise ValidationError("No AutoMLPlan to save.")
    _validate_model_contract(plan, fit_result)
    destination = Path(path)
    persisted_plan, persisted_fit = plan, fit_result
    assets_name = None
    from buildml.automl.adapters.autogluon import _AutoGluonSklearnWrapper

    if isinstance(plan.estimator_, _AutoGluonSklearnWrapper):
        predictor = plan.estimator_._predictor
        if destination.resolve().is_relative_to(Path(predictor.path).resolve()):
            raise ValidationError("AutoML bundle destination must be outside the AutoGluon predictor directory.")
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(plan.estimator_, _AutoGluonSklearnWrapper):
        # Clone into a fresh sidecar, leaving an existing bundle's assets intact
        # if cloning fails. The live Session predictor is never relocated.
        assets_name = "autogluon-" + uuid4().hex
        predictor.clone(str((destination / assets_name).resolve()))
        wrapper = copy(plan.estimator_)
        wrapper._predictor = None
        persisted_plan = replace(plan, estimator_=wrapper)
        if fit_result is not None:
            persisted_fit = replace(fit_result, estimator=wrapper)
    payload = {
        "plan": persisted_plan,
        "fit_result": persisted_fit,
    }
    joblib.dump(payload, destination / "automl_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "automl": None if automl_result is None else automl_result.to_dict(),
    }
    if assets_name is not None:
        meta["autogluon_assets"] = assets_name
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination

def load_automl_bundle(path: str | Path, *, trusted: bool = False) -> tuple[AutoMLPlan, FitResult | None]:
    """Load an AutoML bundle into an :class:`AutoMLPlan` and optional FitResult.

    Validates bundle format version and plan object type before returning.
    Reconstructs a classical FitResult from the plan when none was stored.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``automl_plan.joblib``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    tuple[AutoMLPlan, FitResult | None]
        Deserialized plan with estimator attached, plus optional FitResult for
        classical evaluate/predict paths.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or plan type is wrong.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "automl_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete AutoML bundle at {root}. "
            f"Expected meta.json and automl_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported AutoML bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, AutoMLPlan):
        plan = loaded
        fit_result = None
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        fit_result = loaded.get("fit_result")
    else:
        raise ValidationError(
            "automl_plan.joblib must contain an AutoMLPlan or a payload with key 'plan'."
        )
    if not isinstance(plan, AutoMLPlan):
        raise ValidationError("Loaded plan object is not an AutoMLPlan")
    if fit_result is not None and not isinstance(fit_result, FitResult):
        fit_result = fit_result_from_plan(plan)
    elif fit_result is None:
        fit_result = fit_result_from_plan(plan)
    _validate_model_contract(plan, fit_result)
    assets_name = meta.get("autogluon_assets")
    if assets_name is not None:
        if not isinstance(assets_name, str) or not assets_name or Path(assets_name).is_absolute():
            raise ValidationError("AutoGluon assets must name a relative directory inside the bundle.")
        bundle_root = root.resolve()
        assets_root = (bundle_root / assets_name).resolve()
        if assets_root == bundle_root or not assets_root.is_relative_to(bundle_root) or not assets_root.is_dir():
            raise ValidationError("AutoGluon assets must exist inside the bundle directory.")
        from buildml.automl.adapters.autogluon import _AutoGluonSklearnWrapper
        from buildml.automl.extras import require_autogluon

        if not isinstance(plan.estimator_, _AutoGluonSklearnWrapper):
            raise ValidationError("AutoGluon asset metadata requires an AutoGluon estimator.")
        predictor = require_autogluon(feature="AutoGluon bundle loading").load(str(assets_root))
        wrapper = _AutoGluonSklearnWrapper(predictor, target=plan.target_column,
                                          feature_cols=list(plan.feature_columns))
        plan.estimator_ = wrapper
        if fit_result is not None:
            fit_result.estimator = wrapper
    return plan, fit_result


def _validate_model_contract(plan: AutoMLPlan, fit_result: FitResult | None) -> None:
    """Keep AutoML and classical prediction contracts aligned at persistence.

    Parameters
    ----------
    plan:
        Selected AutoML estimator and raw column contract.
    fit_result:
        Optional classical result carried with the same selected model.

    Raises
    ------
    ValidationError
        When the plan is incomplete or its paired fit result disagrees.
    """
    if not isinstance(plan, AutoMLPlan):
        raise ValidationError("AutoML bundle requires an AutoMLPlan.")
    if plan.task not in {"classification", "regression"}:
        raise ValidationError("AutoML plan has an unsupported task.")
    columns = plan.feature_columns
    if (not isinstance(columns, (tuple, list)) or not columns
            or any(not isinstance(c, str) or not c.strip() for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValidationError("AutoML plan requires unique, nonempty feature column names.")
    if (not isinstance(plan.target_column, str) or not plan.target_column.strip()
            or plan.target_column in columns):
        raise ValidationError("AutoML plan target must be separate from feature columns.")
    if (isinstance(plan.n_train_rows, bool) or not isinstance(plan.n_train_rows, Integral)
            or plan.n_train_rows < 1):
        raise ValidationError("AutoML plan n_train_rows must be a positive integer.")
    if not callable(getattr(plan.estimator_, "predict", None)):
        raise ValidationError("AutoML plan is missing a prediction estimator.")
    if fit_result is None:
        return
    if not isinstance(fit_result, FitResult):
        raise ValidationError("AutoML fit_result must be a FitResult or None.")
    if fit_result.estimator is not plan.estimator_:
        raise ValidationError("AutoML plan and FitResult must refer to the same estimator.")
    if (not isinstance(fit_result.feature_columns, (tuple, list))
            or tuple(fit_result.feature_columns) != tuple(columns)):
        raise ValidationError("AutoML plan and FitResult feature columns must agree in order.")
    if (fit_result.task != plan.task or fit_result.target_column != plan.target_column
            or isinstance(fit_result.n_train_rows, bool)
            or not isinstance(fit_result.n_train_rows, Integral)
            or fit_result.n_train_rows != plan.n_train_rows):
        raise ValidationError("AutoML plan and FitResult task, target and training-row contract must agree.")
