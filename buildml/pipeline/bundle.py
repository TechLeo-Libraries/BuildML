"""Persist fitted preprocess plans with an estimator as one pipeline bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.model.supervised import FitResult
from buildml.pipeline.card import ModelCard, build_model_card, load_model_card, save_model_card
from buildml.pipeline.contract import (
    SCHEMA_CONTRACT_FILENAME,
    SchemaContract,
    build_schema_contract,
    input_columns_from_plans,
    load_schema_contract,
    save_schema_contract,
)
from buildml.preprocess.binning import BinningPlan
from buildml.preprocess.custom import CustomTransformPlan
from buildml.preprocess.dates import DateFeaturePlan
from buildml.preprocess.encode import EncodePlan
from buildml.preprocess.imbalance import ResamplePlan
from buildml.preprocess.impute import SimpleImputePlan
from buildml.preprocess.outliers import OutlierPlan
from buildml.preprocess.reduce import ReducePlan
from buildml.preprocess.scale import ScalePlan
from buildml.preprocess.select import FeatureSelectPlan
from buildml.preprocess.text import TextFeaturePlan

# Bundle directory meta.json format. v1 exploratory bundles remain readable.
BUNDLE_FORMAT = "buildml.pipeline_bundle.v2"
BUNDLE_FORMAT_V1 = "buildml.pipeline_bundle.v1"
SUPPORTED_BUNDLE_FORMATS = {BUNDLE_FORMAT, BUNDLE_FORMAT_V1, None}
# plans.joblib envelope. Unversioned dicts with plan keys are treated as v1.
PLANS_FORMAT = "buildml.plans.v2"
PLANS_FORMAT_V1 = "buildml.plans.v1"
CHECKPOINT_COMPATIBILITY = (
    "Pipeline bundles and checkpoints are complementary, not interchangeable. "
    "A pipeline bundle stores fitted preprocess plans, the estimator, and a model card; "
    "it does not embed dataset rows, split indices, or full Session history. "
    "A checkpoint stores data, roles, splits, history, and optional plan metadata; "
    "it does not embed the estimator. Store them side by side when both resume and "
    "inference are required. Reload plans+model via load_pipeline_bundle; reload data "
    "via checkpoint_load. Bundle meta format is buildml.pipeline_bundle.v2; "
    "plans.joblib uses buildml.plans.v2 with a migration path for older unversioned "
    "plan dicts."
)


@dataclass(slots=True)
class PipelineBundle:
    """Coherent fitted preprocess + estimator artifact."""

    fit_result: FitResult
    impute_plan: SimpleImputePlan | None = None
    encode_plan: EncodePlan | None = None
    scale_plan: ScalePlan | None = None
    date_plan: DateFeaturePlan | None = None
    outlier_plan: OutlierPlan | None = None
    binning_plan: BinningPlan | None = None
    feature_select_plan: FeatureSelectPlan | None = None
    text_plan: TextFeaturePlan | None = None
    reduce_plan: ReducePlan | None = None
    custom_plan: CustomTransformPlan | None = None
    resample_plan: ResamplePlan | None = None
    model_card: ModelCard | None = None
    schema_contract: SchemaContract | None = None
    plans_format: str = PLANS_FORMAT
    bundle_format: str = BUNDLE_FORMAT

    def to_meta(self) -> dict[str, Any]:
        return {
            "format": self.bundle_format,
            "plans_format": self.plans_format,
            "buildml_version": __version__,
            "fit": self.fit_result.to_dict(),
            "impute_plan": None if self.impute_plan is None else self.impute_plan.to_dict(),
            "encode_plan": None if self.encode_plan is None else self.encode_plan.to_dict(),
            "scale_plan": None if self.scale_plan is None else self.scale_plan.to_dict(),
            "date_plan": None if self.date_plan is None else self.date_plan.to_dict(),
            "outlier_plan": None if self.outlier_plan is None else self.outlier_plan.to_dict(),
            "binning_plan": None if self.binning_plan is None else self.binning_plan.to_dict(),
            "feature_select_plan": (
                None if self.feature_select_plan is None else self.feature_select_plan.to_dict()
            ),
            "text_plan": None if self.text_plan is None else self.text_plan.to_dict(),
            "reduce_plan": None if self.reduce_plan is None else self.reduce_plan.to_dict(),
            "custom_plan": None if self.custom_plan is None else self.custom_plan.to_dict(),
            "resample_plan": None if self.resample_plan is None else self.resample_plan.to_dict(),
            "has_model_card": self.model_card is not None,
            "has_schema_contract": self.schema_contract is not None,
            "schema_contract_file": SCHEMA_CONTRACT_FILENAME,
            "compatibility": CHECKPOINT_COMPATIBILITY,
        }


def pack_plans_payload(
    *,
    impute_plan: SimpleImputePlan | None = None,
    encode_plan: EncodePlan | None = None,
    scale_plan: ScalePlan | None = None,
    date_plan: DateFeaturePlan | None = None,
    outlier_plan: OutlierPlan | None = None,
    binning_plan: BinningPlan | None = None,
    feature_select_plan: FeatureSelectPlan | None = None,
    text_plan: TextFeaturePlan | None = None,
    reduce_plan: ReducePlan | None = None,
    custom_plan: CustomTransformPlan | None = None,
    resample_plan: ResamplePlan | None = None,
) -> dict[str, Any]:
    """Wrap plan objects in the versioned ``plans.joblib`` envelope."""
    return {
        "format": PLANS_FORMAT,
        "buildml_version": __version__,
        "plans": {
            "impute_plan": impute_plan,
            "encode_plan": encode_plan,
            "scale_plan": scale_plan,
            "date_plan": date_plan,
            "outlier_plan": outlier_plan,
            "binning_plan": binning_plan,
            "feature_select_plan": feature_select_plan,
            "text_plan": text_plan,
            "reduce_plan": reduce_plan,
            "custom_plan": custom_plan,
            "resample_plan": resample_plan,
        },
    }


def unpack_plans_payload(loaded: Any) -> tuple[dict[str, Any], str]:
    """Normalize a ``plans.joblib`` payload to a flat plan dict + format label.
    Accepts:
    - v2 envelope ``{format, buildml_version, plans: {...}}``
    - v1 / unversioned flat dict with ``*_plan`` keys
    """
    empty = {
        "impute_plan": None,
        "encode_plan": None,
        "scale_plan": None,
        "date_plan": None,
        "outlier_plan": None,
        "binning_plan": None,
        "feature_select_plan": None,
        "text_plan": None,
        "reduce_plan": None,
        "custom_plan": None,
        "resample_plan": None,
    }
    if not isinstance(loaded, dict):
        raise ValidationError("plans.joblib payload must be a mapping")
    fmt = loaded.get("format")
    if fmt == PLANS_FORMAT or (
        fmt is None and "plans" in loaded and isinstance(loaded["plans"], dict)
    ):
        if fmt not in {PLANS_FORMAT, None}:
            raise ValidationError(f"Unsupported plans.joblib format '{fmt}'")
        plans = dict(empty)
        plans.update({k: loaded["plans"].get(k) for k in empty})
        return plans, PLANS_FORMAT if fmt == PLANS_FORMAT else PLANS_FORMAT_V1
    # Flat v1 / legacy: plan keys at the top level.
    plan_keys = set(empty)
    if plan_keys.intersection(loaded.keys()):
        plans = dict(empty)
        plans.update({k: loaded.get(k) for k in empty})
        return plans, PLANS_FORMAT_V1 if fmt in {None, PLANS_FORMAT_V1} else str(fmt)
    raise ValidationError(
        "Unrecognized plans.joblib payload. Expected a buildml.plans.v2 envelope "
        "or a flat dict with impute_plan/encode_plan/... keys."
    )


def save_pipeline_bundle(
    path: str | Path,
    *,
    fit_result: FitResult,
    impute_plan: SimpleImputePlan | None = None,
    encode_plan: EncodePlan | None = None,
    scale_plan: ScalePlan | None = None,
    date_plan: DateFeaturePlan | None = None,
    outlier_plan: OutlierPlan | None = None,
    binning_plan: BinningPlan | None = None,
    feature_select_plan: FeatureSelectPlan | None = None,
    text_plan: TextFeaturePlan | None = None,
    reduce_plan: ReducePlan | None = None,
    custom_plan: CustomTransformPlan | None = None,
    resample_plan: ResamplePlan | None = None,
    model_card: ModelCard | None = None,
    dataset_schema: dict[str, Any] | None = None,
    roles: dict[str, Any] | None = None,
    input_columns: list[str] | tuple[str, ...] | None = None,
    schema_contract: SchemaContract | None = None,
    history: list[dict[str, Any]] | None = None,
    metrics: dict[str, dict[str, float]] | None = None,
    title: str | None = None,
) -> Path:
    """Save a pipeline bundle directory.

    Layout
    ------
    ``model.joblib``, ``plans.joblib`` (``buildml.plans.v2``), ``meta.json``
    (``buildml.pipeline_bundle.v2``), ``schema_contract.json``,
    ``model_card.json``, ``model_card.md``.

    Notes
    -----
    A pipeline bundle is not a Session checkpoint. Checkpoints restore data,
    roles, splits, history, and optional plan metadata; pipeline bundles restore
    fitted transforms and the estimator feature contract. Resample plans are
    stored for lineage only — resampling is a train-row rewrite, not an
    inference-time transform. Older bundles without ``schema_contract.json``
    remain loadable; score-time contract checks are skipped with a warning.
    """
    if fit_result is None:
        raise ValidationError("fit_result is required to save a pipeline bundle")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    preprocess_summary = {
        "impute": None if impute_plan is None else impute_plan.to_dict(),
        "encode": None if encode_plan is None else encode_plan.to_dict(),
        "scale": None if scale_plan is None else scale_plan.to_dict(),
        "dates": None if date_plan is None else date_plan.to_dict(),
        "outliers": None if outlier_plan is None else outlier_plan.to_dict(),
        "binning": None if binning_plan is None else binning_plan.to_dict(),
        "feature_select": None if feature_select_plan is None else feature_select_plan.to_dict(),
        "text": None if text_plan is None else text_plan.to_dict(),
        "reduce": None if reduce_plan is None else reduce_plan.to_dict(),
        "custom": None if custom_plan is None else custom_plan.to_dict(),
        "resample": None if resample_plan is None else resample_plan.to_dict(),
    }
    plan_inputs = input_columns_from_plans(
        {
            "impute_plan": impute_plan,
            "encode_plan": encode_plan,
            "scale_plan": scale_plan,
            "date_plan": date_plan,
            "outlier_plan": outlier_plan,
            "binning_plan": binning_plan,
            "feature_select_plan": feature_select_plan,
            "text_plan": text_plan,
            "reduce_plan": reduce_plan,
            "custom_plan": custom_plan,
        }
    )
    resolved_inputs = input_columns
    if resolved_inputs is None and plan_inputs:
        resolved_inputs = plan_inputs
    if resolved_inputs is None:
        # No preprocess plans: score frame must already match the estimator contract.
        resolved_inputs = list(fit_result.feature_columns)
    contract = schema_contract or build_schema_contract(
        schema=dataset_schema,
        roles=roles,
        feature_columns=fit_result.feature_columns,
        target_column=fit_result.target_column,
        input_columns=resolved_inputs,
    )
    card = model_card or build_model_card(
        fit_result=fit_result,
        dataset_schema=dataset_schema,
        preprocess_summary=preprocess_summary,
        history=history,
        metrics=metrics,
        title=title,
        lineage={
            "artifact": "pipeline_bundle",
            "format": BUNDLE_FORMAT,
            "plans_format": PLANS_FORMAT,
            "contains_checkpoint": False,
            "contains_raw_dataset": False,
            "has_schema_contract": True,
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
            "plans_present": sorted(
                key for key, value in preprocess_summary.items() if value is not None
            ),
        },
    )
    joblib.dump(fit_result.estimator, root / "model.joblib")
    joblib.dump(
        pack_plans_payload(
            impute_plan=impute_plan,
            encode_plan=encode_plan,
            scale_plan=scale_plan,
            date_plan=date_plan,
            outlier_plan=outlier_plan,
            binning_plan=binning_plan,
            feature_select_plan=feature_select_plan,
            text_plan=text_plan,
            reduce_plan=reduce_plan,
            custom_plan=custom_plan,
            resample_plan=resample_plan,
        ),
        root / "plans.joblib",
    )
    save_schema_contract(root, contract)
    bundle = PipelineBundle(
        fit_result=fit_result,
        impute_plan=impute_plan,
        encode_plan=encode_plan,
        scale_plan=scale_plan,
        date_plan=date_plan,
        outlier_plan=outlier_plan,
        binning_plan=binning_plan,
        feature_select_plan=feature_select_plan,
        text_plan=text_plan,
        reduce_plan=reduce_plan,
        custom_plan=custom_plan,
        resample_plan=resample_plan,
        model_card=card,
        schema_contract=contract,
        plans_format=PLANS_FORMAT,
        bundle_format=BUNDLE_FORMAT,
    )
    (root / "meta.json").write_text(
        json.dumps(bundle.to_meta(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_model_card(root, card)
    return root


def load_pipeline_bundle(path: str | Path) -> PipelineBundle:
    """Load a pipeline bundle saved by :func:`save_pipeline_bundle`.

    Reads ``buildml.pipeline_bundle.v2`` and migrates older v1 / unversioned
    ``meta.json`` plus flat ``plans.joblib`` payloads. Missing
    ``schema_contract.json`` is tolerated for legacy bundles.
    """
    root = Path(path)
    model_path = root / "model.joblib"
    plans_path = root / "plans.joblib"
    meta_path = root / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise ValidationError(f"Pipeline bundle incomplete at '{root}'")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt not in SUPPORTED_BUNDLE_FORMATS and "fit" not in meta:
        raise ValidationError(f"Unrecognized pipeline bundle format '{fmt}' at '{root}'")
    estimator = joblib.load(model_path)
    fit_meta = meta["fit"]
    fit_result = FitResult(
        estimator=estimator,
        task=fit_meta["task"],
        feature_columns=tuple(fit_meta["feature_columns"]),
        target_column=fit_meta["target_column"],
        n_train_rows=int(fit_meta["n_train_rows"]),
    )
    plans: dict[str, Any] = {
        "impute_plan": None,
        "encode_plan": None,
        "scale_plan": None,
        "date_plan": None,
        "outlier_plan": None,
        "binning_plan": None,
        "feature_select_plan": None,
        "text_plan": None,
        "reduce_plan": None,
        "custom_plan": None,
        "resample_plan": None,
    }
    plans_format = PLANS_FORMAT_V1
    if plans_path.exists():
        loaded = joblib.load(plans_path)
        plans, plans_format = unpack_plans_payload(loaded)
    card = None
    card_path = root / "model_card.json"
    if card_path.exists():
        card = load_model_card(root)
    contract = load_schema_contract(root)
    return PipelineBundle(
        fit_result=fit_result,
        impute_plan=plans.get("impute_plan"),
        encode_plan=plans.get("encode_plan"),
        scale_plan=plans.get("scale_plan"),
        date_plan=plans.get("date_plan"),
        outlier_plan=plans.get("outlier_plan"),
        binning_plan=plans.get("binning_plan"),
        feature_select_plan=plans.get("feature_select_plan"),
        text_plan=plans.get("text_plan"),
        reduce_plan=plans.get("reduce_plan"),
        custom_plan=plans.get("custom_plan"),
        resample_plan=plans.get("resample_plan"),
        model_card=card,
        schema_contract=contract,
        plans_format=plans_format,
        bundle_format=fmt or BUNDLE_FORMAT_V1,
    )
