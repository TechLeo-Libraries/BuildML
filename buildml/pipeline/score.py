"""Score-time prediction through a saved pipeline bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.pipeline.bundle import PipelineBundle, load_pipeline_bundle
from buildml.pipeline.contract import (
    SchemaContractValidation,
    coerce_score_frame,
    raise_for_contract,
)
from buildml.preprocess.apply import ApplyPlansResult, apply_preprocess_plans


@dataclass(slots=True)
class PipelinePredictResult:
    """Predictions from a loaded pipeline bundle on a new frame."""

    predictions: pd.Series
    probabilities: pd.DataFrame | None = None
    apply_result: ApplyPlansResult | None = None
    feature_columns: tuple[str, ...] = ()
    task: str = ""
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    contract_validation: SchemaContractValidation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "task": self.task,
            "feature_columns": list(self.feature_columns),
            "has_probabilities": self.probabilities is not None,
            "warnings": list(self.warnings),
            "applied": None if self.apply_result is None else list(self.apply_result.applied),
            "skipped": None if self.apply_result is None else list(self.apply_result.skipped),
            "contract_validation": (
                None if self.contract_validation is None else self.contract_validation.to_dict()
            ),
        }


def predict_from_pipeline(
    path_or_bundle: str | Path | PipelineBundle,
    data: Dataset | pd.DataFrame,
    *,
    roles: dict[str, ColumnRole | str] | None = None,
    return_proba: bool = False,
    apply_plans: bool = True,
) -> PipelinePredictResult:
    """Load a pipeline bundle and score a new frame in one call.

    Parameters
    ----------
    path_or_bundle:
        Bundle directory path or an already loaded :class:`PipelineBundle`.
    data:
        Score frame as a :class:`~buildml.data.dataset.Dataset` or Pandas
        DataFrame. Roles may be supplied when ``data`` is a bare frame.
    roles:
        Optional column roles for bare DataFrames.
    return_proba:
        When True and the estimator supports ``predict_proba``, also return
        class probabilities.
    apply_plans:
        When True (default), replay fitted preprocess plans from the bundle
        before prediction. Set False only when ``data`` already matches the
        estimator feature contract.

    Returns
    -------
    PipelinePredictResult
        Label predictions, optional probabilities, apply warnings, and the
        feature contract used.

    Notes
    -----
    Resample plans are lineage-only and are never reapplied. When the bundle
    includes ``schema_contract.json``, incoming frames are checked for missing
    columns and dtype-family mismatches before prediction. Older bundles without
    a contract remain usable; only the fitted feature-column check applies.
    """
    bundle = (
        path_or_bundle
        if isinstance(path_or_bundle, PipelineBundle)
        else load_pipeline_bundle(path_or_bundle)
    )
    fit_result = bundle.fit_result
    warnings: list[str] = []
    apply_result: ApplyPlansResult | None = None
    working: Dataset | pd.DataFrame = data

    raw_frame = data.frame if isinstance(data, Dataset) else data
    if not isinstance(raw_frame, pd.DataFrame):
        raise ValidationError("predict_from_pipeline expects a Dataset or pandas.DataFrame")

    plan_present = any(
        plan is not None
        for plan in (
            bundle.impute_plan,
            bundle.encode_plan,
            bundle.scale_plan,
            bundle.date_plan,
            bundle.outlier_plan,
            bundle.binning_plan,
            bundle.feature_select_plan,
            bundle.text_plan,
            bundle.reduce_plan,
            bundle.custom_plan,
        )
    )

    # Coerce + validate raw input against the persisted schema contract.
    contract_stage = "input" if (apply_plans and plan_present) else "features"
    coerced_frame, contract_validation = coerce_score_frame(
        raw_frame,
        bundle.schema_contract,
        stage=contract_stage,
    )
    warnings.extend(contract_validation.warnings)
    raise_for_contract(contract_validation, allow_extra=True)

    # Prefer the coerced frame when the caller passed a bare DataFrame.
    if isinstance(data, pd.DataFrame) and contract_validation.coerced_columns:
        data = coerced_frame
    elif isinstance(data, Dataset) and contract_validation.coerced_columns:
        data = Dataset.from_transformed(
            data,
            coerced_frame,
            schema=data.schema,
            roles=dict(data.roles),
            sync_native=False,
        )

    if apply_plans and plan_present:
        apply_result = apply_preprocess_plans(
            data,
            {
                "impute_plan": bundle.impute_plan,
                "encode_plan": bundle.encode_plan,
                "scale_plan": bundle.scale_plan,
                "date_plan": bundle.date_plan,
                "outlier_plan": bundle.outlier_plan,
                "binning_plan": bundle.binning_plan,
                "feature_select_plan": bundle.feature_select_plan,
                "text_plan": bundle.text_plan,
                "reduce_plan": bundle.reduce_plan,
                "custom_plan": bundle.custom_plan,
                "resample_plan": bundle.resample_plan,
            },
            roles=roles,
        )
        working = apply_result.dataset
        warnings.extend(apply_result.warnings)
    elif apply_plans and bundle.resample_plan is not None:
        warnings.append(
            "ResamplePlan is lineage-only and was not reapplied at score time."
        )

    frame = working.frame if isinstance(working, Dataset) else working
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("predict_from_pipeline expects a Dataset or pandas.DataFrame")

    feature_columns = list(fit_result.feature_columns)
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        hint = ""
        if not apply_plans and plan_present:
            hint = " Pass apply_plans=True to replay bundle preprocess plans first."
        elif apply_plans and plan_present:
            hint = (
                " After plan replay the score frame still lacks the fitted feature "
                "contract — check date/encode/binning/select outputs."
            )
        raise ValidationError(
            f"Score frame missing feature columns required by the pipeline: {missing}.{hint}"
        )

    # Extra columns are ignored; require a non-empty design matrix.
    x = frame.loc[:, feature_columns]
    if x.empty and len(frame) > 0:
        raise ValidationError("Score frame has rows but no usable feature columns after selection")

    estimator = fit_result.estimator
    try:
        preds = estimator.predict(x)
    except Exception as exc:  # noqa: BLE001 - surface estimator schema errors clearly
        raise ValidationError(
            f"Pipeline estimator failed during predict: {exc}. "
            "Verify score-time columns match the fitted feature contract "
            f"{feature_columns}."
        ) from exc

    predictions = pd.Series(preds, index=x.index, name="prediction")
    probabilities: pd.DataFrame | None = None
    if return_proba:
        if not hasattr(estimator, "predict_proba"):
            warnings.append(
                "return_proba=True but the estimator has no predict_proba; "
                "probabilities were omitted."
            )
        else:
            try:
                proba = estimator.predict_proba(x)
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(
                    f"Pipeline estimator failed during predict_proba: {exc}."
                ) from exc
            # Unwrap sklearn Pipeline final step classes when present.
            model = estimator
            if hasattr(estimator, "named_steps") and "model" in getattr(
                estimator, "named_steps", {}
            ):
                model = estimator.named_steps["model"]
            classes = getattr(model, "classes_", range(proba.shape[1]))
            columns = [f"proba_{c}" for c in classes]
            probabilities = pd.DataFrame(proba, columns=columns, index=x.index)

    return PipelinePredictResult(
        predictions=predictions,
        probabilities=probabilities,
        apply_result=apply_result,
        feature_columns=tuple(feature_columns),
        task=str(fit_result.task),
        warnings=warnings,
        n_rows=int(len(predictions)),
        contract_validation=contract_validation,
    )
