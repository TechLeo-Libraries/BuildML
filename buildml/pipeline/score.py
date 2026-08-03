"""Take a raw frame through the saved transforms and out as predictions.

The path from a raw record to a prediction has several places to go wrong, and
this module walks all of them in order: coerce the incoming types toward what
the contract expects, validate what is left, replay each fitted transform in the
order it was applied during training, confirm the estimator's feature columns
are all present, then predict.

Each step exists because skipping it produces a wrong answer rather than an
error. Types that were not coerced silently become object columns. Transforms
that were not replayed leave raw values where scaled ones belong. Feature
columns that do not match leave the estimator reading the wrong column as the
wrong feature.

Warnings accumulate through the process rather than being raised, because most
of them are informative rather than fatal — extra columns ignored, a resample
plan skipped, a coercion applied. They are worth reading on the first run
against a new data source.

See Also
--------
buildml.pipeline.bundle : The artifact being scored from.
buildml.pipeline.contract : The checks applied before predicting.
"""

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
    """The predictions, and a record of everything that happened to get them.

    The predictions are the answer; the rest is the audit trail. When a
    production score looks wrong, ``warnings``, ``apply_result``, and
    ``contract_validation`` together say what the frame arrived as, what was
    changed, which transforms ran, and which were skipped.

    Attributes
    ----------
    predictions:
        The predicted values, indexed to match the input frame so they can be
        joined back to source records.
    probabilities:
        Class probabilities as a frame with one ``proba_<class>`` column per
        class, or ``None`` when not requested or unsupported.
    apply_result:
        What the transform replay did — which plans applied, which were skipped,
        and why. ``None`` when no plans ran.
    feature_columns:
        The columns fed to the estimator, in order.
    task:
        ``'classification'`` or ``'regression'``.
    warnings:
        Everything non-fatal, gathered from coercion, validation, and replay.
    n_rows:
        How many predictions were produced.
    contract_validation:
        The schema check, including any coerced columns.

    Notes
    -----
    **The index is preserved deliberately.** Predictions can be assigned
    straight back onto the source frame without relying on positional
    alignment.

    **An empty ``warnings`` on a first run against a new source is worth as much
    as a populated one.** It means the frame arrived exactly as the contract
    expected.

    See Also
    --------
    predict_from_pipeline : Producing this.
    """

    predictions: pd.Series
    probabilities: pd.DataFrame | None = None
    apply_result: ApplyPlansResult | None = None
    feature_columns: tuple[str, ...] = ()
    task: str = ""
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    contract_validation: SchemaContractValidation | None = None

    def to_dict(self) -> dict[str, Any]:
        """Summarise the run as plain data, without the predictions themselves.

        The values are deliberately excluded. This is the record you log for
        every scoring run — row counts, which transforms ran, what the contract
        found — and writing the predictions into a log would be both enormous
        and, for personal data, inappropriate.

        Returns
        -------
        dict
            Row count, task, feature columns, whether probabilities were
            produced, the warnings, the applied and skipped plans, and the
            contract validation.

        Notes
        -----
        **Read ``predictions`` and ``probabilities`` from the attributes.**
        """
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
    """Take a raw frame through coercion, validation, and replay to predictions.

    The one call that gets from a record to an answer. It coerces incoming types
    toward the contract, validates what remains and raises if the frame cannot
    be scored, replays the fitted transforms in training order, checks the
    estimator's feature columns are present, and predicts.

    Accepts either a bundle directory or an already-loaded bundle. Pass the
    loaded object when scoring repeatedly — reloading per batch re-reads and
    re-unpickles the model every time.

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

    Raises
    ------
    ValidationError
        If ``data`` is neither a Dataset nor a DataFrame; if the frame fails
        contract validation; if feature columns are still missing after replay;
        if the frame has rows but no usable features; or if the estimator itself
        fails during predict. The last case is wrapped rather than propagated,
        because a bare scikit-learn shape error says nothing about which column
        was wrong — the wrapped message names the expected contract.

    Notes
    -----
    **A missing feature column after replay usually means an unseen category.**
    The message distinguishes the two causes: plans that were never replayed
    because ``apply_plans=False``, and plans that ran but produced a different
    set of columns than at training time. The second is the common one in
    production, and it happens when encoding meets a category the training data
    did not contain.

    **Resample plans are never replayed.** Resampling rewrites training rows to
    rebalance classes; there is nothing to apply at inference. A warning records
    that it was skipped.

    **Extra columns are ignored, not rejected.** Only the fitted feature columns
    are selected, so an upstream system may carry whatever else it likes.

    **``return_proba`` on an estimator without ``predict_proba`` warns rather
    than failing.** The predictions are still valid; the probabilities are
    simply absent.

    **Bundles without a contract still work**, with only the feature-column
    check applied — which catches missing columns but not wrong types.

    Examples
    --------
    Scoring a batch, and reading the audit trail::

        bundle = load_pipeline_bundle("artifacts/churn-v3")
        result = predict_from_pipeline(bundle, new_frame, return_proba=True)

        frame["churn_risk"] = result.probabilities["proba_1"]
        for message in result.warnings:
            print(message)

    See Also
    --------
    buildml.pipeline.bundle.load_pipeline_bundle : Loading once, scoring often.
    buildml.pipeline.contract.validate_score_frame : Checking without scoring.
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
