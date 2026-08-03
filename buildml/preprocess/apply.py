"""Ordered re-application of fitted preprocess plans at score time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.binning import BinningPlan, transform_binning
from buildml.preprocess.custom import CustomTransformPlan, transform_custom
from buildml.preprocess.dates import DateFeaturePlan, extract_date_features
from buildml.preprocess.encode import EncodePlan, transform_encoder
from buildml.preprocess.imbalance import ResamplePlan
from buildml.preprocess.impute import SimpleImputePlan, transform_simple_imputer
from buildml.preprocess.outliers import OutlierPlan, apply_outlier_plan
from buildml.preprocess.reduce import ReducePlan, transform_reducer
from buildml.preprocess.scale import ScalePlan, transform_scaler
from buildml.preprocess.select import FeatureSelectPlan, transform_feature_selector
from buildml.preprocess.text import TextFeaturePlan, transform_text_features

# Score-time order. Date expansion runs first when present so later numeric
# steps see created calendar columns. Resample is intentionally absent.
SCORE_TIME_ORDER = (
    "dates",
    "impute",
    "outliers",
    "encode",
    "text",
    "binning",
    "scale",
    "reduce",
    "feature_select",
    "custom",
)
PLAN_KEY_ALIASES = {
    "date_plan": "dates",
    "dates": "dates",
    "impute_plan": "impute",
    "impute": "impute",
    "outlier_plan": "outliers",
    "outliers": "outliers",
    "encode_plan": "encode",
    "encode": "encode",
    "text_plan": "text",
    "text": "text",
    "binning_plan": "binning",
    "binning": "binning",
    "scale_plan": "scale",
    "scale": "scale",
    "reduce_plan": "reduce",
    "reduce": "reduce",
    "feature_select_plan": "feature_select",
    "feature_select": "feature_select",
    "custom_plan": "custom",
    "custom": "custom",
    "resample_plan": "resample",
    "resample": "resample",
}


@dataclass(slots=True)
class ApplyPlansResult:
    """What came out of replaying a set of plans, and what did not run.

    Attributes
    ----------
    dataset:
        The transformed data, ready for prediction.
    applied:
        The steps that ran, in execution order.
    skipped:
        Steps that were not supplied. Check this against what training did:
        a step present at training time and absent here means the model is
        being fed features in a different representation than it learned on.
    warnings:
        Conditions that did not stop the replay but change how to read it —
        a resample plan ignored, a drop action rewritten as capping, an unusual
        rate of unseen categories.
    split_plan:
        Row membership after the replay. Rebuilt when an outlier plan removed
        rows; otherwise the one passed in, or ``None`` when scoring a frame
        that has no partitions.
    """

    dataset: Dataset
    applied: tuple[str, ...]
    skipped: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)
    split_plan: SplitPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        """Summarise the replay as plain JSON-safe values.

        Intended for logs and serving telemetry, so the transformed dataset is
        summarised by its shape rather than embedded.

        Returns
        -------
        dict
            Keys ``applied``, ``skipped``, ``warnings``, ``n_rows``,
            ``columns``, and ``has_split_plan``.
        """
        return {
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
            "n_rows": self.dataset.n_rows,
            "columns": self.dataset.columns,
            "has_split_plan": self.split_plan is not None,
        }


def apply_preprocess_plans(
    data: Dataset | pd.DataFrame,
    plans: dict[str, Any] | None = None,
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
    split_plan: SplitPlan | None = None,
    roles: dict[str, ColumnRole | str] | None = None,
) -> ApplyPlansResult:
    """Replay a set of fitted plans over new data, in the order training used.

    This is the bridge between training and serving. A model trained through a
    session saw its features after a specific sequence of transformations, and
    new data must go through exactly that sequence, with exactly those learned
    constants, or the numbers reaching the model mean something different from
    what it learned on. Getting the order or the constants wrong is the classic
    cause of a model that scored well in a notebook and fails in production.

    Nothing is fitted here. Every plan you pass must already carry its
    training-fitted state.

    Parameters
    ----------
    data:
        A :class:`~buildml.data.dataset.Dataset` or a plain
        :class:`~pandas.DataFrame` of rows to transform.
    plans:
        A mapping of plans, such as the payload loaded from a saved pipeline or
        checkpoint. Both long keys like ``impute_plan`` and short ones like
        ``impute`` are accepted. Any keyword argument below overrides the
        matching entry.
    impute_plan:
        Fill values to apply.
    encode_plan:
        Category vocabulary and output layout to apply.
    scale_plan:
        Scaling constants to apply.
    date_plan:
        Timestamp expansion to replay.
    outlier_plan:
        Fences to apply. See the notes on how ``'drop'`` is handled.
    binning_plan:
        Band edges to apply.
    feature_select_plan:
        The feature subset to narrow to.
    text_plan:
        Text vocabulary to apply.
    reduce_plan:
        Dimensionality reduction to project through.
    custom_plan:
        A registered custom transform to replay. Its name must still be
        registered in this process.
    resample_plan:
        Accepted for lineage only, never replayed. See the notes.
    split_plan:
        Row membership, needed only in two cases: target encoding writing
        out-of-fold values onto training rows, and an outlier plan that drops
        rows and therefore has to rebuild partitions. Omit it for a pure
        scoring frame, where there is no training membership to respect —
        target encoding then applies the frozen full-training means to every
        row, which is the correct behaviour at inference time.
    roles:
        Column roles to attach when ``data`` is a bare DataFrame. Without them,
        steps that consult roles fall back to dtype inference.

    Returns
    -------
    ApplyPlansResult
        The transformed dataset, which steps ran, which were skipped, and any
        warnings raised along the way.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A plan expects a column the data does not have, a plan object is of an
        unexpected type, or a custom transform is no longer registered.

    Notes
    -----
    **Order is fixed:** dates, impute, outliers, encode, text, binning, scale,
    reduce, feature selection, custom. It matches the order a session applies
    them, and it is not configurable — dates expand first so the columns they
    create can take part in later numeric steps, and selection comes last so it
    sees the finished matrix.

    **Missing plans are skipped, not defaulted.** Only the plans you supply
    run, and the rest are reported in ``skipped``. Compare that list against
    what training actually did; a plan quietly absent at serving time is a
    silent correctness bug.

    **Resampling is never replayed.** It rebalanced the training set to help
    the model learn; applying it at scoring time would fabricate rows. Passing
    the plan records it in the lineage and adds a warning, nothing more.

    **A dropping outlier plan becomes a capping one.** Removing rows during
    scoring would mean refusing to predict for them, which is almost never what
    a caller wants, so the fences are applied by clipping instead. This is
    recorded as a warning.

    See Also
    --------
    buildml.session.Session.apply_preprocess_plans : The session-level entry point.
    buildml.session.Session.predict_from_pipeline : Replay and score in one call.
    """
    resolved = _resolve_plans(
        plans,
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
    )
    dataset = _as_dataset(data, roles=roles)
    warnings: list[str] = []
    applied: list[str] = []
    skipped: list[str] = []
    working_split = split_plan
    if resolved.get("resample") is not None:
        skipped.append("resample")
        warnings.append(
            "ResamplePlan is lineage-only and is not reapplied at score time. "
            "Train-row rewriting happened during Session.resample; score frames "
            "keep their natural row membership."
        )
    if not any(resolved.get(key) is not None for key in SCORE_TIME_ORDER):
        raise ValidationError(
            "No score-time preprocess plans were supplied. Pass fitted plans "
            "from a checkpoint or pipeline bundle (impute/encode/scale/dates/"
            "outliers/text/binning/reduce/feature_select/custom)."
        )
    for step in SCORE_TIME_ORDER:
        plan = resolved.get(step)
        if plan is None:
            continue
        if step == "dates":
            dataset = _apply_dates(dataset, plan)
            applied.append(step)
            continue
        if step == "impute":
            _require_columns(dataset, plan.columns, "Impute")
            dataset = transform_simple_imputer(dataset, plan)
            applied.append(step)
            continue
        if step == "outliers":
            dataset, working_split, step_warnings = _apply_outliers(dataset, plan, working_split)
            warnings.extend(step_warnings)
            applied.append(step)
            continue
        if step == "encode":
            dataset, step_warnings = _apply_encode(dataset, plan, working_split)
            warnings.extend(step_warnings)
            applied.append(step)
            continue
        if step == "text":
            _require_columns(dataset, plan.columns, "Text")
            dataset, _ = transform_text_features(dataset, plan)
            applied.append(step)
            continue
        if step == "binning":
            _require_columns(dataset, plan.columns, "Binning")
            dataset, _ = transform_binning(dataset, plan)
            applied.append(step)
            continue
        if step == "scale":
            _require_columns(dataset, plan.columns, "Scale")
            dataset = transform_scaler(dataset, plan)
            applied.append(step)
            continue
        if step == "reduce":
            _require_columns(dataset, plan.columns, "Reduce")
            dataset, _ = transform_reducer(dataset, plan)
            applied.append(step)
            continue
        if step == "feature_select":
            missing = [c for c in plan.selected_features_ if c not in dataset.columns]
            if missing:
                raise ValidationError(
                    "Feature-select plan expects columns that are missing after "
                    f"earlier score-time steps: {missing}. The held-out schema is "
                    "incompatible with the fitted feature contract."
                )
            dataset, _ = transform_feature_selector(dataset, plan)
            applied.append(step)
            continue
        if step == "custom":
            _require_columns(dataset, plan.columns, "Custom")
            dataset, custom_result = transform_custom(dataset, plan)
            warnings.extend(custom_result.warnings)
            applied.append(step)
    return ApplyPlansResult(
        dataset=dataset,
        applied=tuple(applied),
        skipped=tuple(skipped),
        warnings=warnings,
        split_plan=working_split,
    )


def _resolve_plans(
    plans: dict[str, Any] | None,
    **explicit: Any,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {key: None for key in (*SCORE_TIME_ORDER, "resample")}
    payload = dict(plans or {})
    # Unwrap versioned plans.joblib envelopes.
    if "plans" in payload and isinstance(payload["plans"], dict):
        inner = payload["plans"]
        if any(key in PLAN_KEY_ALIASES for key in inner):
            payload = inner
    for key, value in payload.items():
        if key in {"format", "buildml_version", "plans"}:
            continue
        alias = PLAN_KEY_ALIASES.get(key)
        if alias is None:
            continue
        resolved[alias] = value
    for name, value in explicit.items():
        if value is None:
            continue
        alias = PLAN_KEY_ALIASES.get(name)
        if alias is not None:
            resolved[alias] = value
    return resolved


def _as_dataset(
    data: Dataset | pd.DataFrame,
    *,
    roles: dict[str, ColumnRole | str] | None,
) -> Dataset:
    if isinstance(data, Dataset):
        if roles:
            # Promote Pandas when needed, then rebuild native so score-time
            # transforms keep an honest engine handle.
            frame = data._ensure_pandas()
            data = Dataset.from_transformed(
                data,
                frame,
                schema=data.schema,
                roles=dict(data.roles),
            )
            data.set_roles(roles)
        return data
    if not isinstance(data, pd.DataFrame):
        raise ValidationError("apply_preprocess_plans expects a Dataset or pandas.DataFrame")
    dataset = Dataset.from_pandas(data, schema=schema_from_dataframe(data), source="score_frame")
    if roles:
        dataset.set_roles(roles)
    return dataset


def _require_columns(dataset: Dataset, columns: tuple[str, ...] | list[str], label: str) -> None:
    missing = [c for c in columns if c not in dataset.columns]
    if missing:
        raise ValidationError(
            f"{label} plan columns missing from score frame: {missing}. "
            "Restore date/encode steps first if those columns were derived."
        )


def _apply_dates(dataset: Dataset, plan: DateFeaturePlan) -> Dataset:
    _require_columns(dataset, plan.columns, "Date")
    out, _ = extract_date_features(
        dataset,
        columns=list(plan.columns),
        include_time=plan.include_time,
        drop_original=plan.drop_original,
    )
    # Created-column contract check (names must match the fitted plan).
    missing_created = [c for c in plan.created_columns if c not in out.columns]
    if missing_created:
        raise ValidationError(
            f"Date plan created-column contract mismatch after reapplication: {missing_created}"
        )
    return out


def _apply_outliers(
    dataset: Dataset,
    plan: OutlierPlan,
    split_plan: SplitPlan | None,
) -> tuple[Dataset, SplitPlan | None, list[str]]:
    _require_columns(dataset, plan.columns, "Outlier")
    warnings: list[str] = []
    working = plan
    if plan.action == "drop" and split_plan is None:
        working = OutlierPlan(
            columns=plan.columns,
            method=plan.method,
            action="cap",
            lower_=dict(plan.lower_),
            upper_=dict(plan.upper_),
            n_flagged_train=plan.n_flagged_train,
            n_dropped=0,
            iqr_multiplier=plan.iqr_multiplier,
            zscore_threshold=plan.zscore_threshold,
        )
        warnings.append(
            "OutlierPlan action='drop' cannot rebuild partitions without a SplitPlan; "
            "score-time reapplication uses the frozen fences as caps instead."
        )
    if split_plan is None:
        # Cap/detect without partition rewrite.
        if working.action == "detect":
            return dataset, None, warnings
        frame = dataset._ensure_pandas().copy()
        for column in working.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.clip(
                lower=working.lower_[column],
                upper=working.upper_[column],
            )
        out = Dataset.from_transformed(
            dataset,
            frame,
            schema=schema_from_dataframe(frame),
        )
        return out, None, warnings
    dataset, new_split, _, _ = apply_outlier_plan(dataset, split_plan, working)
    return dataset, new_split, warnings


def _apply_encode(
    dataset: Dataset,
    plan: EncodePlan,
    split_plan: SplitPlan | None,
) -> tuple[Dataset, list[str]]:
    _require_columns(dataset, plan.columns, "Encode")
    if plan.method != "target":
        dataset, _ = transform_encoder(dataset, plan, split_plan=split_plan)
        return dataset, []
    if split_plan is not None:
        dataset, _ = transform_encoder(dataset, plan, split_plan=split_plan)
        return dataset, []
    # Score-only path: apply frozen full-train maps to every row.
    frame = dataset._ensure_pandas().copy()
    prior = float(plan.target_prior_ if plan.target_prior_ is not None else 0.0)
    roles = {k: v for k, v in dataset.roles.items() if k not in plan.columns}
    for column, out_name in zip(plan.columns, plan.feature_names_, strict=True):
        mapping = plan.target_maps_.get(column, {})
        frame[out_name] = frame[column].astype(str).map(mapping).fillna(prior).astype(float)
        roles[out_name] = ColumnRole.FEATURE
    frame = frame.drop(columns=list(plan.columns))
    out = Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
    return out, [
        "Target encoding applied frozen full-train category maps to all score rows "
        "(no SplitPlan supplied for out-of-fold train rewriting)."
    ]
