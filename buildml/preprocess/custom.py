"""Registered custom transforms with an explicit train-fit contract.

Contract
--------
1. ``fit(train_frame, params) -> artifact`` may inspect **train rows only**.
2. ``transform(frame, artifact) -> frame`` must be deterministic given the
   artifact and must not read labels or holdout statistics that were not frozen
   inside the artifact during fit.
3. Fitted plans store ``(name, columns, params, artifact)``. Score-time replay
   requires the same name to remain registered (or the artifact alone must be
   sufficient for ``transform``). Prefer picklable artifacts so pipeline /
   checkpoint ``plans.joblib`` round-trips work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.result import PreprocessResult

FitFn = Callable[[pd.DataFrame, Mapping[str, Any]], Any]
TransformFn = Callable[[pd.DataFrame, Any], pd.DataFrame]
OutputColumnsFn = Callable[[Any, tuple[str, ...]], list[str]]


@dataclass(slots=True)
class CustomTransformSpec:
    """Registered custom transform definition."""

    name: str
    fit: FitFn
    transform: TransformFn
    description: str = ""
    output_columns: OutputColumnsFn | None = None
    drop_input_columns: bool = False
    serializable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "drop_input_columns": self.drop_input_columns,
            "serializable": self.serializable,
            "has_output_columns_fn": self.output_columns is not None,
        }


@dataclass(slots=True)
class CustomTransformPlan:
    """Train-fitted custom transform instance."""

    name: str
    columns: tuple[str, ...]
    params: dict[str, Any]
    feature_names_: tuple[str, ...]
    artifact_: Any = field(repr=False)
    drop_input_columns: bool = False
    serializable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "params": dict(self.params),
            "feature_names_": list(self.feature_names_),
            "drop_input_columns": self.drop_input_columns,
            "serializable": self.serializable,
            "description": self.description,
            "artifact_type": type(self.artifact_).__name__,
        }


_REGISTRY: dict[str, CustomTransformSpec] = {}


def register_transform(
    name: str,
    *,
    fit: FitFn,
    transform: TransformFn,
    description: str = "",
    output_columns: OutputColumnsFn | None = None,
    drop_input_columns: bool = False,
    serializable: bool = True,
    overwrite: bool = False,
) -> CustomTransformSpec:
    """Register a custom transform under ``name``.

    Parameters
    ----------
    fit:
        ``fit(train_frame[columns], params) -> artifact``. Must not use holdout
        rows. Callers receive only the selected columns.
    transform:
        ``transform(frame[columns], artifact) -> DataFrame`` with the same index
        as the input frame. May return new columns, replacements, or both.
    output_columns:
        Optional ``(artifact, input_columns) -> list[str]`` naming created
        columns. When omitted, transform output column names are recorded after
        the first fit.
    drop_input_columns:
        When True, input columns are removed after transform unless they also
        appear in the transform output.
    serializable:
        When True, the fitted artifact is expected to be joblib-picklable for
        pipeline/checkpoint persistence. Set False for process-local callables.
    overwrite:
        Replace an existing registration with the same name.
    """
    key = str(name).strip()
    if not key:
        raise ValidationError("Custom transform name must be non-empty")
    if key in _REGISTRY and not overwrite:
        raise ValidationError(
            f"Custom transform '{key}' is already registered. Pass overwrite=True to replace it."
        )
    if not callable(fit) or not callable(transform):
        raise ValidationError("fit and transform must be callable")
    spec = CustomTransformSpec(
        name=key,
        fit=fit,
        transform=transform,
        description=str(description or ""),
        output_columns=output_columns,
        drop_input_columns=bool(drop_input_columns),
        serializable=bool(serializable),
    )
    _REGISTRY[key] = spec
    return spec


def unregister_transform(name: str) -> None:
    """Remove a registered transform (testing / process cleanup)."""
    _REGISTRY.pop(str(name), None)


def get_transform(name: str) -> CustomTransformSpec:
    """Return a registered transform spec or raise ``ValidationError``."""
    try:
        return _REGISTRY[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValidationError(
            f"Unknown custom transform '{name}'. Registered: {known}. "
            "Call buildml.preprocess.register_transform(...) first."
        ) from exc


def list_transforms() -> tuple[CustomTransformSpec, ...]:
    """Return registered custom transforms in name order."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def fit_custom_transform(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    name: str,
    columns: list[str],
    params: Mapping[str, Any] | None = None,
) -> CustomTransformPlan:
    """Fit a registered custom transform on train rows only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    spec = get_transform(name)
    cols = tuple(validate_column_names(list(columns), dataset.columns))
    if not cols:
        raise ValidationError("columns must be a non-empty list for apply_custom_transform")
    train = frame_for_partition(dataset, split_plan, "train")
    safe_params = dict(params or {})
    artifact = spec.fit(train[list(cols)].copy(), safe_params)
    # Probe transform on train to freeze the output schema contract.
    probe = spec.transform(train[list(cols)].copy(), artifact)
    if not isinstance(probe, pd.DataFrame):
        raise ValidationError(
            f"Custom transform '{spec.name}' transform() must return a pandas.DataFrame"
        )
    if not probe.index.equals(train.index):
        raise ValidationError(
            f"Custom transform '{spec.name}' must preserve the input row index"
        )
    if spec.output_columns is not None:
        feature_names = [str(c) for c in spec.output_columns(artifact, cols)]
    else:
        feature_names = [str(c) for c in probe.columns]
    if not feature_names:
        raise ValidationError(f"Custom transform '{spec.name}' produced no output columns")
    return CustomTransformPlan(
        name=spec.name,
        columns=cols,
        params=safe_params,
        feature_names_=tuple(feature_names),
        artifact_=artifact,
        drop_input_columns=spec.drop_input_columns,
        serializable=spec.serializable,
        description=spec.description,
    )


def transform_custom(
    dataset: Dataset,
    plan: CustomTransformPlan,
) -> tuple[Dataset, PreprocessResult]:
    """Apply a train-fitted custom plan using the registered transform."""
    spec = get_transform(plan.name)
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Custom transform columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    transformed = spec.transform(frame[list(plan.columns)].copy(), plan.artifact_)
    if not isinstance(transformed, pd.DataFrame):
        raise ValidationError(
            f"Custom transform '{plan.name}' transform() must return a pandas.DataFrame"
        )
    if not transformed.index.equals(frame.index):
        raise ValidationError(
            f"Custom transform '{plan.name}' must preserve the input row index"
        )
    # Align to the fitted column contract.
    for name in plan.feature_names_:
        if name not in transformed.columns:
            raise ValidationError(
                f"Custom transform '{plan.name}' missing expected output column '{name}'"
            )
    transformed = transformed.loc[:, list(plan.feature_names_)]

    roles = dict(dataset.roles)
    if plan.drop_input_columns:
        drop = [c for c in plan.columns if c not in plan.feature_names_]
        frame = frame.drop(columns=drop, errors="ignore")
        for column in drop:
            roles.pop(column, None)
    # Overwrite / add output columns.
    for column in plan.feature_names_:
        frame[column] = transformed[column]
        if column not in roles or roles.get(column) not in {
            ColumnRole.TARGET,
            ColumnRole.ID,
            ColumnRole.GROUP,
            ColumnRole.TIME,
            ColumnRole.WEIGHT,
        }:
            roles[column] = ColumnRole.FEATURE

    new_dataset = Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
    warnings: list[str] = []
    if not plan.serializable:
        warnings.append(
            f"Custom transform '{plan.name}' is marked serializable=False; "
            "pipeline/checkpoint plan persistence may fail or be incomplete."
        )
    return new_dataset, _build_result(plan, warnings=warnings)


def _build_result(
    plan: CustomTransformPlan,
    *,
    warnings: list[str] | None = None,
) -> PreprocessResult:
    evidence = [
        Evidence(
            key="apply_custom_transform.contract",
            kind=EvidenceKind.METRIC,
            summary="Train-fitted custom transform contract.",
            value={
                "name": plan.name,
                "columns": list(plan.columns),
                "feature_names": list(plan.feature_names_),
                "params": dict(plan.params),
                "serializable": plan.serializable,
            },
            source="train.custom_transform",
            limitations=(
                "BuildML enforces train-only fit scope; "
                "correctness of the callable is caller-owned.",
            ),
        )
    ]
    findings = [
        Finding(
            key="apply_custom_transform.applied",
            title=f"Custom transform '{plan.name}' fitted on train",
            detail=(
                f"Applied registered transform '{plan.name}' on {len(plan.columns)} "
                f"column(s); output width={len(plan.feature_names_)}."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations = [
        Recommendation(
            key="apply_custom_transform.reregister",
            title="Keep the transform registered for score-time replay",
            rationale=(
                "Pipeline and checkpoint reload require the same name to be registered "
                "in-process (unless transform needs only the pickled artifact)."
            ),
            priority=ActionPriority.NEXT,
            action=Action(
                key="apply_custom_transform.list-action",
                label="buildml.preprocess.list_transforms()",
                operation="list_transforms",
                parameters={},
            ),
            based_on=("apply_custom_transform.applied",),
            caveats=("Non-serializable artifacts will not survive process restart.",),
        )
    ]
    return PreprocessResult(
        operation="apply_custom_transform",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Custom transform '{plan.name}' fitted on train and applied to all rows.",
            plan.description or "No description was registered for this transform.",
        ],
        limitations=[
            "Caller-supplied fit/transform must honor the train-only contract.",
            "Unknown categories or score-time schema drift are transform-specific.",
        ],
        recommendations=recommendations,
        methods=[f"Registered transform '{plan.name}'."],
        warnings=list(warnings or []),
    )
