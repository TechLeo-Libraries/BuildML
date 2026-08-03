"""Add your own preprocessing step without giving up the leakage guarantees.

The built-in steps cover the common ground, but domain work always needs
something specific: a currency conversion using rates from your training
period, a geographic clustering, a bespoke text cleaner. This is where those
go — registered once, then usable anywhere the built-in steps are, and subject
to the same discipline.

That discipline is a contract in three parts.

First, your ``fit`` function sees the training rows and nothing else. It
receives only the columns you nominated, and it returns an *artifact* — a plain
object holding everything the step learned. Whatever your step needs to know
must end up in there.

Second, your ``transform`` function receives a frame and that artifact, and
must be deterministic given the pair. It must not compute anything new from the
data it is transforming: reading the incoming batch's mean, or its labels, is
precisely the leak this arrangement exists to prevent. If the batch could
change the answer, the step is not reproducible at inference time either.

Third, the fitted plan records the transform's name alongside the artifact, so
replaying it later requires that name to still be registered. Make the artifact
picklable and the whole thing round-trips through a saved pipeline; leave it
unpicklable and you are limited to the current process, which the registration
records honestly via its ``serializable`` flag.
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
    """A registered transform: the two functions plus how to treat their output.

    This is the recipe, not an instance of it. One spec can be fitted many
    times against different datasets and column sets, each producing its own
    :class:`CustomTransformPlan`.

    Attributes
    ----------
    name:
        The registry key, used to look the spec up when replaying a plan.
    fit:
        ``fit(train_frame, params) -> artifact``. Receives only the nominated
        columns of the training rows.
    transform:
        ``transform(frame, artifact) -> DataFrame``. Must preserve the input's
        row index so results stay joinable.
    description:
        What the transform does, surfaced in listings and reports.
    output_columns:
        Optional ``(artifact, input_columns) -> list[str]`` declaring the
        output names up front. Worth providing when the output width depends on
        what was learned, since it removes the need to infer names from a probe
        run.
    drop_input_columns:
        Whether the source columns are removed after transforming. Columns that
        also appear in the output are kept regardless.
    serializable:
        Whether the artifact is expected to survive joblib pickling. Set this
        to ``False`` for a closure or an open handle, and the plan will
        correctly report that it cannot be saved rather than failing later.
    """

    name: str
    fit: FitFn
    transform: TransformFn
    description: str = ""
    output_columns: OutputColumnsFn | None = None
    drop_input_columns: bool = False
    serializable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the registration's metadata as plain JSON-safe values.

        The two callables are omitted, since functions do not serialise; what
        remains describes the transform for listings and model cards.

        Returns
        -------
        dict
            Keys ``name``, ``description``, ``drop_input_columns``,
            ``serializable``, and ``has_output_columns_fn``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "drop_input_columns": self.drop_input_columns,
            "serializable": self.serializable,
            "has_output_columns_fn": self.output_columns is not None,
        }


@dataclass(slots=True)
class CustomTransformPlan:
    """One fitted instance of a registered transform, ready to replay.

    Attributes
    ----------
    name:
        The registered transform this came from. It must still be registered
        when the plan is replayed, since the transform function itself is not
        stored here.
    columns:
        The source columns the transform was fitted against.
    params:
        The configuration passed at fit time, kept so the fit is reproducible.
    feature_names_:
        The output columns, frozen at fit time. This is the contract: a later
        transform producing different columns is an error rather than a silent
        change of shape.
    artifact_:
        Whatever your ``fit`` function returned — the learned state.
    drop_input_columns:
        Whether the source columns are removed after transforming.
    serializable:
        Whether this plan can be saved. ``False`` means the artifact will not
        pickle, so the plan is confined to this process.
    description:
        Copied from the spec, so the plan explains itself in a report.
    """

    name: str
    columns: tuple[str, ...]
    params: dict[str, Any]
    feature_names_: tuple[str, ...]
    artifact_: Any = field(repr=False)
    drop_input_columns: bool = False
    serializable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the plan's metadata as plain JSON-safe values.

        The artifact itself is omitted — it can be any object — and replaced by
        its type name, which is enough for a reader to understand what was
        stored.

        Returns
        -------
        dict
            The configuration and output layout, plus ``artifact_type``.
        """
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
    """Make your own preprocessing step available to the rest of the library.

    Registration is process-wide and takes effect immediately. Once registered,
    the transform can be fitted with :func:`fit_custom_transform` and replayed
    from a saved plan, exactly like a built-in step.

    Parameters
    ----------
    name:
        The key to register under. This is stored in every plan the transform
        produces, so treat it as a stable identifier — renaming it breaks
        replay of previously saved plans.
    fit:
        ``fit(train_frame, params) -> artifact``. Receives a copy of the
        nominated columns from the training rows only, and whatever you passed
        as ``params``. Return anything: a dict of learned constants, a fitted
        scikit-learn object, a lookup table. Everything the transform will need
        must be in it.
    transform:
        ``transform(frame, artifact) -> DataFrame``. Must return a DataFrame
        carrying the same row index as its input, and must depend on nothing
        beyond its two arguments — no globals that change, no statistics
        recomputed from the incoming frame.
    description:
        A sentence explaining what the transform does, shown by
        :func:`list_transforms` and included in reports.
    output_columns:
        Optional ``(artifact, input_columns) -> list[str]`` declaring the
        output names. Without it, the names are inferred by running the
        transform once on the training rows at fit time; supplying it avoids
        that probe.
    drop_input_columns:
        Remove the source columns after transforming. Set this when your
        transform replaces its inputs rather than adding to them. A column that
        appears in the output is kept regardless of this setting.
    serializable:
        Whether the artifact will survive joblib pickling. Leave it ``True`` for
        ordinary data and fitted estimators. Set it ``False`` when the artifact
        closes over a function, a file handle, or a network client, so plans
        report honestly that they cannot be saved.
    overwrite:
        Allow replacing an existing registration. Off by default, so a name
        collision is an error rather than a silent substitution that would
        change the behaviour of an already-fitted plan.

    Returns
    -------
    CustomTransformSpec
        The stored registration.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The name is blank, ``fit`` or ``transform`` is not callable, or the
        name is taken and ``overwrite`` is ``False``.

    Notes
    -----
    **Registration is not persisted.** It lives in the current process, so a
    script that loads a saved pipeline containing a custom plan must register
    the transform again before replaying it. Registering at import time in a
    module both training and serving import is the reliable pattern.

    **Custom transforms cannot go inside a cross-validation fold.** They are
    session-wide only, because an arbitrary callable cannot be verified to be
    fold-safe. Where a step can be expressed with the built-ins, put it in a
    :class:`~buildml.preprocess.fold.PreprocessRecipe` instead.

    Examples
    --------
    >>> def fit_log_offset(frame, params):  # doctest: +SKIP
    ...     return {"offset": float(frame.min().min())}
    >>> def apply_log(frame, artifact):  # doctest: +SKIP
    ...     import numpy as np
    ...     return np.log1p(frame - artifact["offset"])
    >>> register_transform(  # doctest: +SKIP
    ...     "log_shift",
    ...     fit=fit_log_offset,
    ...     transform=apply_log,
    ...     description="Shift to non-negative, then log1p.",
    ... )

    See Also
    --------
    fit_custom_transform : Fits a registered transform to a dataset.
    list_transforms : What is currently registered.
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
    """Remove a transform from the registry.

    Mainly for tests, which need to leave the process-wide registry as they
    found it, and for reloading a definition during development. Removing a
    name does not invalidate plans already fitted from it, but those plans can
    no longer be replayed until the name is registered again.

    Parameters
    ----------
    name:
        The registration to remove. Unknown names are ignored rather than
        raising, so cleanup code can run unconditionally.

    See Also
    --------
    register_transform : The inverse.
    """
    _REGISTRY.pop(str(name), None)


def get_transform(name: str) -> CustomTransformSpec:
    """Look up a registered transform by name.

    Used internally whenever a plan is fitted or replayed, and useful directly
    when you want to inspect a registration.

    Parameters
    ----------
    name:
        The registered name.

    Returns
    -------
    CustomTransformSpec
        The registration.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        Nothing is registered under that name. The message lists what is,
        which usually reveals either a typo or a module that was never
        imported.

    See Also
    --------
    list_transforms : Everything currently registered.
    """
    try:
        return _REGISTRY[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValidationError(
            f"Unknown custom transform '{name}'. Registered: {known}. "
            "Call buildml.preprocess.register_transform(...) first."
        ) from exc


def list_transforms() -> tuple[CustomTransformSpec, ...]:
    """List every transform currently registered in this process.

    The first thing to check when a replay fails with an unknown-transform
    error: if the name is absent, the module that registers it has not been
    imported.

    Returns
    -------
    tuple of CustomTransformSpec
        All registrations, sorted by name. Empty before anything is
        registered — there are no built-in entries here.

    See Also
    --------
    register_transform : Adds an entry.
    """
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def fit_custom_transform(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    name: str,
    columns: list[str],
    params: Mapping[str, Any] | None = None,
) -> CustomTransformPlan:
    """Fit a registered transform to the training rows and freeze its output shape.

    Your ``fit`` function runs against the training rows, and the returned
    artifact is captured. The transform is then run once on those same rows as
    a probe, which both verifies the contract — a DataFrame back, with the row
    index intact — and records the output column names, so a later replay
    producing a different shape is caught rather than quietly reshaping the
    frame.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are passed
        to your fit function.
    split_plan:
        The split defining the training rows. Required, for the same reason it
        is required by every other fit in this package.
    name:
        The registered transform to fit.
    columns:
        Which columns your transform operates on. Required and non-empty —
        unlike the built-in steps there is no sensible default, since only you
        know what the transform expects.
    params:
        Configuration passed through to your fit function unchanged, and stored
        on the plan so the fit is reproducible.

    Returns
    -------
    CustomTransformPlan
        The artifact and the frozen output layout, ready to apply.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        The transform is not registered, ``columns`` is empty or names a column
        that does not exist, the transform returned something other than a
        DataFrame, it changed the row index, or it produced no columns.

    Notes
    -----
    The index check is not a formality. A transform that resets, sorts, or
    filters the index silently misaligns every row against its label, which
    produces a model that trains without complaint and predicts nonsense.

    See Also
    --------
    transform_custom : Applies the plan produced here.
    register_transform : Defines the transform first.
    """
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
    """Replay a fitted custom transform across every row.

    Looks the transform up by name, runs it with the stored artifact, and
    checks the result against the layout frozen at fit time.

    Parameters
    ----------
    dataset:
        The dataset to transform. Every column the plan names must be present.
    plan:
        A plan from :func:`fit_custom_transform`.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.preprocess.result.PreprocessResult)
        The transformed dataset, and a narrated record of what the step
        produced.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The transform is no longer registered, a required column is missing,
        the transform did not return a DataFrame, or it changed the row index.

    Notes
    -----
    Registration must be in place before this runs. A saved pipeline stores the
    artifact but not the function, so a serving process has to import the
    module that registers the transform before it can score anything.

    See Also
    --------
    fit_custom_transform : Produces the plan this consumes.
    """
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
