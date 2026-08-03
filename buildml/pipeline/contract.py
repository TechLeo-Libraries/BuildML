"""Check the incoming frame looks like the training frame, before predicting.

A model will produce a number for almost any input. Rename a column and it may
raise; reorder them and it may not; pass strings where floats were expected and
pandas may quietly upcast to object, at which point the estimator sees garbage
and returns a confident answer. None of these announce themselves.

A schema contract is what the training frame looked like — the columns, their
types, their roles, their nullability — saved alongside the model and checked
before each batch of predictions. It turns a silent wrong answer into a clear
error naming the column.

Two stages are checked, because two different frames are involved. The *input*
stage checks the raw frame you supply, before the fitted transforms run. The
*features* stage checks what comes out the other side, against what the
estimator was actually fitted on. A mismatch at input usually means the caller
sent the wrong thing; a mismatch at features usually means a transform behaved
differently than at training time, most often because a category it had never
seen produced a different set of one-hot columns.

Type checking is by *family* rather than exact dtype, deliberately. A column
that was ``int64`` in training and arrives as ``Int64`` from a Parquet round
trip is the same column. Insisting on exact equality would make the contract
fire constantly on differences that do not matter, and a check that cries wolf
gets turned off.

See Also
--------
buildml.pipeline.score : Where these checks run during prediction.
buildml.pipeline.card : The human-readable half of the same record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole, TableSchema

SCHEMA_CONTRACT_FORMAT = "buildml.schema_contract.v1"
SCHEMA_CONTRACT_FILENAME = "schema_contract.json"

DtypeFamily = Literal["numeric", "bool", "string", "datetime", "categorical", "other"]
# Roles required at score time when present on the training contract.
_SCORE_REQUIRED_ROLES = frozenset(
    {
        ColumnRole.FEATURE.value,
        ColumnRole.GROUP.value,
        ColumnRole.TIME.value,
        ColumnRole.ID.value,
    }
)


@dataclass(slots=True)
class SchemaContract:
    """What the training frame looked like, recorded so inputs can be checked.

    Two column lists, for the two stages. ``columns`` is the raw frame expected
    before transforms run; ``feature_columns`` is what the estimator was fitted
    on afterwards. They differ whenever encoding, binning, or feature selection
    is involved, and conflating them produces confusing failures — a one-hot
    output column is not something a caller can supply.

    Parameters
    ----------
    columns:
        Ordered input columns expected before plan replay (score-time raw frame).
    dtypes:
        Declared dtype strings from the training schema.
    dtype_families:
        Coarse families used for score-time type checks.
    roles:
        Column roles recorded at save time.
    feature_columns:
        Estimator feature contract after plan replay.
    target_column:
        Training target name when known (not required at score time).
    required_roles:
        Role names that must appear among score-time input columns when those
        roles were present at save time (feature/group/time/id).
    nullable:
        Declared nullability from the training schema when known.
    encoded_numeric_columns:
        Columns that are numeric after encode/bin replay (one-hot / target /
        ordinal outputs). Used for dtype-family compatibility at the features
        stage.

    Notes
    -----
    **The target is excluded from ``columns`` on purpose.** Score-time frames do
    not have labels — that is the point of scoring — so requiring the target
    would fail every real prediction request.

    **Roles are required by *kind*, not by name.** If training had a group
    column, the contract requires *some* column with the group role, not that
    exact name. Column names change between systems; the role is the thing that
    has to survive.

    See Also
    --------
    build_schema_contract : Deriving one at save time.
    validate_score_frame : Checking a frame against one.
    """

    columns: tuple[str, ...]
    dtypes: dict[str, str] = field(default_factory=dict)
    dtype_families: dict[str, DtypeFamily] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    feature_columns: tuple[str, ...] = ()
    target_column: str | None = None
    required_roles: tuple[str, ...] = ()
    nullable: dict[str, bool] = field(default_factory=dict)
    encoded_numeric_columns: tuple[str, ...] = ()
    format: str = SCHEMA_CONTRACT_FORMAT
    buildml_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        """Convert the contract to JSON-safe plain data.

        Written to ``schema_contract.json`` in the bundle, sorted and indented
        so two versions of a model can be diffed to see exactly how the expected
        input changed.

        Returns
        -------
        dict
            Every field, with tuples as lists and mappings copied.

        See Also
        --------
        from_dict : The inverse.
        """
        return {
            "format": self.format,
            "buildml_version": self.buildml_version,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "dtype_families": dict(self.dtype_families),
            "roles": dict(self.roles),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "required_roles": list(self.required_roles),
            "nullable": dict(self.nullable),
            "encoded_numeric_columns": list(self.encoded_numeric_columns),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchemaContract:
        """Rebuild a contract from stored JSON, refusing an unknown format.

        Unlike the model card, the format label is enforced here. A card that is
        partly understood still informs; a contract that is partly understood
        would validate against incomplete expectations and pass frames it should
        reject, which is worse than not checking at all.

        Parameters
        ----------
        payload:
            The parsed contents of ``schema_contract.json``.

        Returns
        -------
        SchemaContract
            The reconstructed contract. Absent optional fields default empty,
            and a missing format label is treated as the current one, since
            unlabelled contracts predate versioning.

        Raises
        ------
        ValidationError
            If the payload is not a mapping, or carries a format label this
            version does not know.

        See Also
        --------
        to_dict : The inverse.
        """
        if not isinstance(payload, dict):
            raise ValidationError("schema contract payload must be a mapping")
        fmt = payload.get("format")
        if fmt not in {SCHEMA_CONTRACT_FORMAT, None}:
            raise ValidationError(f"Unsupported schema contract format '{fmt}'")
        families_raw = payload.get("dtype_families") or {}
        families: dict[str, DtypeFamily] = {}
        for key, value in families_raw.items():
            families[str(key)] = str(value)  # type: ignore[assignment]
        return cls(
            columns=tuple(str(c) for c in payload.get("columns", [])),
            dtypes={str(k): str(v) for k, v in (payload.get("dtypes") or {}).items()},
            dtype_families=families,
            roles={str(k): str(v) for k, v in (payload.get("roles") or {}).items()},
            feature_columns=tuple(str(c) for c in payload.get("feature_columns", [])),
            target_column=(
                None
                if payload.get("target_column") is None
                else str(payload.get("target_column"))
            ),
            required_roles=tuple(str(r) for r in payload.get("required_roles", [])),
            nullable={
                str(k): bool(v) for k, v in (payload.get("nullable") or {}).items()
            },
            encoded_numeric_columns=tuple(
                str(c) for c in payload.get("encoded_numeric_columns", [])
            ),
            format=str(fmt or SCHEMA_CONTRACT_FORMAT),
            buildml_version=str(payload.get("buildml_version") or __version__),
        )


@dataclass(slots=True)
class SchemaContractValidation:
    """Everything wrong with a frame, not just the first thing.

    Reporting all problems at once matters here more than usual. Fixing a score
    frame is often an integration task against a system you do not control, and
    discovering three missing columns and two type mismatches in one pass beats
    five deploy-and-retry cycles.

    Attributes
    ----------
    ok:
        Whether the frame passes. Note that ``ok`` can be ``True`` alongside
        warnings, and is ``True`` when there was no contract to check against.
    missing_columns:
        Expected columns absent from the frame. Always a failure.
    extra_columns:
        Columns the contract does not mention. A warning by default, since
        upstream systems commonly carry extra fields.
    wrong_type_columns:
        Family mismatches, each naming the column, what was expected, what
        arrived, and the actual dtype.
    missing_roles:
        Required roles with no column present to fill them.
    coerced_columns:
        Columns :func:`coerce_score_frame` converted. Populated only by that
        function.
    warnings:
        Non-fatal observations, including all-null columns that were
        non-nullable in training.
    stage:
        ``'input'`` or ``'features'``, so a failure can be traced to the right
        side of the transforms.
    contract_present:
        ``False`` for a bundle with no contract, meaning nothing was checked.

    Notes
    -----
    **``ok=True`` with ``contract_present=False`` means unchecked, not valid.**
    Older bundles have no contract, and passing a frame through unvalidated is
    the compatible behaviour — but it is not evidence the frame was right.

    See Also
    --------
    validate_score_frame : Producing this.
    raise_for_contract : Turning a failure into an exception.
    """

    ok: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    wrong_type_columns: list[dict[str, str]] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    coerced_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage: str = "input"
    contract_present: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation outcome to plain data.

        Suitable for logging or returning in an HTTP error body, so a caller who
        sent a bad frame gets the specifics rather than "invalid input".

        Returns
        -------
        dict
            Every field, with lists copied.
        """
        return {
            "ok": self.ok,
            "missing_columns": list(self.missing_columns),
            "extra_columns": list(self.extra_columns),
            "wrong_type_columns": list(self.wrong_type_columns),
            "missing_roles": list(self.missing_roles),
            "coerced_columns": list(self.coerced_columns),
            "warnings": list(self.warnings),
            "stage": self.stage,
            "contract_present": self.contract_present,
        }


def dtype_family(dtype: Any) -> DtypeFamily:
    """Reduce a dtype to one of six families, so equivalent types compare equal.

    Exact dtype comparison is the wrong test for a score-time contract. The same
    column can be ``int64`` from pandas, ``Int64`` from a nullable read,
    ``int32`` from Arrow, and ``Float64`` from Polars — all of them the same
    column as far as a model is concerned. Comparing families keeps the check
    meaningful without firing on round-trip noise.

    Matching is by substring on the lowercased name, which handles the many
    spellings the ecosystem produces without enumerating them.

    Parameters
    ----------
    dtype:
        A dtype object or its string name. Anything with a ``name`` attribute is
        read from that; otherwise the value is stringified.

    Returns
    -------
    str
        One of ``'numeric'``, ``'bool'``, ``'string'``, ``'datetime'``,
        ``'categorical'``, or ``'other'``.

    Notes
    -----
    **Booleans are checked before numerics** because ``'bool'`` would otherwise
    be caught by neither and fall through, and because a boolean column is
    meaningfully different from an integer one even though the two are
    interchangeable at score time.

    **``'other'`` is a genuine answer, not a failure.** Object columns holding
    lists, or exotic extension types, land here and are treated as compatible
    with everything, since nothing useful can be asserted about them.

    Examples
    --------
    >>> import numpy as np
    >>> dtype_family(np.dtype("int64"))
    'numeric'
    >>> dtype_family("Float64")
    'numeric'
    >>> dtype_family("datetime64[ns]")
    'datetime'
    >>> dtype_family("object")
    'string'
    >>> dtype_family("complex128")
    'other'

    See Also
    --------
    families_compatible : Deciding whether two families may substitute.
    """
    name = str(getattr(dtype, "name", dtype)).lower()
    if "bool" in name:
        return "bool"
    if "datetime" in name or name.startswith("date"):
        return "datetime"
    if "category" in name or "categorical" in name:
        return "categorical"
    # Encoded / engine dtypes often surface as Float64, Int64, Utf8, etc.
    if any(
        token in name
        for token in (
            "int",
            "float",
            "uint",
            "double",
            "number",
            "decimal",
        )
    ):
        return "numeric"
    if any(token in name for token in ("object", "string", "str", "unicode", "utf")):
        return "string"
    return "other"


def families_compatible(expected: DtypeFamily, actual: DtypeFamily) -> bool:
    """Decide whether an arriving family can stand in for the expected one.

    Identical families always match. Beyond that, two substitutions are allowed,
    both because they happen constantly in practice and neither changes what the
    model sees.

    Strings and categoricals are interchangeable: whether pandas represents a
    column as ``object`` or ``category`` depends on how it was read, not on the
    data. Booleans and numerics are interchangeable because ``True``/``False``
    survives a CSV or Arrow round trip as ``1``/``0`` more often than not.

    Anything involving ``'other'`` passes, since nothing can be asserted about a
    family that could not be identified, and failing on it would reject valid
    frames for no reason.

    Parameters
    ----------
    expected:
        The family recorded in the contract.
    actual:
        The family of the column that arrived.

    Returns
    -------
    bool
        Whether the substitution is acceptable.

    Notes
    -----
    **Compatibility is symmetric.** The pair is compared as a set, so it does
    not matter which side is which.

    **This is a deliberately loose check.** It catches a string where a number
    belongs, or a date where a category belongs — the mistakes that produce
    nonsense predictions — and lets through the representational differences
    that do not.

    Examples
    --------
    >>> families_compatible("numeric", "numeric")
    True
    >>> families_compatible("string", "categorical")
    True
    >>> families_compatible("bool", "numeric")
    True
    >>> families_compatible("numeric", "string")
    False
    >>> families_compatible("numeric", "other")
    True

    See Also
    --------
    dtype_family : Producing the families being compared.
    """
    if expected == "other" or actual == "other":
        return True
    if expected == actual:
        return True
    compatible = {
        frozenset({"string", "categorical"}),
        # Bool often arrives as 0/1 ints from CSV / Arrow round-trips.
        frozenset({"bool", "numeric"}),
    }
    return frozenset({expected, actual}) in compatible


def input_columns_from_plans(plans: dict[str, Any] | None) -> list[str]:
    """Work out which raw columns a caller must supply, from the plans themselves.

    The plans know what they consume, so the contract can be derived rather than
    hand-written — and a derived list does not drift out of date the way a
    hand-written one does.

    The subtlety is exclusion. An encode plan reads ``city`` and produces
    ``city_London``, ``city_Paris``, and so on. Those outputs are *not* inputs;
    requiring them would demand that the caller do the encoding the bundle
    exists to perform. Engineered names are collected across all plans and
    subtracted from the result.

    Parameters
    ----------
    plans:
        The fitted plans keyed by name, as stored in a bundle. ``None`` or empty
        yields an empty list.

    Returns
    -------
    list of str
        The raw input columns in first-seen order, deduplicated, with
        engineered outputs removed.

    Notes
    -----
    **Order is first-seen across plans, not the frame's order.** It is a set of
    requirements rather than a layout; column *presence* is what gets checked,
    not position.

    **An empty result means no plan declared its inputs**, which is why
    :func:`~buildml.pipeline.bundle.save_pipeline_bundle` falls back to the
    estimator's feature columns rather than writing an empty contract.

    See Also
    --------
    build_schema_contract : Where this feeds in.
    """
    if not plans:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    engineered: set[str] = set()

    encode_plan = plans.get("encode_plan")
    if encode_plan is not None:
        names = getattr(encode_plan, "feature_names_", None)
        if names is None and isinstance(encode_plan, dict):
            names = encode_plan.get("feature_names_")
        engineered.update(str(c) for c in names or [])

    binning_plan = plans.get("binning_plan")
    if binning_plan is not None:
        labels = getattr(binning_plan, "labels_", None)
        if labels is None and isinstance(binning_plan, dict):
            labels = binning_plan.get("labels_")
        if isinstance(labels, dict):
            for values in labels.values():
                engineered.update(str(v) for v in values or [])
        created = getattr(binning_plan, "created_columns", None)
        if created is None and isinstance(binning_plan, dict):
            created = binning_plan.get("created_columns")
        engineered.update(str(c) for c in created or [])

    date_plan = plans.get("date_plan")
    if date_plan is not None:
        created = getattr(date_plan, "created_columns", None)
        if created is None and isinstance(date_plan, dict):
            created = date_plan.get("created_columns")
        engineered.update(str(c) for c in created or [])

    text_plan = plans.get("text_plan")
    if text_plan is not None:
        names = getattr(text_plan, "feature_names_", None)
        if names is None and isinstance(text_plan, dict):
            names = text_plan.get("feature_names_")
        engineered.update(str(c) for c in names or [])

    reduce_plan = plans.get("reduce_plan")
    if reduce_plan is not None:
        names = getattr(reduce_plan, "feature_names_", None)
        if names is None and isinstance(reduce_plan, dict):
            names = reduce_plan.get("feature_names_")
        engineered.update(str(c) for c in names or [])

    custom_plan = plans.get("custom_plan")
    if custom_plan is not None:
        names = getattr(custom_plan, "feature_names_", None)
        if names is None and isinstance(custom_plan, dict):
            names = custom_plan.get("feature_names_")
        engineered.update(str(c) for c in names or [])

    # Feature-select operates after transforms; its selected names are not raw inputs.
    key_attrs = (
        ("impute_plan", "columns"),
        ("encode_plan", "columns"),
        ("scale_plan", "columns"),
        ("date_plan", "columns"),
        ("outlier_plan", "columns"),
        ("binning_plan", "columns"),
        ("text_plan", "columns"),
        ("reduce_plan", "columns"),
        ("custom_plan", "columns"),
    )
    for plan_key, attr in key_attrs:
        plan = plans.get(plan_key)
        if plan is None:
            continue
        values = getattr(plan, attr, None)
        if values is None and isinstance(plan, dict):
            values = plan.get(attr)
        for column in values or []:
            name = str(column)
            if name in engineered or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def build_schema_contract(
    *,
    schema: TableSchema | dict[str, Any] | None,
    roles: dict[str, ColumnRole | str] | None,
    feature_columns: tuple[str, ...] | list[str],
    target_column: str | None,
    input_columns: list[str] | tuple[str, ...] | None = None,
    encoded_numeric_columns: list[str] | tuple[str, ...] | None = None,
) -> SchemaContract:
    """Derive the contract from what was known at training time.

    Assembles the expected input from the training schema, the roles, and the
    plans' declared inputs, then makes three adjustments that matter.

    The target is removed, because score-time frames have no labels. Feature
    columns are ordered first when roles are known, so the important columns
    lead the list in error messages and in the saved JSON. And columns that are
    numeric only *after* encoding are recorded separately, so the features-stage
    check expects numbers from a column that was a string at input.

    Parameters
    ----------
    schema:
        The training schema, as a :class:`~buildml.core.types.TableSchema` or
        its dictionary form. When ``None``, types cannot be checked and every
        column gets the permissive ``'other'`` family.
    roles:
        Column roles at training time. Without them the contract cannot require
        roles at score time, only columns.
    feature_columns:
        What the estimator was fitted on — the features-stage expectation.
    target_column:
        The training target, excluded from the expected inputs.
    input_columns:
        Override the raw inputs, normally from
        :func:`input_columns_from_plans`. When given, it replaces whatever the
        schema implied.
    encoded_numeric_columns:
        Feature columns that are numeric post-transform. Inferred when omitted,
        by taking feature columns that are already numeric or that do not appear
        among the inputs — the latter being engineered outputs.

    Returns
    -------
    SchemaContract
        The contract, ready to save.

    Notes
    -----
    **A contract with no schema still checks column presence.** Types are
    skipped, roles are skipped, and missing columns are still caught, which is
    the most common failure anyway.

    **The inference of ``encoded_numeric_columns`` is a heuristic**, and a good
    one for the standard plans. Pass the list explicitly when a custom transform
    produces columns that do not follow the pattern.

    See Also
    --------
    validate_score_frame : Using the contract.
    """
    nullable: dict[str, bool] = {}
    if isinstance(schema, TableSchema):
        fields = schema.fields
        columns = tuple(schema.columns)
        dtypes = {f.name: f.dtype for f in fields}
        nullable = {f.name: bool(f.nullable) for f in fields}
    elif isinstance(schema, dict):
        table = TableSchema.from_dict(schema)
        columns = tuple(table.columns)
        dtypes = {f.name: f.dtype for f in table.fields}
        nullable = {f.name: bool(f.nullable) for f in table.fields}
    else:
        columns = tuple(str(c) for c in (input_columns or feature_columns))
        dtypes = {}

    role_map = {
        str(k): (v.value if isinstance(v, ColumnRole) else str(v)) for k, v in (roles or {}).items()
    }
    if input_columns is not None:
        columns = tuple(str(c) for c in input_columns)
    # Score-time frames usually omit the target; do not require it.
    if target_column is not None:
        columns = tuple(c for c in columns if c != target_column)
    # Prefer feature-role columns when roles are known.
    feature_role_cols = [
        name for name, role in role_map.items() if role == ColumnRole.FEATURE.value
    ]
    if feature_role_cols:
        ordered = [c for c in columns if c in feature_role_cols]
        # Keep any non-feature inputs that still appear (group/time/id) after features.
        ordered.extend(
            c
            for c in columns
            if c not in ordered and role_map.get(c) != ColumnRole.TARGET.value
        )
        if ordered:
            columns = tuple(ordered)

    families = {name: dtype_family(dtype) for name, dtype in dtypes.items()}
    # Ensure every declared column has a family entry.
    for name in columns:
        families.setdefault(name, "other")

    # Encoded/bin outputs are numeric at the features stage even when source
    # columns were string/categorical at input time.
    feature_names = tuple(str(c) for c in feature_columns)
    encoded = tuple(str(c) for c in (encoded_numeric_columns or ()))
    if not encoded:
        encoded = tuple(
            name
            for name in feature_names
            if families.get(name) == "numeric" or name not in columns
        )
    for name in encoded:
        families[name] = "numeric"

    required_roles = tuple(
        sorted(
            {
                role
                for name, role in role_map.items()
                if name in columns and role in _SCORE_REQUIRED_ROLES
            }
        )
    )

    return SchemaContract(
        columns=columns,
        dtypes=dtypes,
        dtype_families=families,  # type: ignore[arg-type]
        roles=role_map,
        feature_columns=feature_names,
        target_column=target_column,
        required_roles=required_roles,
        nullable=nullable,
        encoded_numeric_columns=encoded,
    )


def save_schema_contract(path: str | Path, contract: SchemaContract) -> Path:
    """Write the contract as sorted, indented JSON in the bundle directory.

    Sorted and indented so the file diffs cleanly: comparing two versions of a
    model shows exactly which columns or types changed, which is the review
    question that matters when redeploying.

    Parameters
    ----------
    path:
        The bundle directory, created if missing.
    contract:
        The contract to write.

    Returns
    -------
    Path
        The path of the written file, not the directory — useful for logging the
        exact artifact.

    Raises
    ------
    OSError
        If the directory cannot be created or the file written.

    See Also
    --------
    load_schema_contract : Reading it back.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / SCHEMA_CONTRACT_FILENAME
    destination.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_schema_contract(path: str | Path) -> SchemaContract | None:
    """Read the contract if the bundle has one, and ``None`` if it does not.

    Absence is a normal outcome, not an error. Bundles written before contracts
    existed have no such file, and refusing to load them would break artifacts
    that are otherwise fine. Callers handle the ``None`` by skipping validation
    with a warning.

    Parameters
    ----------
    path:
        The bundle directory, or the contract file itself.

    Returns
    -------
    SchemaContract or None
        The contract, or ``None`` when no file is present.

    Raises
    ------
    ValidationError
        If a file exists but carries an unsupported format. Present-but-unknown
        is different from absent: something wrote a contract this version cannot
        interpret, and ignoring it would validate against nothing while
        appearing to validate.
    json.JSONDecodeError
        If the file is not valid JSON.

    See Also
    --------
    save_schema_contract : Writing it.
    """
    root = Path(path)
    contract_path = root / SCHEMA_CONTRACT_FILENAME if root.is_dir() else root
    if not contract_path.exists():
        return None
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    return SchemaContract.from_dict(payload)


def validate_score_frame(
    frame: pd.DataFrame,
    contract: SchemaContract | None,
    *,
    stage: Literal["input", "features"] = "input",
    allow_extra: bool = True,
    strict_types: bool = True,
    check_roles: bool = True,
) -> SchemaContractValidation:
    """Compare a frame against the contract and report every discrepancy.

    Checks four things and collects all of them before returning: columns that
    are missing, columns that are extra, columns whose type family does not
    match, and roles with no column to fill them. Nothing raises — the caller
    decides what is fatal, usually via :func:`raise_for_contract`.

    Parameters
    ----------
    frame:
        The frame to check.
    contract:
        The contract from the bundle, or ``None`` for a bundle that has none,
        in which case validation is skipped and the result says so.
    stage:
        ``'input'`` to check the raw frame before transforms, ``'features'`` to
        check the transformed frame against what the estimator expects.
    allow_extra:
        Treat unexpected columns as a warning rather than a failure. Defaults to
        permissive, because upstream systems routinely carry fields the model
        does not use.
    strict_types:
        Treat family mismatches as failures. Turning this off records them as
        warnings instead, which is occasionally right when a downstream
        transform will fix the type anyway.
    check_roles:
        Check required roles. Input stage only; roles have no meaning after
        transforms.

    Returns
    -------
    SchemaContractValidation
        Every discrepancy found, and whether the frame passes overall.

    Notes
    -----
    **Type checks only cover columns that are present.** A missing column is
    reported once, as missing, rather than twice.

    **Nullability is advisory.** A column that was non-nullable in training and
    arrives entirely null produces a warning, never a failure — the model will
    still predict, and whether that is acceptable depends on the column.

    **Extra columns at the features stage deserve more suspicion than at
    input.** They usually mean a transform produced something unexpected, most
    often an unseen category creating an extra one-hot column.

    Examples
    --------
    Check before predicting, and act on the specifics::

        result = validate_score_frame(frame, bundle.schema_contract)
        if not result.ok:
            print(result.missing_columns, result.wrong_type_columns)
            raise_for_contract(result)

    See Also
    --------
    coerce_score_frame : Fixing what can be fixed before checking.
    raise_for_contract : Turning a failure into an exception.
    """
    if contract is None:
        return SchemaContractValidation(
            ok=True,
            warnings=[
                "Bundle has no schema_contract.json; skipped score-time schema "
                "contract validation (legacy bundle)."
            ],
            stage=stage,
            contract_present=False,
        )

    expected = (
        list(contract.columns)
        if stage == "input" and contract.columns
        else list(contract.feature_columns)
    )
    if not expected:
        # Fall back to feature columns when input list was empty.
        expected = list(contract.feature_columns)

    present = [str(c) for c in frame.columns]
    missing = [c for c in expected if c not in present]
    extra = [c for c in present if c not in expected]
    wrong: list[dict[str, str]] = []
    warnings: list[str] = []
    missing_roles: list[str] = []

    if allow_extra and extra:
        warnings.append(f"Score frame has extra columns ignored by the contract: {extra}")

    check_cols = [c for c in expected if c in frame.columns]
    encoded_numeric = set(contract.encoded_numeric_columns)
    for column in check_cols:
        expected_family = contract.dtype_families.get(column)
        if expected_family is None:
            continue
        if stage == "features" and column in encoded_numeric:
            expected_family = "numeric"
        actual_family = dtype_family(frame[column].dtype)
        if not families_compatible(expected_family, actual_family):
            wrong.append(
                {
                    "column": column,
                    "expected_family": expected_family,
                    "actual_family": actual_family,
                    "actual_dtype": str(frame[column].dtype),
                }
            )

    if check_roles and stage == "input" and contract.required_roles:
        present_roles = {
            contract.roles[c]
            for c in present
            if c in contract.roles and c != contract.target_column
        }
        missing_roles = [r for r in contract.required_roles if r not in present_roles]
        if missing_roles:
            warnings.append(
                f"Score frame is missing columns for required role(s): {missing_roles}"
            )

    # Nullability is advisory: warn when a non-nullable training column is all-null.
    for column in check_cols:
        if contract.nullable.get(column) is False and frame[column].isna().all():
            warnings.append(
                f"Column '{column}' was non-nullable at fit time but is all-null at score time."
            )

    ok = (
        not missing
        and not (extra and not allow_extra)
        and not (wrong and strict_types)
        and not missing_roles
    )
    if wrong and not strict_types:
        warnings.append(f"Score frame dtype-family mismatches (non-strict): {wrong}")

    return SchemaContractValidation(
        ok=ok,
        missing_columns=missing,
        extra_columns=extra,
        wrong_type_columns=wrong,
        missing_roles=missing_roles,
        warnings=warnings,
        stage=stage,
        contract_present=True,
    )


def coerce_score_frame(
    frame: pd.DataFrame,
    contract: SchemaContract | None,
    *,
    stage: Literal["input", "features"] = "input",
) -> tuple[pd.DataFrame, SchemaContractValidation]:
    """Convert what can safely be converted, then report what is left.

    JSON has no numeric type distinction worth relying on, CSV has no types at
    all, and both are how score frames usually arrive. A column of ``"42"``
    strings that should be integers is not a real mismatch, it is a transport
    artefact, and failing on it makes the contract an obstacle rather than a
    guard.

    Four conversions are attempted, each conservative: numeric from
    numeric-looking strings, boolean from strictly 0/1 numerics, datetime from
    parseable strings, and string from anything. A conversion that fails leaves
    the column untouched and surfaces in the returned validation, so nothing is
    hidden.

    Parameters
    ----------
    frame:
        The frame to coerce. Never modified — a copy is returned.
    contract:
        The contract to coerce toward, or ``None`` to copy and skip.
    stage:
        Which expectation to use, as in :func:`validate_score_frame`.

    Returns
    -------
    tuple of (DataFrame, SchemaContractValidation)
        The coerced copy, and its validation with ``coerced_columns`` listing
        what changed.

    Notes
    -----
    **Missing columns are never invented.** A column that is absent stays
    absent and is reported as missing. Filling it would mean guessing values,
    and a guessed value produces a prediction that looks real.

    **Boolean coercion requires strictly 0 and 1.** A column of arbitrary
    integers is not silently truncated to booleans.

    **Numeric coercion uses ``errors='coerce'``, which turns unparseable values
    into nulls.** A column of mostly-numeric strings with a few ``'N/A'``
    entries converts, and those entries become missing. That is usually what was
    meant, and it is worth knowing it happened — check ``coerced_columns``
    against your expectations rather than assuming a clean conversion.

    **Coercion is not free on large frames.** It touches every expected column;
    skip it when the source already produces correct types.

    Examples
    --------
    The usual score-time entry point::

        frame, result = coerce_score_frame(raw_frame, bundle.schema_contract)
        raise_for_contract(result)

    See Also
    --------
    validate_score_frame : Checking without converting.
    """
    if contract is None:
        return frame.copy(), validate_score_frame(frame, None, stage=stage)

    out = frame.copy()
    coerced: list[str] = []
    expected = (
        list(contract.columns)
        if stage == "input" and contract.columns
        else list(contract.feature_columns)
    )
    if not expected:
        expected = list(contract.feature_columns)
    encoded_numeric = set(contract.encoded_numeric_columns)

    for column in expected:
        if column not in out.columns:
            continue
        family = contract.dtype_families.get(column, "other")
        if stage == "features" and column in encoded_numeric:
            family = "numeric"
        series = out[column]
        try:
            if family == "numeric":
                converted = pd.to_numeric(series, errors="coerce")
                # Only accept when at least one non-null value converted or all null.
                if converted.notna().any() or series.isna().all():
                    out[column] = converted
                    coerced.append(column)
            elif family == "bool":
                if pd.api.types.is_bool_dtype(series):
                    continue
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.dropna().isin([0, 1]).all():
                    out[column] = numeric.astype("boolean")
                    coerced.append(column)
            elif family == "datetime":
                converted = pd.to_datetime(series, errors="coerce")
                if converted.notna().any() or series.isna().all():
                    out[column] = converted
                    coerced.append(column)
            elif family in {"string", "categorical"}:
                out[column] = series.astype("string")
                coerced.append(column)
        except Exception:  # noqa: BLE001 - leave column unchanged
            continue

    result = validate_score_frame(out, contract, stage=stage, strict_types=True)
    result.coerced_columns = coerced
    if coerced:
        result.warnings.append(f"Coerced columns toward contract families: {coerced}")
    return out, result


def raise_for_contract(result: SchemaContractValidation, *, allow_extra: bool = True) -> None:
    """Turn a failed validation into an exception naming every problem.

    The separation between checking and raising is deliberate: some callers want
    to inspect and recover, others want to stop. This is the stopping half, and
    it builds a message that lists missing columns, missing roles, and each
    wrong-typed column with what was expected and what arrived — enough to fix
    the caller without a debugging session.

    Passing validations return silently, so this can be called unconditionally.

    Parameters
    ----------
    result:
        The validation to act on.
    allow_extra:
        Whether extra columns are acceptable. Should match what was passed to
        the validation; when they were already treated as a warning, they are
        left out of the message.

    Raises
    ------
    ValidationError
        If the validation failed. The message names the stage, so it is clear
        whether the caller's frame was wrong or a transform misbehaved.

    See Also
    --------
    validate_score_frame : Producing the result.
    """
    if result.ok:
        return
    parts: list[str] = []
    if result.missing_columns:
        parts.append(f"missing columns: {result.missing_columns}")
    if result.extra_columns and not allow_extra:
        parts.append(f"extra columns: {result.extra_columns}")
    if result.missing_roles:
        parts.append(f"missing roles: {result.missing_roles}")
    if result.wrong_type_columns:
        details = [
            (
                f"{item['column']} (expected {item['expected_family']}, "
                f"got {item['actual_family']} / {item['actual_dtype']})"
            )
            for item in result.wrong_type_columns
        ]
        parts.append("wrong-type columns: " + "; ".join(details))
    raise ValidationError(
        "Score frame failed schema contract validation "
        f"({result.stage}): " + "; ".join(parts)
    )
