"""Score-time schema contract for pipeline bundles."""

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
    """Input / feature contract persisted with a pipeline bundle.

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
    """Outcome of validating a score frame against a contract."""

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
    """Map a pandas/numpy dtype (or string) to a coarse family."""
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
    """Return True when score-time families are interchangeable."""
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
    """Collect raw score-time input columns referenced by fitted plans.

    Engineered outputs (for example one-hot feature names) are excluded so the
    contract describes the frame expected *before* plan replay.
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
    """Build a score-time contract from training schema/roles/feature columns."""
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
    """Write ``schema_contract.json`` under a pipeline bundle directory."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / SCHEMA_CONTRACT_FILENAME
    destination.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_schema_contract(path: str | Path) -> SchemaContract | None:
    """Load a schema contract when present; return None for older bundles."""
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
    """Validate a score frame against a persisted contract.

    Parameters
    ----------
    frame:
        Incoming score DataFrame.
    contract:
        Contract from the bundle, or ``None`` for legacy bundles (skip).
    stage:
        ``input`` checks raw columns before plan replay; ``features`` checks
        the estimator feature contract after transforms.
    allow_extra:
        When True, extra columns produce warnings rather than errors.
    strict_types:
        When True, dtype-family mismatches are errors.
    check_roles:
        When True (input stage), require that each ``required_roles`` entry has
        at least one present column with that role.
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
    """Return a copy with best-effort dtype coercion toward the contract.

    Coercion is conservative:

    - numeric ← bool / numeric-looking strings
    - bool ← 0/1 numeric
    - datetime ← parseable strings
    - string/categorical ← cast to string

    Failures leave the column unchanged and appear as wrong-type entries when
    revalidated. Does not invent missing columns.
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
    """Raise :class:`ValidationError` when contract validation failed."""
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
