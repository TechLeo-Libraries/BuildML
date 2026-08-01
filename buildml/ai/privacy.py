"""Egress privacy controls for AI operator."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buildml.ai.types import EgressLevel


@dataclass(slots=True)
class EgressConfig:
    """Configuration for what data may leave the machine to an LLM provider."""

    level: EgressLevel = EgressLevel.STATS_ONLY
    allow_columns: tuple[str, ...] | None = None
    deny_columns: tuple[str, ...] = ()
    rename_columns: dict[str, str] = field(default_factory=dict)
    strip_headers: bool = False
    sample_rows: int = 5
    redact_patterns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "allow_columns": list(self.allow_columns) if self.allow_columns else None,
            "deny_columns": list(self.deny_columns),
            "rename_columns": dict(self.rename_columns),
            "strip_headers": self.strip_headers,
            "sample_rows": self.sample_rows,
            "redact_patterns": list(self.redact_patterns),
        }


@dataclass(frozen=True, slots=True)
class EgressManifest:
    """What will (or did) leave the machine for a single LLM call."""

    level: EgressLevel
    columns_sent: tuple[str, ...]
    columns_denied: tuple[str, ...]
    columns_renamed: dict[str, str]
    rows_sent: int
    estimated_tokens: int | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "columns_sent": list(self.columns_sent),
            "columns_denied": list(self.columns_denied),
            "columns_renamed": dict(self.columns_renamed),
            "rows_sent": self.rows_sent,
            "estimated_tokens": self.estimated_tokens,
            "warnings": list(self.warnings),
        }


_PII_PATTERNS = (
    re.compile(r"(?i)(email|e[-_]?mail)", re.IGNORECASE),
    re.compile(r"(?i)(phone|mobile|cell)", re.IGNORECASE),
    re.compile(r"(?i)(ssn|social.*security)", re.IGNORECASE),
    re.compile(r"(?i)(address|street|zip|postal)", re.IGNORECASE),
    re.compile(r"(?i)(credit.*card|card.*number)", re.IGNORECASE),
    re.compile(r"(?i)(password|passwd|pwd)", re.IGNORECASE),
    re.compile(r"(?i)(birth.*date|dob|date.*birth)", re.IGNORECASE),
    re.compile(r"(?i)(first.*name|last.*name|full.*name)", re.IGNORECASE),
)


def detect_pii_columns(columns: list[str]) -> list[str]:
    """Heuristic detection of columns that may contain PII.

    Returns column names that match common PII patterns. This is a non-blocking
    warning helper, not a guarantee.
    """
    suspicious = []
    for col in columns:
        for pattern in _PII_PATTERNS:
            if pattern.search(col):
                suspicious.append(col)
                break
    return suspicious


def filter_columns(
    columns: list[str],
    *,
    allow: tuple[str, ...] | None = None,
    deny: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Apply allow/deny lists to column names.

    Returns (allowed_columns, denied_columns).
    """
    if allow is not None:
        allowed = [c for c in columns if c in allow]
        denied = [c for c in columns if c not in allow]
    else:
        allowed = [c for c in columns if c not in deny]
        denied = [c for c in columns if c in deny]
    return allowed, denied


def rename_columns(
    columns: list[str],
    mapping: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Apply renaming to columns.

    Returns (renamed_columns, applied_renames).
    """
    renamed = []
    applied: dict[str, str] = {}
    for col in columns:
        if col in mapping:
            renamed.append(mapping[col])
            applied[col] = mapping[col]
        else:
            renamed.append(col)
    return renamed, applied


def scrub_headers(columns: list[str]) -> list[str]:
    """Replace column names with generic placeholders."""
    return [f"col_{i}" for i in range(len(columns))]


def redact_value(value: Any, patterns: tuple[str, ...] = ()) -> Any:
    """Redact sensitive patterns from a string value."""
    if not isinstance(value, str):
        return value
    result = value
    for pattern in patterns:
        result = re.sub(pattern, "[REDACTED]", result)
    return result


def build_schema_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Build SCHEMA_ONLY payload and manifest."""
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    pii_warnings = detect_pii_columns(sent_columns)
    warnings = tuple(
        f"Column '{c}' matches PII pattern; consider deny_columns or rename_columns."
        for c in pii_warnings
    )

    schema_info = {
        "columns": sent_columns,
        "dtypes": {
            sent_columns[i]: str(df[allowed[i]].dtype)
            for i in range(len(allowed))
        },
        "row_count": len(df),
    }

    manifest = EgressManifest(
        level=EgressLevel.SCHEMA_ONLY,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=0,
        estimated_tokens=_estimate_tokens(str(schema_info)),
        warnings=warnings,
    )

    return schema_info, manifest


def build_stats_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Build STATS_ONLY payload and manifest."""
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    pii_warnings = detect_pii_columns(sent_columns)
    warnings = tuple(
        f"Column '{c}' matches PII pattern; consider deny_columns or rename_columns."
        for c in pii_warnings
    )

    stats: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(sent_columns),
        "columns": {},
    }

    for i, orig_col in enumerate(allowed):
        col_name = sent_columns[i]
        series = df[orig_col]
        col_stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_stats["mean"] = float(desc.get("mean", 0))
            col_stats["std"] = float(desc.get("std", 0))
            col_stats["min"] = float(desc.get("min", 0))
            col_stats["max"] = float(desc.get("max", 0))
        stats["columns"][col_name] = col_stats

    manifest = EgressManifest(
        level=EgressLevel.STATS_ONLY,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=0,
        estimated_tokens=_estimate_tokens(str(stats)),
        warnings=warnings,
    )

    return stats, manifest


def build_redacted_sample_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Build REDACTED_SAMPLE payload with PII columns masked."""
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    pii_columns = detect_pii_columns(allowed)
    sample_size = min(config.sample_rows, len(df))
    sample_df = df[allowed].head(sample_size).copy()

    for pii_col in pii_columns:
        if pii_col in sample_df.columns:
            sample_df[pii_col] = sample_df[pii_col].apply(_hash_value)

    for pat in config.redact_patterns:
        for col in sample_df.columns:
            if sample_df[col].dtype == object:
                sample_df[col] = sample_df[col].apply(
                    lambda x, p=pat: redact_value(x, (p,))
                )

    sample_df.columns = pd.Index(sent_columns)
    sample_records = sample_df.to_dict(orient="records")

    pii_warnings = tuple(
        f"Column '{c}' masked (PII pattern detected)." for c in pii_columns
    )

    payload = {
        "row_count": len(df),
        "sample_rows": sample_size,
        "columns": sent_columns,
        "sample": sample_records,
    }

    manifest = EgressManifest(
        level=EgressLevel.REDACTED_SAMPLE,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=sample_size,
        estimated_tokens=_estimate_tokens(str(payload)),
        warnings=pii_warnings,
    )

    return payload, manifest


def build_full_sample_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Build FULL_SAMPLE payload (explicit opt-in only)."""
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    sample_size = min(config.sample_rows, len(df))
    sample_df = df[allowed].head(sample_size).copy()
    sample_df.columns = pd.Index(sent_columns)
    sample_records = sample_df.to_dict(orient="records")

    pii_columns = detect_pii_columns(sent_columns)
    warnings = tuple(
        f"Column '{c}' matches PII pattern; FULL_SAMPLE sends raw values."
        for c in pii_columns
    )

    payload = {
        "row_count": len(df),
        "sample_rows": sample_size,
        "columns": sent_columns,
        "sample": sample_records,
    }

    manifest = EgressManifest(
        level=EgressLevel.FULL_SAMPLE,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=sample_size,
        estimated_tokens=_estimate_tokens(str(payload)),
        warnings=warnings,
    )

    return payload, manifest


def build_egress_payload(
    df: pd.DataFrame | None,
    config: EgressConfig,
) -> tuple[dict[str, Any] | None, EgressManifest]:
    """Build egress payload and manifest for the configured level.

    Returns (payload_dict, manifest). If df is None, returns empty manifest.
    """
    if df is None or len(df) == 0:
        return None, EgressManifest(
            level=config.level,
            columns_sent=(),
            columns_denied=(),
            columns_renamed={},
            rows_sent=0,
            estimated_tokens=0,
            warnings=("No dataset attached; schema-only context unavailable.",),
        )

    if config.level == EgressLevel.SCHEMA_ONLY:
        return build_schema_payload(df, config)
    elif config.level == EgressLevel.STATS_ONLY:
        return build_stats_payload(df, config)
    elif config.level == EgressLevel.REDACTED_SAMPLE:
        return build_redacted_sample_payload(df, config)
    elif config.level == EgressLevel.FULL_SAMPLE:
        return build_full_sample_payload(df, config)
    else:
        return build_stats_payload(df, config)


def _hash_value(value: Any) -> str:
    """Hash a value for redaction."""
    if pd.isna(value):
        return "[NULL]"
    return f"[HASH:{hashlib.sha256(str(value).encode()).hexdigest()[:8]}]"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words / 0.75)."""
    words = len(text.split())
    return int(words / 0.75)
