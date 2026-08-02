"""Format, schema, and scale detection helpers."""

from __future__ import annotations

import importlib.util
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import DataMode, EngineName, SchemaField, TableSchema

# Heuristic thresholds (bytes). Documented; always overridable by the user.
MEMORY_SOFT_LIMIT = 250 * 1024 * 1024  # 250 MiB
LAZY_SOFT_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GiB
# Opt-in hard refuse default when BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES is set,
# or when callers pass hard_limit_bytes explicitly. None means no hard refuse.
MEMORY_HARD_LIMIT: int | None = None
_ENV_HARD = os.environ.get("BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES")
if _ENV_HARD is not None and str(_ENV_HARD).strip():
    try:
        MEMORY_HARD_LIMIT = int(_ENV_HARD)
    except ValueError:
        MEMORY_HARD_LIMIT = None


@dataclass(slots=True)
class MaterializationTelemetry:
    """Peak-footprint estimate and gate outcome for a materialization boundary."""

    context: str
    nbytes: int
    soft_limit_bytes: int
    hard_limit_bytes: int | None
    soft_exceeded: bool
    hard_exceeded: bool
    warnings: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "nbytes": self.nbytes,
            "nbytes_mib": round(self.nbytes / (1024 * 1024), 3),
            "soft_limit_bytes": self.soft_limit_bytes,
            "hard_limit_bytes": self.hard_limit_bytes,
            "soft_exceeded": self.soft_exceeded,
            "hard_exceeded": self.hard_exceeded,
            "warnings": list(self.warnings),
            "guidance": list(self.guidance),
        }


def schema_from_dataframe(frame: pd.DataFrame) -> TableSchema:
    """Infer a :class:`TableSchema` from a Pandas DataFrame."""
    fields = tuple(
        SchemaField(
            name=str(col),
            dtype=str(frame[col].dtype),
            nullable=bool(frame[col].isna().any()),
        )
        for col in frame.columns
    )
    return TableSchema(fields=fields)


def detect_path_format(path: Path) -> str:
    """Detect a tabular file format from suffix."""
    suffix = path.suffix.lower()
    mapping = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".feather": "arrow",
        ".arrow": "arrow",
        ".ipc": "arrow",
    }
    return mapping.get(suffix, "unknown")


def estimate_path_bytes(path: Path) -> int | None:
    """Return file size in bytes when available."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def estimate_dataframe_bytes(frame: pd.DataFrame) -> int:
    """Rough in-memory footprint estimate for a DataFrame."""
    return int(frame.memory_usage(deep=True).sum())


def check_materialization(
    frame: pd.DataFrame,
    *,
    context: str,
    soft_limit_bytes: int = MEMORY_SOFT_LIMIT,
    hard_limit_bytes: int | None = MEMORY_HARD_LIMIT,
    on_soft: Literal["warn", "ignore"] = "warn",
    on_hard: Literal["error", "warn"] = "error",
) -> MaterializationTelemetry:
    """Evaluate soft/hard materialization gates and return telemetry.

    Parameters
    ----------
    frame:
        Frame about to be copied or consumed as an sklearn design matrix.
    context:
        Short label for warnings and history (for example ``estimator fit``).
    soft_limit_bytes:
        Soft disclosure threshold (default 250 MiB). Exceeding emits a warning
        unless ``on_soft='ignore'``.
    hard_limit_bytes:
        Optional hard refuse threshold. Defaults to
        ``BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES`` when set, else ``None``
        (no hard refuse). Pass an explicit int to refuse oversized copies.
    on_soft / on_hard:
        Soft: ``warn`` or ``ignore``. Hard: ``error`` raises
        :class:`~buildml.core.errors.ValidationError`, or ``warn`` only.

    Notes
    -----
    BuildML never silently materializes huge frames: soft exceedance always
    returns guidance in the telemetry object, and hard exceedance refuses by
    default when a hard limit is configured. Prefer lazy prep and
    materialize only the train design matrix at the estimator boundary.
    """
    nbytes = estimate_dataframe_bytes(frame)
    soft_exceeded = nbytes >= soft_limit_bytes
    hard_exceeded = hard_limit_bytes is not None and nbytes >= hard_limit_bytes
    notes: list[str] = []
    guidance = [
        "Prefer engine='polars' or 'duckdb' with mode='lazy' for prep.",
        "Materialize only the train design matrix needed for fit/predict, not the full prep frame.",
        "Set hard_limit_bytes or BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES "
        "to refuse oversized copies.",
    ]
    mb = nbytes / (1024 * 1024)
    if soft_exceeded:
        soft_mb = soft_limit_bytes / (1024 * 1024)
        message = (
            f"Materializing ~{mb:.1f} MiB for {context} exceeds the "
            f"{soft_mb:.0f} MiB soft limit. Prefer lazy prep, then "
            "materialize only the train design matrix. Soft gates disclose risk; "
            "they do not refuse unless a hard limit is also configured."
        )
        notes.append(message)
        if on_soft == "warn":
            warnings.warn(message, UserWarning, stacklevel=3)
    if hard_exceeded:
        assert hard_limit_bytes is not None
        hard_mb = hard_limit_bytes / (1024 * 1024)
        hard_message = (
            f"Materializing ~{mb:.1f} MiB for {context} exceeds the "
            f"{hard_mb:.0f} MiB hard limit. Narrow columns/rows, use a lazy "
            "engine for prep, or raise hard_limit_bytes explicitly if the copy "
            "is intentional."
        )
        notes.append(hard_message)
        if on_hard == "error":
            raise ValidationError(hard_message)
        warnings.warn(hard_message, UserWarning, stacklevel=3)
    return MaterializationTelemetry(
        context=context,
        nbytes=nbytes,
        soft_limit_bytes=soft_limit_bytes,
        hard_limit_bytes=hard_limit_bytes,
        soft_exceeded=soft_exceeded,
        hard_exceeded=hard_exceeded,
        warnings=notes,
        guidance=guidance if (soft_exceeded or hard_exceeded) else [],
    )


def warn_if_large_materialization(
    frame: pd.DataFrame,
    *,
    context: str,
    soft_limit_bytes: int = MEMORY_SOFT_LIMIT,
    hard_limit_bytes: int | None = MEMORY_HARD_LIMIT,
) -> list[str]:
    """Emit soft (and optional hard) materialization gate messages.

    Returns warning strings for callers that record structured notes. Soft
    gates disclose scale risk; hard gates refuse when configured.
    """
    telemetry = check_materialization(
        frame,
        context=context,
        soft_limit_bytes=soft_limit_bytes,
        hard_limit_bytes=hard_limit_bytes,
    )
    return list(telemetry.warnings)


def available_engines() -> tuple[EngineName, ...]:
    """Return engines whose packages are importable."""
    found: list[EngineName] = [EngineName.PANDAS]
    if importlib.util.find_spec("polars") is not None:
        found.append(EngineName.POLARS)
    if importlib.util.find_spec("duckdb") is not None:
        found.append(EngineName.DUCKDB)
    return tuple(found)


def recommend_mode(
    *,
    byte_estimate: int | None,
    row_estimate: int | None,
) -> DataMode:
    """Recommend a data mode from scale estimates."""
    if byte_estimate is None:
        if row_estimate is not None and row_estimate >= 5_000_000:
            return DataMode.LAZY
        return DataMode.MEMORY
    if byte_estimate >= MEMORY_SOFT_LIMIT:
        return DataMode.LAZY
    return DataMode.MEMORY


def recommend_engine(
    *,
    mode: DataMode,
    installed: tuple[EngineName, ...],
) -> tuple[EngineName, list[str]]:
    """Recommend an engine and collect related warnings."""
    warnings: list[str] = []
    if mode == DataMode.MEMORY:
        return EngineName.PANDAS, warnings

    if EngineName.POLARS in installed:
        return EngineName.POLARS, warnings
    if EngineName.DUCKDB in installed:
        return EngineName.DUCKDB, warnings

    warnings.append(
        "Large/lazy mode recommended, but neither Polars nor DuckDB is installed. "
        "Install with: pip install 'buildml[engines]'. Continuing with Pandas when safe."
    )
    return EngineName.PANDAS, warnings


def build_scale_details(
    *,
    byte_estimate: int | None,
    row_estimate: int | None,
    mocked_bytes: int | None = None,
) -> dict[str, Any]:
    """Extra scale metadata for ingest reports."""
    return {
        "memory_soft_limit_bytes": MEMORY_SOFT_LIMIT,
        "lazy_soft_limit_bytes": LAZY_SOFT_LIMIT,
        "memory_hard_limit_bytes": MEMORY_HARD_LIMIT,
        "byte_estimate": byte_estimate,
        "row_estimate": row_estimate,
        "mocked_bytes": mocked_bytes,
        "materialization_guidance": (
            "Soft gates warn near 250 MiB; configure "
            "BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES or hard_limit_bytes to refuse. "
            "Keep prep lazy when possible and materialize only the train design matrix."
        ),
    }
