"""Structured result objects returned by BuildML APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.core.types import DataMode, EngineName, TableSchema


@dataclass(slots=True)
class IngestReport:
    """Summary of automated ingest detection and recommendations.

    Parameters
    ----------
    source_type:
        Kind of source provided (``dataframe``, ``csv``, ``parquet``, ``arrow``, …).
    format_name:
        Detected or declared file/format name.
    schema:
        Inferred table schema.
    row_estimate:
        Estimated or exact row count when available.
    byte_estimate:
        Estimated on-disk/in-memory bytes when available.
    recommended_mode:
        Suggested :class:`~buildml.core.types.DataMode`.
    recommended_engine:
        Suggested :class:`~buildml.core.types.EngineName`.
    available_engines:
        Engines importable in the current environment.
    warnings:
        Non-fatal notices for the user (scale, missing extras, etc.).
    details:
        Extra structured metadata for debugging.
    """

    source_type: str
    format_name: str
    schema: TableSchema
    row_estimate: int | None
    byte_estimate: int | None
    recommended_mode: DataMode
    recommended_engine: EngineName
    available_engines: tuple[EngineName, ...] = ()
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "format_name": self.format_name,
            "schema": self.schema.to_dict(),
            "row_estimate": self.row_estimate,
            "byte_estimate": self.byte_estimate,
            "recommended_mode": self.recommended_mode.value,
            "recommended_engine": self.recommended_engine.value,
            "available_engines": [e.value for e in self.available_engines],
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }
