"""Result objects shared across packages, rather than owned by one.

Most result types live with the operation that produces them. The ones here are
the exceptions — structures several packages need, which would otherwise create
an import cycle if they lived in any one of them.

See Also
--------
buildml.ingest : Where :class:`IngestReport` is produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.core.types import DataMode, EngineName, TableSchema


@dataclass(slots=True)
class IngestReport:
    """What ingest found in the data, and what it suggests you do about it.

    Produced before any modelling, from inspecting the source. The
    recommendations are the useful part: a two-gigabyte Parquet file and a
    thousand-row CSV want different engines and different modes, and the choice
    is easier to make from measured size and available extras than from
    guesswork.

    Recommendations are exactly that. Nothing is applied until you act on it.

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

    Notes
    -----
    **``recommended_engine`` only ever names an engine that is installed.** It
    is chosen from ``available_engines``, so a large file on a base install is
    recommended Pandas with a warning about the size, rather than an engine you
    cannot use. The warning is where the better option gets mentioned.

    **``row_estimate`` may be an estimate.** For CSV it is inferred rather than
    counted, since counting means reading the file — which is the cost ingest is
    trying to help you avoid.

    See Also
    --------
    buildml.core.types.EngineName : What the recommended engines offer.
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
        """Convert the report to JSON-safe plain data.

        Enums become their string values and the schema is expanded, so the
        result can go straight into a log, a run record, or checkpoint metadata.

        Returns
        -------
        dict
            Every field, with enums as strings and collections copied so later
            mutation of the report does not alter what was recorded.
        """
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
