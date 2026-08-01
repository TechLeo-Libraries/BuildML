"""Engine protocol shared by Pandas/Polars/DuckDB adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from buildml.core.types import EngineName


@runtime_checkable
class Engine(Protocol):
    """Minimal tabular engine contract for BuildML dataset operations."""

    name: EngineName

    def from_pandas(self, frame: pd.DataFrame) -> Any:
        """Convert a Pandas DataFrame into the engine's native table type."""
        ...

    def to_pandas(self, table: Any) -> pd.DataFrame:
        """Materialize an engine table as a Pandas DataFrame copy."""
        ...

    def n_rows(self, table: Any) -> int:
        """Return row count."""
        ...

    def columns(self, table: Any) -> list[str]:
        """Return column names."""
        ...

    def head(self, table: Any, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows as Pandas."""
        ...

    def select_columns(self, table: Any, columns: list[str]) -> Any:
        """Project ``table`` to the named columns without full-frame widen."""
        ...

    def sample_rows(
        self,
        table: Any,
        n: int,
        *,
        random_state: int | None = None,
    ) -> Any:
        """Return up to ``n`` rows (engine-native), optionally seeded."""
        ...

    def filter_rows(self, table: Any, mask: list[bool] | tuple[bool, ...]) -> Any:
        """Keep rows where ``mask`` is True (length must match ``n_rows``)."""
        ...

    def aggregate(
        self,
        table: Any,
        aggregations: dict[str, str | list[str]],
        *,
        by: list[str] | None = None,
    ) -> Any:
        """Return grouped (or global) aggregations as an engine-native table.

        ``aggregations`` maps column name → function name or list of functions.
        Supported functions: ``sum``, ``mean``, ``min``, ``max``, ``count``,
        ``n_unique``, ``std``, ``median``, and integer percentiles ``q0``..``q100``.
        Use ``{"*": "count"}`` for a row count.
        """
        ...
