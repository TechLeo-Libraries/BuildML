"""Pandas engine adapter (core default)."""

from __future__ import annotations

import pandas as pd

from buildml.core.types import EngineName
from buildml.data.engines.aggregate import (
    aggregate_pandas,
    normalize_aggregations,
    validate_aggregate_columns,
)


class PandasEngine:
    """Identity adapter around Pandas DataFrames."""

    name = EngineName.PANDAS

    def from_pandas(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()

    def to_pandas(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()

    def n_rows(self, table: pd.DataFrame) -> int:
        return int(len(table))

    def columns(self, table: pd.DataFrame) -> list[str]:
        return list(table.columns.astype(str))

    def head(self, table: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return table.head(n).copy()

    def select_columns(self, table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return table.loc[:, list(columns)].copy()

    def sample_rows(
        self,
        table: pd.DataFrame,
        n: int,
        *,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        take = min(int(n), int(len(table)))
        return table.sample(n=take, random_state=random_state).copy()

    def filter_rows(
        self,
        table: pd.DataFrame,
        mask: list[bool] | tuple[bool, ...],
    ) -> pd.DataFrame:
        if len(mask) != len(table):
            raise ValueError(
                f"filter mask length {len(mask)} does not match table rows {len(table)}"
            )
        return table.loc[list(mask)].copy()

    def aggregate(
        self,
        table: pd.DataFrame,
        aggregations: dict[str, str | list[str]],
        *,
        by: list[str] | None = None,
    ) -> pd.DataFrame:
        pairs = normalize_aggregations(aggregations)
        validate_aggregate_columns(self.columns(table), by, pairs)
        return aggregate_pandas(table, pairs, by=by)
