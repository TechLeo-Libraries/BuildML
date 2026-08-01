"""Shared enums and schema structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ColumnRole(str, Enum):
    """Semantic role of a column in a modeling workflow."""

    FEATURE = "feature"
    TARGET = "target"
    GROUP = "group"
    TIME = "time"
    ID = "id"
    WEIGHT = "weight"
    IGNORE = "ignore"


class DataMode(str, Enum):
    """How a dataset is held and processed."""

    MEMORY = "memory"
    LAZY = "lazy"
    OUT_OF_CORE = "out_of_core"


class EngineName(str, Enum):
    """Supported compute engines."""

    PANDAS = "pandas"
    POLARS = "polars"
    DUCKDB = "duckdb"


@dataclass(frozen=True, slots=True)
class SchemaField:
    """One column in a table schema."""

    name: str
    dtype: str
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Ordered schema for a tabular dataset."""

    fields: tuple[SchemaField, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self.fields]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [
                {"name": f.name, "dtype": f.dtype, "nullable": f.nullable} for f in self.fields
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TableSchema:
        fields = tuple(
            SchemaField(
                name=str(item["name"]),
                dtype=str(item["dtype"]),
                nullable=bool(item.get("nullable", True)),
            )
            for item in payload.get("fields", [])
        )
        return cls(fields=fields)
