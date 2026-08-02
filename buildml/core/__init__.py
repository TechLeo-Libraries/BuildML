"""Core types, errors, and result objects for BuildML 2.x."""

from buildml.core.errors import (
    BuildMLError,
    IngestError,
    LeakageError,
    MissingExtraError,
    ValidationError,
)
from buildml.core.results import IngestReport
from buildml.core.types import (
    ColumnRole,
    DataMode,
    EngineName,
    SchemaField,
    TableSchema,
    coerce_data_mode,
)

__all__ = [
    "BuildMLError",
    "ColumnRole",
    "DataMode",
    "EngineName",
    "IngestError",
    "IngestReport",
    "LeakageError",
    "MissingExtraError",
    "SchemaField",
    "TableSchema",
    "ValidationError",
    "coerce_data_mode",
]
