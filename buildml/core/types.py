"""The vocabulary every other package agrees on.

Three ideas recur throughout BuildML and are defined here once so they mean the
same thing everywhere: what a column is *for* (:class:`ColumnRole`), how the data
is being held (:class:`DataMode`), which library is doing the work
(:class:`EngineName`), and what the table looks like (:class:`TableSchema`).

Roles are the load-bearing one. Most tools ask you to hand a model an ``X`` and
a ``y`` at the moment of fitting. BuildML asks you to say what each column means
once, and then every operation downstream knows which column is the target and
must never become a feature, which identifies a row and must never be modelled,
which groups rows that have to stay together across a split, and which orders
them in time. That single declaration is what lets leakage be caught by the
library rather than by the reader.

See Also
--------
buildml.core.validation : Checking roles and column names against a schema.
buildml.data.dataset : Where roles are attached to real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ColumnRole(str, Enum):
    """What a column is for, declared once and enforced everywhere after.

    Assigning roles is how a frame of columns becomes a modelling problem. Each
    role changes what the library will and will not do with the column.

    Attributes
    ----------
    FEATURE:
        An input the model may learn from.
    TARGET:
        What is being predicted. Never becomes a feature, and its statistics may
        only be computed from training rows.
    GROUP:
        Rows that belong together: the same patient, customer, or session.
        Group-aware splitters keep them in one partition, because a model that
        saw one visit from a patient in training has effectively seen that
        patient at test time.
    TIME:
        The ordering column. Time-series splits use it to keep the future out of
        the past, which random splitting cannot do.
    ID:
        A row identifier. Kept for tracing predictions back to records, excluded
        from features, because an identifier that correlates with the target
        gives a model a shortcut that will not exist in production.
    WEIGHT:
        Per-row importance, passed through to estimators that accept
        ``sample_weight``.
    IGNORE:
        Present in the data, excluded from everything.

    Notes
    -----
    **``ID`` and ``IGNORE`` both exclude a column, for different reasons.** An
    identifier is kept and carried alongside predictions; an ignored column is
    simply set aside.

    **The role is a promise the library keeps on your behalf.** Marking a column
    ``GROUP`` does nothing on its own: it takes effect the moment a
    group-aware splitter or cross-validator runs.

    See Also
    --------
    buildml.core.validation.validate_role_name : Accepting a role by name.
    """

    FEATURE = "feature"
    TARGET = "target"
    GROUP = "group"
    TIME = "time"
    ID = "id"
    WEIGHT = "weight"
    IGNORE = "ignore"


class DataMode(str, Enum):
    """How a dataset is held and processed.

    ``memory`` materializes tables eagerly. ``lazy`` keeps a native lazy/scan
    handle when an engine supports it (Polars LazyFrame). There is no separate
    out-of-core *fitting* mode: sklearn still requires an in-memory design
    matrix. Legacy string ``out_of_core`` is accepted as an alias of
    ``lazy`` via :func:`coerce_data_mode`.
    """

    MEMORY = "memory"
    LAZY = "lazy"


def coerce_data_mode(mode: DataMode | str) -> DataMode:
    """Accept a data mode as an enum or a string, including the legacy spelling.

    Checkpoints written by older versions recorded ``out_of_core`` as a mode.
    That distinction no longer exists: it was always the same thing as lazy :
    so the old name is mapped forward rather than rejected, and old checkpoints
    keep loading.

    Parameters
    ----------
    mode:
        A :class:`DataMode`, or its string value. Case and surrounding
        whitespace are ignored.

    Returns
    -------
    DataMode
        The normalised mode.

    Raises
    ------
    ValueError
        If the string matches no known mode. Raised by the enum itself rather
        than wrapped, since this is a programming error rather than user input.

    Examples
    --------
    >>> coerce_data_mode("lazy")
    <DataMode.LAZY: 'lazy'>
    >>> coerce_data_mode("out_of_core")
    <DataMode.LAZY: 'lazy'>
    >>> coerce_data_mode(DataMode.MEMORY)
    <DataMode.MEMORY: 'memory'>

    See Also
    --------
    DataMode : What the modes mean.
    """
    if isinstance(mode, DataMode):
        return mode
    value = str(mode).strip().lower()
    if value == "out_of_core":
        return DataMode.LAZY
    return DataMode(value)


class EngineName(str, Enum):
    """Which library performs the table operations underneath.

    The engine changes how the same operation is executed, not what it means. A
    filter is a filter in all three; only the speed, the memory profile, and the
    availability of laziness differ.

    Attributes
    ----------
    PANDAS:
        Always available, eager, and the widest in library support. The default,
        and the right answer until the data stops fitting comfortably in memory.
    POLARS:
        Faster and considerably leaner, with real lazy execution so a chain of
        operations is optimised as one plan instead of materialising each step.
        Requires the ``polars`` extra.
    DUCKDB:
        Executes SQL against files without loading them, which makes it the
        option for data larger than memory. Requires the ``duckdb`` extra.

    Notes
    -----
    **Choosing a non-Pandas engine does not remove the Pandas boundary.**
    scikit-learn needs an in-memory design matrix, so whatever the engine does
    upstream, the frame is materialised before an estimator sees it.

    See Also
    --------
    DataMode : Eager versus lazy holding, which interacts with the engine.
    """

    PANDAS = "pandas"
    POLARS = "polars"
    DUCKDB = "duckdb"


@dataclass(frozen=True, slots=True)
class SchemaField:
    """One column's name, type, and whether it may be null.

    Attributes
    ----------
    name:
        The column name as it appears in the data.
    dtype:
        The type as a string, in the engine's own spelling, so it round-trips
        through JSON without a translation layer that could lose information.
    nullable:
        Whether nulls are permitted. Defaults to ``True``, which is the safe
        assumption for ingested data.

    See Also
    --------
    TableSchema : The ordered collection of these.
    """

    name: str
    dtype: str
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class TableSchema:
    """The columns of a table, in order, with their types.

    Order is part of the schema, not incidental. Checkpoint reattach compares a
    saved schema against the current data to decide whether the saved roles and
    splits still apply, and column order is one of the things that has to hold.

    Attributes
    ----------
    fields:
        The columns in order. A tuple, and the dataclass is frozen, so a schema
        captured at one moment cannot be mutated by later work.

    Notes
    -----
    **Dtypes are strings rather than parsed types.** A schema has to survive
    JSON and three engines with different type systems, and comparing the
    engine's own spelling is more honest than mapping everything onto a lowest
    common denominator.

    See Also
    --------
    buildml.checkpoint.validate.validate_reattach : Comparing two schemas.
    """

    fields: tuple[SchemaField, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> list[str]:
        """Return just the column names, in schema order.

        The common case when only names are needed: checking membership,
        reporting a difference: without unpacking the fields.

        Returns
        -------
        list of str
            The names in order.
        """
        return [f.name for f in self.fields]

    def to_dict(self) -> dict[str, Any]:
        """Convert the schema to JSON-safe plain data.

        Used when writing checkpoint metadata, where the schema has to be
        readable by a later version, or by something that is not BuildML.

        Returns
        -------
        dict
            ``{'fields': [{'name', 'dtype', 'nullable'}, ...]}`` in schema
            order.

        See Also
        --------
        from_dict : The inverse.
        """
        return {
            "fields": [
                {"name": f.name, "dtype": f.dtype, "nullable": f.nullable} for f in self.fields
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TableSchema:
        """Rebuild a schema from the plain data produced by :meth:`to_dict`.

        Tolerant by design, because it reads metadata that older versions wrote.
        A missing ``fields`` key yields an empty schema, and a field without
        ``nullable`` defaults to nullable.

        Parameters
        ----------
        payload:
            The mapping to read, normally straight from a checkpoint's
            ``meta.json``.

        Returns
        -------
        TableSchema
            The reconstructed schema, preserving the order in the payload.

        Raises
        ------
        KeyError
            If a field entry lacks ``name`` or ``dtype``. These have no
            defensible default, so a malformed payload fails loudly rather than
            producing a schema that silently omits a column.

        Examples
        --------
        >>> schema = TableSchema.from_dict(
        ...     {"fields": [{"name": "age", "dtype": "int64"}]}
        ... )
        >>> schema.columns
        ['age']
        >>> schema.fields[0].nullable
        True
        >>> TableSchema.from_dict({}).columns
        []

        See Also
        --------
        to_dict : The inverse.
        """
        fields = tuple(
            SchemaField(
                name=str(item["name"]),
                dtype=str(item["dtype"]),
                nullable=bool(item.get("nullable", True)),
            )
            for item in payload.get("fields", [])
        )
        return cls(fields=fields)
