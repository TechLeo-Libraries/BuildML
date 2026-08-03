"""The small set of operations every tabular engine must provide.

Pandas, Polars, and DuckDB disagree about almost everything — how a table is
represented, whether work is eager or lazy, what a filter looks like. This
protocol names the handful of operations BuildML actually needs from a table
before modelling starts, so the rest of the library can call them without
knowing which engine is underneath.

The set is deliberately small. Every method here is either something that
narrows the data before it reaches memory (project, filter, aggregate, sample)
or something needed to describe it (columns, row count, head). Anything richer
belongs in engine-specific code, reached through
:meth:`~buildml.data.dataset.Dataset.to_engine`.

Two operations are optional and absent from the protocol: ``filter_expr``, which
pushes a predicate string into the engine, and ``is_lazy_handle``. Adapters
provide them where the engine supports them, and callers check with
:func:`getattr` before use.

See Also
--------
buildml.data.dataset.Dataset : The caller.
buildml.data.engines.get_engine : Obtaining an adapter.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from buildml.core.types import EngineName


@runtime_checkable
class Engine(Protocol):
    """What BuildML requires of a tabular backend.

    Implemented by the pandas, Polars, and DuckDB adapters in this package.
    Because it is a :class:`~typing.Protocol`, conformance is structural — an
    adapter satisfies it by having the right methods, without inheriting
    anything.

    Attributes
    ----------
    name:
        Which engine this is. Used to resolve adapters and to record in
        metadata which backend produced a result.

    Notes
    -----
    **The ``table`` argument is opaque and engine-specific.** A Polars adapter
    receives a Polars object, a DuckDB adapter a relation. Passing one engine's
    table to another's adapter is a bug the protocol cannot catch, since every
    signature types it as :class:`~typing.Any`.

    **``runtime_checkable`` only checks that methods exist**, not their
    signatures. ``isinstance`` against this protocol is a weak assertion.
    """

    name: EngineName

    def from_pandas(self, frame: pd.DataFrame) -> Any:
        """Convert a DataFrame into this engine's table type.

        The entry into the engine, used when data arrived as pandas but the
        prep work should run natively.

        Parameters
        ----------
        frame:
            The data.

        Returns
        -------
        Any
            An engine-native table.

        Notes
        -----
        **Eager, and a full pass over the data.** Worth it only when the
        operations that follow can stay native.

        The DuckDB adapter also accepts a ``connection`` argument so repeated
        conversions can reuse one connection rather than opening several.
        """
        ...

    def to_pandas(self, table: Any) -> pd.DataFrame:
        """Materialise an engine table as a DataFrame.

        The exit from the engine. Lazy plans execute here, and the entire result
        lands in memory.

        Parameters
        ----------
        table:
            An engine-native table.

        Returns
        -------
        pandas.DataFrame
            The materialised data.

        Notes
        -----
        **This is the memory event.** Narrow with :meth:`select_columns` and
        :meth:`filter_rows` first; afterwards there is nothing left to save.
        """
        ...

    def n_rows(self, table: Any) -> int:
        """Report how many rows the table holds.

        Needed for sizing, for validating mask lengths, and for metadata.

        Parameters
        ----------
        table:
            An engine-native table.

        Returns
        -------
        int
            The row count.

        Notes
        -----
        **Not always cheap.** A lazy plan must execute to be counted, so this
        can be the operation that triggers all the deferred work.
        """
        ...

    def columns(self, table: Any) -> list[str]:
        """List the column names, in order.

        The schema question, asked without touching any values.

        Parameters
        ----------
        table:
            An engine-native table.

        Returns
        -------
        list of str
            Column names.

        Notes
        -----
        **Cheap everywhere**, including on lazy plans, since schemas are known
        without execution.
        """
        ...

    def head(self, table: Any, n: int = 5) -> pd.DataFrame:
        """Return the first few rows as pandas, for inspection.

        A peek at the data that does not require materialising the table.

        Parameters
        ----------
        table:
            An engine-native table.
        n:
            How many rows.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows.

        Notes
        -----
        **Returns pandas, not a native table**, because the only reason to ask
        for the first few rows is to look at them.

        Engines may still scan more than ``n`` rows to produce them, depending
        on the plan.
        """
        ...

    def select_columns(self, table: Any, columns: list[str]) -> Any:
        """Keep only the named columns.

        The highest-value operation in this protocol: on a wide table, dropping
        unused columns before materialising is the difference between a load
        that fits in memory and one that does not.

        Parameters
        ----------
        table:
            An engine-native table.
        columns:
            Which to keep, in the desired order.

        Returns
        -------
        Any
            A projected native table, still in the engine.

        Notes
        -----
        **Stays native.** The result is not materialised, so projections can be
        chained with filters before a single load at the end.
        """
        ...

    def sample_rows(
        self,
        table: Any,
        n: int,
        *,
        random_state: int | None = None,
    ) -> Any:
        """Draw a random subset of rows.

        Unlike :meth:`head`, this is representative of the whole table, which
        matters when the data is sorted or grouped.

        Parameters
        ----------
        table:
            An engine-native table.
        n:
            How many rows to draw.
        random_state:
            Seed, for a reproducible draw.

        Returns
        -------
        Any
            A sampled native table.

        Notes
        -----
        **Seeding is honoured per engine, and the engines do not agree.** The
        same seed produces different rows on Polars and DuckDB; reproducibility
        holds within an engine, not across them.

        Fewer than ``n`` rows come back when the table is smaller.
        """
        ...

    def filter_rows(self, table: Any, mask: list[bool] | tuple[bool, ...]) -> Any:
        """Keep the rows where the mask is true.

        The row-narrowing counterpart to :meth:`select_columns`, driven by a
        boolean computed in Python.

        Parameters
        ----------
        table:
            An engine-native table.
        mask:
            One boolean per row, aligned to current order.

        Returns
        -------
        Any
            A filtered native table.

        Notes
        -----
        **The mask must be exactly as long as the table.** A shorter or longer
        one means it was built against different data, and applying it would
        keep the wrong rows.

        **A Python-side mask forces the rows to be enumerated**, so the saving
        is smaller than with a native predicate. Adapters that offer
        ``filter_expr`` push the condition into the scan instead.
        """
        ...

    def aggregate(
        self,
        table: Any,
        aggregations: dict[str, str | list[str]],
        *,
        by: list[str] | None = None,
    ) -> Any:
        """Summarise columns, optionally per group.

        Turns a large table into a small one inside the engine, which is what
        makes summarising data too large for memory possible at all.

        Parameters
        ----------
        table:
            An engine-native table.
        aggregations:
            Column to function name, or to a list of function names. Supported:
            ``sum``, ``mean``, ``min``, ``max``, ``count``, ``n_unique``,
            ``std``, ``median``, and integer percentiles ``q0``..``q100``. Use
            ``{"*": "count"}`` for a row count.
        by:
            Group-by columns. Omit for a single summary row.

        Returns
        -------
        Any
            A native table of results, one row per group. Output columns are
            named ``{column}_{func}``.

        Notes
        -----
        **Quantiles differ slightly across engines** on tied values, because the
        interpolation rules are not identical.

        **Not a modelling transform.** Aggregates computed over a whole table
        and fed back as features leak the test set; group statistics used as
        features belong in :mod:`buildml.preprocess`, where they are fitted on
        train only.
        """
        ...
