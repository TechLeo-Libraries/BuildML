"""Satisfy the engine protocol using pandas itself.

Nothing here is optimised, and nothing is deferred: pandas is already an
in-memory DataFrame, so there is no plan to build and no scan to push work into.
The adapter exists so that code written against the engine protocol runs
unchanged when no optional engine is installed.

The one consistent behaviour worth knowing is that **every method returns a
copy**. Views into a shared frame would let an operation on one Dataset change
another, which is a difficult bug to find; copying costs memory and buys
predictability.

See Also
--------
buildml.data.engines.base.Engine : The contract.
buildml.data.engines.polars_engine : When the data is large enough to matter.
"""

from __future__ import annotations

import pandas as pd

from buildml.core.types import EngineName
from buildml.data.engines.aggregate import (
    aggregate_pandas,
    normalize_aggregations,
    validate_aggregate_columns,
)


class PandasEngine:
    """The default backend: pandas, wrapped in the engine interface.

    Always available, since pandas is a core dependency. Suitable for data that
    comfortably fits in memory; beyond that, an engine that can defer work will
    do better.

    Attributes
    ----------
    name:
        :attr:`~buildml.core.types.EngineName.PANDAS`.

    Notes
    -----
    **Nothing is lazy.** Every operation computes immediately and materialises
    its result, so ``project`` then ``filter`` builds two full intermediates
    where Polars or DuckDB would build none.

    See Also
    --------
    buildml.data.engines.base.Engine : The contract.
    """

    name = EngineName.PANDAS

    def from_pandas(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Take a copy of the frame.

        There is no conversion to do: the copy is the whole operation, and it
        is what stops the Dataset from aliasing the caller's data.

        Parameters
        ----------
        frame:
            The data.

        Returns
        -------
        pandas.DataFrame
            A copy.

        Notes
        -----
        **Peak memory is briefly doubled**, which is the cost of not having the
        Dataset change underneath you when the original is edited.
        """
        return frame.copy()

    def to_pandas(self, table: pd.DataFrame) -> pd.DataFrame:
        """Take a copy of the frame.

        The inverse of :meth:`from_pandas`, and equally trivial: the data is
        already pandas, so only the defensive copy remains.

        Parameters
        ----------
        table:
            The data.

        Returns
        -------
        pandas.DataFrame
            A copy.

        Notes
        -----
        **Unlike the other engines, this materialises nothing**: there was
        never a plan. Code that treats this as the expensive boundary is right
        about Polars and DuckDB and wrong about pandas.
        """
        return table.copy()

    def n_rows(self, table: pd.DataFrame) -> int:
        """Report how many rows the frame holds.

        The count is already known, so unlike the lazy engines this never
        triggers any work.

        Parameters
        ----------
        table:
            The data.

        Returns
        -------
        int
            The row count.

        Notes
        -----
        Free: pandas tracks its own length.
        """
        return int(len(table))

    def columns(self, table: pd.DataFrame) -> list[str]:
        """List the column names as strings, in order.

        Names are coerced to ``str`` so that a frame with integer or tuple
        column labels still presents the string names the rest of BuildML
        assumes.

        Parameters
        ----------
        table:
            The data.

        Returns
        -------
        list of str
            Column names.

        Notes
        -----
        **The coercion can collide.** A frame with both ``1`` and ``"1"`` as
        labels reports the same name twice.
        """
        return list(table.columns.astype(str))

    def head(self, table: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the first few rows, copied.

        A peek at the data, useful for checking that a load or transform
        produced what was expected.

        Parameters
        ----------
        table:
            The data.
        n:
            How many rows.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows.

        Notes
        -----
        **The index comes along.** Positions are not reset, so the result's
        index reflects where the rows sat in the original.
        """
        return table.head(n).copy()

    def select_columns(self, table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Keep only the named columns, copied.

        Slices the frame and copies the result, so the projection does not share
        memory with the original.

        Parameters
        ----------
        table:
            The data.
        columns:
            Which to keep, in the desired order.

        Returns
        -------
        pandas.DataFrame
            The projection.

        Raises
        ------
        KeyError
            If a named column does not exist.

        Notes
        -----
        **This saves less than it does on other engines.** The frame is already
        in memory, so nothing is avoided: the copy is smaller, that is all.
        On a lazy engine the same call prevents columns from being read.
        """
        return table.loc[:, list(columns)].copy()

    def sample_rows(
        self,
        table: pd.DataFrame,
        n: int,
        *,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Draw a random subset of rows.

        Unlike :meth:`head`, this is representative of the whole frame, which
        matters when the data is sorted.

        Parameters
        ----------
        table:
            The data.
        n:
            How many rows. Clamped to the row count.
        random_state:
            Seed, for a reproducible draw.

        Returns
        -------
        pandas.DataFrame
            The sampled rows, with their original index.

        Notes
        -----
        **Sampling without replacement**, so no row appears twice.

        **Reproducible within pandas only.** The same seed selects different
        rows on Polars or DuckDB.
        """
        take = min(int(n), int(len(table)))
        return table.sample(n=take, random_state=random_state).copy()

    def filter_rows(
        self,
        table: pd.DataFrame,
        mask: list[bool] | tuple[bool, ...],
    ) -> pd.DataFrame:
        """Keep the rows where the mask is true.

        The length check runs first, since a mask built against different data
        would otherwise select the wrong rows without complaint.

        Parameters
        ----------
        table:
            The data.
        mask:
            One boolean per row, aligned to current order.

        Returns
        -------
        pandas.DataFrame
            The surviving rows, with their original index.

        Raises
        ------
        ValueError
            If the mask length does not match the row count.

        Notes
        -----
        **The index is preserved, not reset.** Downstream code that assumes
        contiguous positions after filtering will misbehave; call
        ``reset_index(drop=True)`` if that is what it expects.
        """
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
        """Summarise columns, optionally per group.

        Validates the request against the frame's columns before computing, so
        a typo produces a clear message rather than a pandas ``KeyError``.

        Parameters
        ----------
        table:
            The data.
        aggregations:
            Column to function name, or to a list of names. See
            :meth:`~buildml.data.engines.base.Engine.aggregate` for the
            supported set.
        by:
            Group-by columns. Omit for a single summary row.

        Returns
        -------
        pandas.DataFrame
            The summary, with output columns named ``{column}_{func}``.

        Raises
        ------
        ValidationError
            If a named column does not exist, or a function is not supported.

        Notes
        -----
        **Everything is held in memory at once**, including the intermediate
        groups. This is the engine most likely to run out of room on a large
        group-by.

        **Quantiles interpolate linearly**, which is the reference behaviour the
        other engines are compared against.
        """
        pairs = normalize_aggregations(aggregations)
        validate_aggregate_columns(self.columns(table), by, pairs)
        return aggregate_pandas(table, pairs, by=by)
