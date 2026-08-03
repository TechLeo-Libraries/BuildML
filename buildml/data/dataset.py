"""Hold a table, and postpone loading all of it for as long as possible.

Scikit-learn needs a NumPy array in memory. That is not negotiable, and it sets
a hard ceiling on what can be fitted. But most of what happens before fitting :
selecting columns, filtering rows, aggregating, sampling: does not need
everything in memory, and doing it in Polars or DuckDB first can mean the array
that eventually gets built is a fraction of the size.

A :class:`Dataset` therefore holds up to two views of the same table: a pandas
frame for anything sklearn-facing, and optionally an engine-native handle for
work that can stay out of pandas. Operations prefer the native path and leave
the pandas cache marked stale, promoting only when something actually asks for
a frame. A Polars ``LazyFrame`` goes further and does not execute at all until
that moment.

Two things follow from this design and are worth stating plainly. **Native
handles do not enable out-of-core fitting**: they narrow what must be
materialised, and the estimator boundary is still a hard limit. And **DuckDB
connections need closing**: a Dataset that opened one owns it, derived Datasets
share it, and ``with dataset:`` releases it on the way out.

See Also
--------
buildml.data.engines : The engine adapters.
buildml.data.splits : Partitioning a dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole, DataMode, EngineName, SchemaField, TableSchema
from buildml.core.validation import validate_column_names, validate_role_name
from buildml.ingest.detect import schema_from_dataframe


@dataclass
class Dataset:
    """Tabular dataset handle owned by a :class:`~buildml.session.Session`.

    The Session may keep a Pandas ``frame`` for sklearn-facing work and, when
    an optional engine is attached, a ``native`` Polars/DuckDB table used for
    projection, filter, and sample before materialization.

    Parameters
    ----------
    frame:
        Pandas DataFrame cache. May be rebuilt from ``native`` on demand.
    schema:
        Column schema.
    mode:
        Current data mode.
    engine:
        Current engine name.
    source:
        Provenance string (path or ``dataframe``).
    roles:
        Mapping of column name → :class:`~buildml.core.types.ColumnRole`.
    native:
        Optional engine-native table (Polars DataFrame/LazyFrame or DuckDB
        relation). When set, prep paths prefer native ops before ``to_pandas``.
        A Polars ``LazyFrame`` collects only on Pandas / sklearn promotion.
    _owns_native_connection:
        When True and ``engine`` is DuckDB, this Dataset owns the connection
        inside the ``DuckDBTable`` handle and will close it on
        :meth:`close_native`. Derived project/filter Datasets share the
        connection with ``_owns_native_connection=False``.

    Notes
    -----
    Sklearn estimators still require an in-memory design matrix. Native handles
    avoid full-width Pandas round-trips for project/filter/sample; they do not
    enable out-of-core fitting.

    DuckDB connection ownership
    ---------------------------
    Root Datasets created by native ingest / ``attach_native`` own the DuckDB
    connection. ``get_engine('duckdb')`` returns a cached adapter that does not
    open a connection per call; relation ops reuse ``DuckDBTable.connection``.
    Call :meth:`close_native` on the owner when finished (tests should always
    close) so connections are not leaked. Shared derived handles must not close
    the owner's connection.

    ``with dataset:`` (and ``with session:``) calls :meth:`close_native` on exit
    so owned DuckDB connections are released even when an exception is raised.
    """

    frame: pd.DataFrame
    schema: TableSchema
    mode: DataMode = DataMode.MEMORY
    engine: EngineName = EngineName.PANDAS
    source: str = "dataframe"
    roles: dict[str, ColumnRole] = field(default_factory=dict)
    native: Any | None = None
    _pandas_stale: bool = field(default=False, repr=False)
    _owns_native_connection: bool = field(default=False, repr=False)

    @classmethod
    def from_pandas(
        cls,
        frame: pd.DataFrame,
        *,
        schema: TableSchema | None = None,
        mode: DataMode = DataMode.MEMORY,
        engine: EngineName = EngineName.PANDAS,
        source: str = "dataframe",
        roles: dict[str, ColumnRole] | None = None,
        native: Any | None = None,
        attach_native: bool = False,
    ) -> Dataset:
        """Wrap an in-memory DataFrame as a Dataset.

        The ordinary entry point when the data is already loaded. The schema is
        inferred from dtypes unless supplied.

        Parameters
        ----------
        frame:
            The data. **Copied**, so later edits to the original do not reach
            into the Dataset.
        schema:
            Column types. Inferred when omitted.
        mode:
            How the data is held. Defaults to in-memory.
        engine:
            Which engine backs prep operations. Defaults to pandas.
        source:
            Where it came from, for provenance.
        roles:
            Column to semantic role. Can be set later with :meth:`set_roles`.
        native:
            An engine-native handle for this same data, when one already
            exists.
        attach_native:
            Build a native handle from the frame. Only meaningful when
            ``engine`` is Polars or DuckDB.

        Returns
        -------
        Dataset
            The wrapped data.

        Notes
        -----
        **The copy is deliberate.** A Dataset that aliased its input would
        change underneath you when the original was modified, which is a
        confusing class of bug in a pipeline.

        **Attaching a native handle is an eager conversion.** It costs a full
        pass over the data, and pays off only when the operations that follow
        can stay native.

        Examples
        --------
        Load into a Polars-backed Dataset::

            dataset = Dataset.from_pandas(
                frame, engine=EngineName.POLARS, attach_native=True,
            )

        See Also
        --------
        from_native : When the engine already loaded it.
        """
        resolved_schema = schema or schema_from_dataframe(frame)
        owns = False
        if native is not None and engine == EngineName.DUCKDB:
            owns = True
        ds = cls(
            frame=frame.copy(),
            schema=resolved_schema,
            mode=mode,
            engine=engine,
            source=source,
            roles=dict(roles or {}),
            native=native,
            _pandas_stale=False,
            _owns_native_connection=owns,
        )
        if attach_native and native is None and engine != EngineName.PANDAS:
            ds.attach_native(rebuild=True)
        return ds

    @classmethod
    def from_native(
        cls,
        native: Any,
        *,
        engine: EngineName | str,
        schema: TableSchema | None = None,
        mode: DataMode = DataMode.LAZY,
        source: str = "native",
        roles: dict[str, ColumnRole] | None = None,
        materialize_pandas: bool = False,
    ) -> Dataset:
        """Wrap a table the engine already loaded, skipping pandas entirely.

        The path that avoids a full-width pandas load on ingest. Polars or
        DuckDB reads the file, and the Dataset holds that handle; a pandas frame
        is built only when something needs one.

        Parameters
        ----------
        native:
            A Polars DataFrame or LazyFrame, or a DuckDB relation.
        engine:
            ``'polars'`` or ``'duckdb'``. Pandas is rejected: there would be
            nothing native about it.
        schema:
            Column types. Inferred when the pandas cache is built.
        mode:
            How the data is held. Defaults to lazy.
        source:
            Where it came from.
        roles:
            Column to semantic role.
        materialize_pandas:
            Build the pandas cache immediately, collecting a LazyFrame if
            needed. Leaving this false is the point of the method.

        Returns
        -------
        Dataset
            Backed by the native handle.

        Raises
        ------
        ValidationError
            If ``engine`` is pandas.

        Notes
        -----
        **Without ``materialize_pandas``, the pandas frame is an empty stub with
        the right column names**, and the schema records every column as
        ``object``. Both are replaced the first time the frame is promoted. Do
        not read dtypes off a Dataset in this state.

        **A DuckDB relation brings a connection that must be closed.** This
        Dataset owns it; call :meth:`close_native` or use ``with``.

        **Still not an out-of-core fit path.** Fitting materialises, and the
        estimator boundary is the same limit it always was.

        See Also
        --------
        from_pandas : When the data is already a DataFrame.
        close_native : Releasing the connection.
        """
        chosen = EngineName(engine)
        if chosen == EngineName.PANDAS:
            raise ValidationError("from_native requires engine='polars' or 'duckdb'")
        from buildml.data.engines import get_engine

        adapter = get_engine(chosen)
        columns = adapter.columns(native)
        owns = chosen == EngineName.DUCKDB
        if materialize_pandas:
            frame = adapter.to_pandas(native)
            resolved_schema = schema or schema_from_dataframe(frame)
            return cls(
                frame=frame,
                schema=resolved_schema,
                mode=mode,
                engine=chosen,
                source=source,
                roles=dict(roles or {}),
                native=native,
                _pandas_stale=False,
                _owns_native_connection=owns,
            )
        stub = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
        resolved_schema = schema or TableSchema(
            fields=tuple(SchemaField(name=c, dtype="object", nullable=True) for c in columns)
        )
        return cls(
            frame=stub,
            schema=resolved_schema,
            mode=mode,
            engine=chosen,
            source=source,
            roles=dict(roles or {}),
            native=native,
            _pandas_stale=True,
            _owns_native_connection=owns,
        )

    @classmethod
    def from_transformed(
        cls,
        source: Dataset,
        frame: pd.DataFrame,
        *,
        schema: TableSchema | None = None,
        roles: dict[str, ColumnRole] | None = None,
        sync_native: bool = True,
    ) -> Dataset:
        """Wrap a transformed frame, carrying the original's context forward.

        Preprocessing runs on pandas and hands back a new frame. This rebuilds a
        Dataset around it while preserving mode, engine, source, and roles: so
        a transform does not silently drop the fact that the data came from
        Parquet, or that it was Polars-backed.

        Parameters
        ----------
        source:
            The Dataset the frame came from. Supplies the context.
        frame:
            The transformed data.
        schema:
            Column types. Inferred from the new frame when omitted, which is
            usually right since a transform can change dtypes.
        roles:
            Column to role. Inherited from ``source`` when omitted.
        sync_native:
            Rebuild a native handle from the transformed frame, so subsequent
            prep can go back to using engine operations.

        Returns
        -------
        Dataset
            The transformed data, in context.

        Notes
        -----
        **Roles are inherited wholesale, including for columns that no longer
        exist.** A transform that drops or renames a column should pass
        ``roles`` explicitly.

        **Syncing the native handle is a rebuild, not a replay.** It converts
        the transformed frame into a fresh native table; it does not express
        the transform as engine operations, and it costs a full pass.

        See Also
        --------
        sync_native : Rebuilding on an existing Dataset.
        """
        resolved_roles = dict(source.roles) if roles is None else dict(roles)
        resolved_schema = schema or schema_from_dataframe(frame)
        attach = bool(sync_native and source.engine != EngineName.PANDAS)
        return cls.from_pandas(
            frame,
            schema=resolved_schema,
            mode=source.mode,
            engine=source.engine,
            source=source.source,
            roles=resolved_roles,
            attach_native=attach,
        )

    @property
    def has_native(self) -> bool:
        """True when an engine-native table is currently attached."""
        return self.native is not None and self.engine != EngineName.PANDAS

    @property
    def has_lazy_native(self) -> bool:
        """True when ``native`` is a Polars LazyFrame (collect-on-promote)."""
        if not self.has_native or self.engine != EngineName.POLARS:
            return False
        from buildml.data.engines import get_engine

        engine = get_engine(self.engine)
        checker = getattr(engine, "is_lazy_handle", None)
        return bool(checker(self.native)) if callable(checker) else False

    @property
    def pandas_stale(self) -> bool:
        """True when the Pandas cache must be rebuilt from ``native``."""
        return bool(self._pandas_stale)

    def attach_native(self, *, rebuild: bool = False) -> Any:
        """Build a native engine handle from the current pandas frame.

        Converts eagerly: a full pass over the data: so that subsequent
        projection, filtering, and aggregation can run in the engine rather
        than in pandas.

        Parameters
        ----------
        rebuild:
            Rebuild even when a handle already exists. Needed after mutating the
            pandas frame, since the existing handle then describes stale data.

        Returns
        -------
        Any
            The native table. For the pandas engine, the frame itself.

        Notes
        -----
        **DuckDB rebuilds reuse the existing connection** where one is present,
        so repeated rebuilds do not accumulate connections. The first build
        opens one and this Dataset takes ownership of it.

        **Calling this on the pandas engine clears any handle and returns the
        frame.** There is no native table to build.

        See Also
        --------
        sync_native : Rebuild, spelled for the post-transform case.
        close_native : Releasing what this acquired.
        """
        if self.engine == EngineName.PANDAS:
            self.native = None
            self._owns_native_connection = False
            return self.frame
        if self.native is not None and not rebuild:
            return self.native
        from buildml.data.engines import get_engine

        engine = get_engine(self.engine)
        frame = self._ensure_pandas()
        if self.engine == EngineName.DUCKDB:
            from buildml.data.engines.duckdb_engine import DuckDBTable

            reuse = None
            if isinstance(self.native, DuckDBTable):
                reuse = self.native.connection
            self.native = engine.from_pandas(frame, connection=reuse)
            if reuse is None:
                self._owns_native_connection = True
        else:
            self.native = engine.from_pandas(frame)
            self._owns_native_connection = False
        self._pandas_stale = False
        return self.native

    def sync_native(self) -> Any:
        """Bring the native handle back into agreement with the pandas frame.

        Call this after pandas-side transforms on a Polars or DuckDB-backed
        Dataset. Without it, the native handle still describes the data as it
        was before the transform, and later native operations would silently
        act on the old table.

        Returns
        -------
        Any
            The rebuilt native table.

        Notes
        -----
        **This costs a full conversion.** Batch pandas transforms and sync once
        at the end rather than after each step.

        No effect for the pandas engine.

        See Also
        --------
        attach_native : The general form.
        invalidate_native : Dropping the handle instead of rebuilding it.
        """
        return self.attach_native(rebuild=True)

    def close_native(self) -> None:
        """Drop the native handle and close an owned DuckDB connection.

        Derived Datasets that share a connection
        (``_owns_native_connection=False``) only drop their handle. Call this
        on the owner Dataset when finished so connections are not leaked.
        Safe to call repeatedly.
        """
        if (
            self.engine == EngineName.DUCKDB
            and self._owns_native_connection
            and self.native is not None
        ):
            from buildml.data.engines.duckdb_engine import DuckDBTable, close_duckdb_connection

            if isinstance(self.native, DuckDBTable):
                close_duckdb_connection(self.native.connection)
        self.native = None
        self._owns_native_connection = False
        self._pandas_stale = False

    def __enter__(self) -> Dataset:
        """Enter a scope that releases native resources on the way out.

        Makes ``with dataset:`` release an owned DuckDB connection even when
        the block raises, which is the difference between a leaked connection
        and a closed one.

        Returns
        -------
        Dataset
            This Dataset.

        See Also
        --------
        close_native : What the exit performs.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Release owned native resources, whether or not the block succeeded.

        Closes an owned DuckDB connection and drops the native handle on the way
        out of ``with dataset:``.

        Parameters
        ----------
        exc_type:
            The exception type, if one is propagating.
        exc:
            The exception.
        tb:
            The traceback.

        Notes
        -----
        Nothing is suppressed: an exception from the block propagates after
        the connection is closed.
        """
        self.close_native()

    def clear_native(self) -> None:
        """Drop the native handle (keeps the Pandas frame).

        Equivalent to :meth:`close_native` for ownership-safe release.
        """
        self.close_native()

    def invalidate_native(self) -> None:
        """Clear the native handle after a Pandas-side mutation.

        Keeps ``engine`` metadata so a later :meth:`sync_native` can rebuild.
        Closes an owned DuckDB connection; shared handles are dropped only.
        Does not imply that subsequent ops remain lazy.
        """
        if self._owns_native_connection:
            self.close_native()
            return
        self.native = None

    def _ensure_pandas(self) -> pd.DataFrame:
        if self._pandas_stale and self.native is not None:
            from buildml.data.engines import get_engine

            self.frame = get_engine(self.engine).to_pandas(self.native)
            self._pandas_stale = False
            if not self.schema.fields:
                self.schema = schema_from_dataframe(self.frame)
            elif all(f.dtype == "object" for f in self.schema.fields) and len(self.frame):
                # Native-ingest stubs use object placeholders; refresh after promote.
                self.schema = schema_from_dataframe(self.frame)
        return self.frame

    @property
    def columns(self) -> list[str]:
        """Return the column names, in order.

        Read from the native handle when one is attached, so this stays correct
        even while the pandas cache is a stub.

        Returns
        -------
        list of str
            Column names.

        Notes
        -----
        **Cheap on every backing.** Even a LazyFrame knows its schema without
        executing anything, so this never triggers a collect.
        """
        if self.has_native:
            from buildml.data.engines import get_engine

            return get_engine(self.engine).columns(self.native)
        return list(self.frame.columns.astype(str))

    @property
    def n_rows(self) -> int:
        """Return the number of rows.

        Counted through the native handle when one is attached, rather than by
        materialising the frame.

        Returns
        -------
        int
            The row count.

        Notes
        -----
        **A LazyFrame must execute its plan to be counted.** On a lazy Polars
        Dataset this is not free, and on an expensive plan it is not fast
        either. Prefer :attr:`columns` when only the schema is needed.
        """
        if self.has_native:
            from buildml.data.engines import get_engine

            return int(get_engine(self.engine).n_rows(self.native))
        return int(len(self.frame))

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first few rows, for looking at the data.

        Pulled through the native handle when there is one, so peeking at a
        large table does not materialise it.

        Parameters
        ----------
        n:
            How many rows.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows.

        Notes
        -----
        **The first rows are not a sample.** Sorted or grouped data shows one
        corner of itself here. Use :meth:`sample` when you want something
        representative.

        See Also
        --------
        sample : A random draw.
        """
        if self.has_native:
            from buildml.data.engines import get_engine

            return get_engine(self.engine).head(self.native, n)
        return self.frame.head(n).copy()

    def sample(self, n: int = 5, *, random_state: int | None = None) -> pd.DataFrame:
        """Return a random draw of rows.

        Sampled in the engine when a native handle is attached, so a
        representative look at a large table does not require loading it.

        Parameters
        ----------
        n:
            How many rows. Clamped to the row count.
        random_state:
            Seed, for a reproducible draw.

        Returns
        -------
        pandas.DataFrame
            The sampled rows.

        Notes
        -----
        **Without a seed, every call returns something different.** Fine for
        exploration, and a problem for anything that gets compared across runs.

        **Sampling counts rows, so a LazyFrame executes.** The plan runs to
        determine how many rows exist before the draw can be clamped.

        See Also
        --------
        head : The first rows, without a count.
        """
        n = min(int(n), self.n_rows)
        if n <= 0:
            return self.head(0)
        if self.has_native:
            from buildml.data.engines import get_engine

            engine = get_engine(self.engine)
            sampled = engine.sample_rows(self.native, n, random_state=random_state)
            return engine.to_pandas(sampled)
        return self.frame.sample(n=n, random_state=random_state).copy()

    def project(self, columns: Sequence[str], *, materialize: bool = False) -> Dataset:
        """Keep only the named columns, dropping the rest in the engine.

        **The single most effective way to reduce what eventually gets
        materialised.** A table with three hundred columns of which twelve are
        modelled costs twenty-five times more to load than it needs to.
        Projecting first means the engine never reads the rest.

        Parameters
        ----------
        columns:
            Which to keep. Order is preserved as given, so this reorders as well
            as selects.
        materialize:
            Force a pandas result and drop the native handle. Leave false to
            keep the projection native.

        Returns
        -------
        Dataset
            A new Dataset with only those columns. Roles are carried across for
            the columns that survive.

        Raises
        ------
        ValidationError
            If any named column does not exist. The message lists the missing
            ones.

        Notes
        -----
        **The result shares the parent's DuckDB connection without owning it.**
        Closing the parent closes it for the projection too. Keep the owner
        alive for as long as anything derived from it is in use.

        **Project before materialising, not after.** Projecting a frame that
        has already been loaded saves nothing that matters.

        Examples
        --------
        Narrow before materialising::

            narrow = dataset.project(["age", "region", "outcome"])
            frame = narrow.to_pandas()

        See Also
        --------
        filter_rows : Narrowing rows instead.
        to_pandas : The materialisation this defers.
        """
        cols = [str(c) for c in columns]
        missing = [c for c in cols if c not in self.columns]
        if missing:
            raise ValidationError(f"project columns missing from dataset: {missing}")
        roles = {k: v for k, v in self.roles.items() if k in cols}
        if self.has_native and not materialize:
            from buildml.data.engines import get_engine

            engine = get_engine(self.engine)
            native = engine.select_columns(self.native, cols)
            # Column stub only; prefer native until an explicit Pandas promote.
            stub = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
            return Dataset(
                frame=stub,
                schema=TableSchema(
                    fields=tuple(SchemaField(name=c, dtype="object", nullable=True) for c in cols)
                ),
                mode=self.mode,
                engine=self.engine,
                source=self.source,
                roles=roles,
                native=native,
                _pandas_stale=True,
                _owns_native_connection=False,
            )
        frame = self._ensure_pandas().loc[:, cols].copy()
        return Dataset.from_pandas(
            frame,
            schema=schema_from_dataframe(frame),
            mode=self.mode,
            engine=EngineName.PANDAS if materialize else self.engine,
            source=self.source,
            roles=roles,
            attach_native=not materialize and self.engine != EngineName.PANDAS,
        )

    def aggregate(
        self,
        aggregations: dict[str, str | list[str]],
        *,
        by: Sequence[str] | None = None,
        materialize: bool = False,
    ) -> Dataset:
        """Summarise columns, optionally grouped, in the engine.

        Turns a large table into a small one: counts per category, means per
        group, a global summary row. Runs natively where possible, which is what
        makes it usable on data that would not fit in memory.

        Parameters
        ----------
        aggregations:
            Mapping of column → aggregate function or list of functions.
            Supported functions: ``sum``, ``mean``, ``min``, ``max``,
            ``count``, ``n_unique``, ``std``, ``median``, and integer
            percentiles ``q0``..``q100`` (also ``quantile_0.25`` → ``q25``).
            Use ``{"*": "count"}`` for a row count. Output columns are named
            ``{column}_{func}`` (or ``count`` for ``*``).
        by:
            Optional group-by columns. When omitted, returns one summary row.
        materialize:
            Force a pandas result.

        Returns
        -------
        Dataset
            The summary table. Roles are cleared, since the columns are new.

        Notes
        -----
        **This is a reporting helper, not a modelling transform.** It is not
        fold-local and not part of
        :class:`~buildml.preprocess.fold.PreprocessRecipe`. Aggregating over the
        whole table and feeding the result back in as a feature is a leak :
        target encoding and similar group statistics belong in preprocessing,
        where they are fitted on train only.

        **Quantiles can differ slightly across engines.** Pandas and Polars
        interpolate linearly; DuckDB uses ``quantile_cont``. Ties are where they
        diverge. Pass ``materialize=True`` when the pandas value is the one that
        matters.

        Examples
        --------
        Mean and count per region::

            summary = dataset.aggregate(
                {"revenue": ["mean", "sum"], "*": "count"}, by=["region"],
            )

        See Also
        --------
        buildml.preprocess : Where fold-local statistics belong.
        """
        from buildml.data.engines import get_engine
        from buildml.data.engines.aggregate import (
            aggregate_pandas,
            normalize_aggregations,
            validate_aggregate_columns,
        )

        pairs = normalize_aggregations(aggregations)
        by_cols = None if by is None else [str(c) for c in by]
        validate_aggregate_columns(self.columns, by_cols, pairs)
        if self.has_native and not materialize:
            engine = get_engine(self.engine)
            native = engine.aggregate(
                self.native,
                {k: v for k, v in aggregations.items()},
                by=by_cols,
            )
            cols = engine.columns(native)
            stub = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
            return Dataset(
                frame=stub,
                schema=TableSchema(
                    fields=tuple(SchemaField(name=c, dtype="object", nullable=True) for c in cols)
                ),
                mode=self.mode,
                engine=self.engine,
                source=self.source,
                roles={},
                native=native,
                _pandas_stale=True,
                _owns_native_connection=False,
            )
        frame = aggregate_pandas(self._ensure_pandas(), pairs, by=by_cols)
        return Dataset.from_pandas(
            frame,
            schema=schema_from_dataframe(frame),
            mode=self.mode,
            engine=EngineName.PANDAS if materialize else self.engine,
            source=self.source,
            roles={},
            attach_native=not materialize and self.engine != EngineName.PANDAS,
        )

    def filter_rows(self, mask: Sequence[bool], *, materialize: bool = False) -> Dataset:
        """Keep the rows where the mask is true.

        The Python-side filter: you supply one boolean per row and the engine
        keeps the true ones. Use it when the condition is easier to express in
        Python than as an engine expression.

        Parameters
        ----------
        mask:
            One boolean per row, aligned to current order. Must be exactly as
            long as the table.
        materialize:
            Force a pandas result.

        Returns
        -------
        Dataset
            The surviving rows, with roles preserved.

        Raises
        ------
        ValidationError
            If the mask length does not match the row count. A mismatch means
            the mask was built against different data, so filtering by it would
            silently keep the wrong rows.

        Notes
        -----
        **Building the mask usually requires reading the column it tests**, so
        the saving here is smaller than with :meth:`filter_expr`, where the
        predicate runs inside the engine and the source is never fully read.

        **The result shares the parent's DuckDB connection without owning it.**

        See Also
        --------
        filter_expr : Pushing the predicate into the engine.
        project : Narrowing columns instead.
        """
        mask_list = [bool(v) for v in mask]
        if len(mask_list) != self.n_rows:
            raise ValidationError(
                f"filter mask length {len(mask_list)} does not match dataset rows {self.n_rows}"
            )
        if self.has_native and not materialize:
            from buildml.data.engines import get_engine

            engine = get_engine(self.engine)
            native = engine.filter_rows(self.native, mask_list)
            cols = engine.columns(native)
            stub = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
            return Dataset(
                frame=stub,
                schema=TableSchema(
                    fields=tuple(SchemaField(name=c, dtype="object", nullable=True) for c in cols)
                ),
                mode=self.mode,
                engine=self.engine,
                source=self.source,
                roles=dict(self.roles),
                native=native,
                _pandas_stale=True,
                _owns_native_connection=False,
            )
        frame = self._ensure_pandas().loc[mask_list].copy()
        return Dataset.from_pandas(
            frame,
            schema=schema_from_dataframe(frame),
            mode=self.mode,
            engine=EngineName.PANDAS if materialize else self.engine,
            source=self.source,
            roles=dict(self.roles),
            attach_native=not materialize and self.engine != EngineName.PANDAS,
        )

    def filter_expr(self, expression: str, *, materialize: bool = False) -> Dataset:
        """Keep rows matching a predicate evaluated inside the engine.

        The efficient filter. Because the condition is a string the engine
        understands, it is applied during the scan: rows that fail are never
        read into memory at all, and on a LazyFrame nothing executes until
        something asks for the result.

        Parameters
        ----------
        expression:
            Engine predicate. DuckDB accepts SQL boolean expressions
            (for example ``\"a\" > 1``). Polars accepts SQL-style predicates
            via ``sql_expr`` when available (for example ``a > 1`` or
            ``\"a\" > 1``), keeping LazyFrames lazy. For simple comparisons
            that should work on both engines, prefer
            :func:`~buildml.data.filter_syntax.portable_filter_expr`.
            Engines without ``filter_expr`` raise
            :class:`~buildml.core.errors.ValidationError`.
        materialize:
            Force a pandas result.

        Returns
        -------
        Dataset
            The matching rows, with roles preserved.

        Raises
        ------
        ValidationError
            If no native handle is attached, or if the engine has no
            expression-filter support. There is no pandas fallback: silently
            evaluating engine SQL in pandas would mean two different dialects
            answering the same question.

        Notes
        -----
        **Dialects are not portable.** DuckDB takes SQL; Polars takes SQL-style
        predicates through ``sql_expr``. Simple comparisons written through
        :func:`~buildml.data.filter_syntax.portable_filter_expr` work on both.
        Joins, window functions, and engine-specific builtins do not.

        Examples
        --------
        Filter during the scan::

            recent = dataset.filter_expr("year >= 2020")

        See Also
        --------
        filter_rows : When the condition is easier in Python.
        buildml.data.filter_syntax.portable_filter_expr : Cross-engine
            predicates.
        """
        if self.has_native and not materialize:
            from buildml.data.engines import get_engine

            engine = get_engine(self.engine)
            pusher = getattr(engine, "filter_expr", None)
            if not callable(pusher):
                raise ValidationError(
                    f"filter_expr is not supported for engine={self.engine.value}; "
                    "use filter_rows(mask=...) or materialize to Pandas first."
                )
            native = pusher(self.native, expression)
            cols = engine.columns(native)
            stub = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
            return Dataset(
                frame=stub,
                schema=TableSchema(
                    fields=tuple(SchemaField(name=c, dtype="object", nullable=True) for c in cols)
                ),
                mode=self.mode,
                engine=self.engine,
                source=self.source,
                roles=dict(self.roles),
                native=native,
                _pandas_stale=True,
                _owns_native_connection=False,
            )
        raise ValidationError(
            "filter_expr requires an attached native engine handle "
            f"(engine={self.engine.value}, has_native={self.has_native})"
        )

    def to_pandas(
        self,
        *,
        hard_limit_bytes: int | None = None,
    ) -> pd.DataFrame:
        """Load everything into a pandas DataFrame.

        The moment the deferral ends. A LazyFrame executes, a DuckDB relation is
        read, and the whole result lands in memory. Everything sklearn touches
        goes through here.

        Parameters
        ----------
        hard_limit_bytes:
            Refuse above this estimated size. Falls back to
            ``BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES`` when omitted.

        Returns
        -------
        pandas.DataFrame
            A copy of the data. Editing it does not affect the Dataset.

        Raises
        ------
        ValidationError
            If the estimated size exceeds a configured hard limit. No hard limit
            is set by default, so this only fires when one is asked for.

        Notes
        -----
        **Above roughly 250 MiB you get a disclosure**, because the difference
        between a fast pipeline and one that swaps is usually a materialisation
        that nobody meant to perform.

        **The size estimate is of the frame, and pandas is not the peak.** The
        conversion holds both representations briefly, and sklearn then copies
        again into a float array. Budget several times the reported figure.

        **Narrow first.** :meth:`project`, :meth:`filter_expr`, and
        :meth:`aggregate` all reduce what has to be loaded, and all of them are
        only useful before this call.

        See Also
        --------
        project : Dropping columns first.
        filter_expr : Dropping rows first.
        """
        from buildml.ingest.detect import MEMORY_HARD_LIMIT, check_materialization

        frame = self._ensure_pandas()
        check_materialization(
            frame,
            context="Dataset.to_pandas()",
            hard_limit_bytes=(MEMORY_HARD_LIMIT if hard_limit_bytes is None else hard_limit_bytes),
        )
        return frame.copy()

    def to_engine(self, engine: EngineName | str | None = None) -> Any:
        """Hand back the data as a native engine table.

        The escape hatch for doing something in Polars or DuckDB that BuildML
        does not wrap: a join, a window function, engine-specific SQL.

        Parameters
        ----------
        engine:
            Which engine to convert to. Defaults to the Dataset's own, in which
            case an attached handle is returned as-is with no conversion.

        Returns
        -------
        Any
            A Polars DataFrame or LazyFrame, a DuckDB relation, or a pandas
            frame, depending on the target.

        Notes
        -----
        **Converting to a different engine round-trips through pandas**, and
        materialises everything on the way. Converting to the attached engine
        does not.

        **DuckDB conversions reuse an existing connection** rather than opening
        another.

        **The returned handle is outside BuildML's tracking.** Whatever you do
        with it does not update the Dataset, and results have to come back
        through :meth:`from_native` or :meth:`from_pandas`.

        See Also
        --------
        from_native : Bringing the result back.
        """
        from buildml.data.engines import get_engine

        target = EngineName(engine) if engine is not None else self.engine
        if self.has_native and target == self.engine:
            return self.native
        adapter = get_engine(target)
        frame = self._ensure_pandas()
        if target == EngineName.DUCKDB:
            from buildml.data.engines.duckdb_engine import DuckDBTable

            reuse = self.native.connection if isinstance(self.native, DuckDBTable) else None
            return adapter.from_pandas(frame, connection=reuse)
        return adapter.from_pandas(frame)

    def to_parquet(self, path: str | Path) -> Path:
        """Write the data to a Parquet file.

        Parquet keeps dtypes, compresses columnwise, and can be read back
        column-at-a-time: which is what makes the lazy paths in this module
        worthwhile on the next run. CSV throws all three away.

        Parameters
        ----------
        path:
            Where to write. Parent directories are created.

        Returns
        -------
        pathlib.Path
            The path written.

        Notes
        -----
        **The frame is materialised first**, so the same memory considerations
        as :meth:`to_pandas` apply. This is not a streaming write.

        **The index is not written.** Anything meaningful in it should be a
        column before saving.

        See Also
        --------
        to_pandas : The materialisation this performs.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_pandas().to_parquet(destination, index=False)
        return destination

    def set_roles(self, mapping: dict[str, str | ColumnRole]) -> None:
        """Tell BuildML what each column means.

        Roles are how the rest of the library knows which column to predict,
        which to group by when splitting, and which to leave out of the feature
        matrix. Without a target role, modelling cannot start; without a group
        role, a grouped split cannot be built.

        Parameters
        ----------
        mapping:
            Column name to role, either as a
            :class:`~buildml.core.types.ColumnRole` or its string name.

        Raises
        ------
        ValidationError
            If a column does not exist, or a role name is not recognised.

        Notes
        -----
        **Merges rather than replaces.** Columns not mentioned keep the roles
        they had, so roles can be assigned across several calls.

        **A column named as an identifier is excluded from features.** Leaving
        a row ID or account number unmarked lets the model memorise it, which
        scores well and generalises to nothing.

        Examples
        --------
        Mark the target and an ID::

            dataset.set_roles({"churned": "target", "customer_id": "id"})

        See Also
        --------
        role_columns : Reading roles back.
        require_target : Asserting exactly one target.
        """
        validate_column_names(mapping.keys(), self.columns)
        resolved: dict[str, ColumnRole] = {}
        for column, role in mapping.items():
            resolved[column] = validate_role_name(role)
        # Keep previous roles for columns not mentioned.
        self.roles.update(resolved)

    def role_columns(self, role: str | ColumnRole) -> list[str]:
        """Return the columns carrying a given role.

        The reverse lookup over :meth:`set_roles`: given a role, which columns
        were marked with it.

        Parameters
        ----------
        role:
            The role to look up, as an enum member or its string name.

        Returns
        -------
        list of str
            Matching column names. Empty when nothing carries that role.

        Raises
        ------
        ValidationError
            If the role name is not recognised.

        Notes
        -----
        **An empty list is not an error here.** Callers that need a role to be
        present must say so: see :meth:`require_target`.

        See Also
        --------
        set_roles : Assigning them.
        """
        target = validate_role_name(role)
        return [name for name, value in self.roles.items() if value == target]

    def require_target(self) -> str:
        """Return the target column, insisting there is exactly one.

        Supervised learning predicts one thing. This is the assertion that says
        so, called by every path that needs a label.

        Returns
        -------
        str
            The target column name.

        Raises
        ------
        ValidationError
            If there is no target, or more than one. The message reports what
            was found.

        Notes
        -----
        **Two targets usually means a leak.** The second column is often
        something derived from the label: a flag, a bucketed version: which
        would be in the feature matrix if it were not caught here.

        See Also
        --------
        set_roles : Assigning the target.
        """
        targets = self.role_columns(ColumnRole.TARGET)
        if len(targets) != 1:
            raise ValidationError(
                f"Expected exactly one target column, found {len(targets)}: {targets}"
            )
        return targets[0]

    def metadata(self) -> dict[str, Any]:
        """Describe the dataset without including any of its data.

        Shape, schema, roles, engine, and provenance: everything needed to
        record what was used in a run, and nothing that would put row values
        into a log.

        Returns
        -------
        dict
            JSON-safe metadata: source, mode, engine, schema, roles, row count,
            column names, and whether native handles are attached.

        Notes
        -----
        **Column names are included.** They are not values, but in a narrow
        schema they can still be revealing.

        **The row count may execute a lazy plan**, since counting rows requires
        it. Calling this on a lazy Dataset is not always free.

        See Also
        --------
        n_rows : The count this includes.
        """
        return {
            "source": self.source,
            "mode": self.mode.value,
            "engine": self.engine.value,
            "schema": self.schema.to_dict(),
            "roles": {k: v.value for k, v in self.roles.items()},
            "n_rows": self.n_rows,
            "columns": self.columns,
            "has_native": self.has_native,
            "has_lazy_native": self.has_lazy_native,
        }
