"""Dataset handle with optional engine-native Polars/DuckDB backing."""

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
        """Create a Dataset from a Pandas DataFrame.

        Parameters
        ----------
        attach_native:
            When True and ``engine`` is Polars or DuckDB, also build and store
            a native handle from ``frame``.
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
        """Create a Dataset from an engine-native table without a Pandas-first load.

        Parameters
        ----------
        native:
            Polars DataFrame/LazyFrame or DuckDB relation already loaded by the
            engine.
        engine:
            ``polars`` or ``duckdb``.
        materialize_pandas:
            When True, immediately promote a Pandas cache (collecting a
            LazyFrame if needed). When False (default), the Pandas ``frame`` is
            a column stub and ``_pandas_stale`` is True until :meth:`to_pandas`
            / preprocess / sklearn boundaries promote it.

        Notes
        -----
        This is not an out-of-core fit path. Sklearn and train-fitted preprocess
        still promote to Pandas when they need a design matrix.
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
        """Build a Dataset after a Pandas/sklearn transform.

        Parameters
        ----------
        source:
            Dataset that produced ``frame`` (provides mode/engine/source).
        frame:
            Transformed Pandas frame.
        sync_native:
            When True and ``source.engine`` is Polars/DuckDB, rebuild an eager
            native handle from ``frame`` so later ``project`` /
            ``prepare_design_frame`` paths keep using engine ops.

        Notes
        -----
        Preprocess transforms themselves still run on Pandas. A synced native
        handle is an explicit post-transform rebuild, not a lazy plan of the
        transform steps.
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
        """Ensure a native engine table is attached for the configured engine.

        Parameters
        ----------
        rebuild:
            When True, rebuild from the current Pandas frame even if a native
            handle already exists. This is an eager conversion, not a lazy plan.
            DuckDB rebuilds reuse the existing connection when present.

        Returns
        -------
        The native table object.
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
        """Rebuild the native handle from the current Pandas frame (eager).

        Use after a sequence of Pandas-backed transforms when ``engine`` is
        Polars/DuckDB. Does nothing meaningful for the Pandas engine.
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
        """Return ``self`` for ``with dataset:`` ownership scopes."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Release owned native resources via :meth:`close_native`."""
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
        if self.has_native:
            from buildml.data.engines import get_engine

            return get_engine(self.engine).columns(self.native)
        return list(self.frame.columns.astype(str))

    @property
    def n_rows(self) -> int:
        if self.has_native:
            from buildml.data.engines import get_engine

            return int(get_engine(self.engine).n_rows(self.native))
        return int(len(self.frame))

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows as a DataFrame copy."""
        if self.has_native:
            from buildml.data.engines import get_engine

            return get_engine(self.engine).head(self.native, n)
        return self.frame.head(n).copy()

    def sample(self, n: int = 5, *, random_state: int | None = None) -> pd.DataFrame:
        """Return a random sample of rows (materialized Pandas copy).

        Prefers the native engine when attached; sklearn still receives Pandas.
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
        """Return a column-projected Dataset, preferring native engine ops.

        Parameters
        ----------
        columns:
            Columns to keep (order preserved).
        materialize:
            When True, force a Pandas frame on the result and clear ``native``.
            When False (default) and a native handle exists, keep a projected
            native table and mark the Pandas cache stale until ``to_pandas``.

        Notes
        -----
        Prefer ``project`` before ``to_pandas`` / sklearn materialization so
        Polars and DuckDB can drop unused columns natively.
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
        """Return grouped or global aggregations, preferring native engine ops.

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
            When True, force a Pandas-only result.

        Notes
        -----
        Aggregation is a tabular prep helper, not a modeling transform. It does
        not learn fold-local statistics and is not part of
        :class:`~buildml.preprocess.fold.PreprocessRecipe`. Roles are cleared
        on the result because the schema is a new summary table.

        Quantiles use continuous/linear interpolation on Pandas and Polars, and
        ``quantile_cont`` on DuckDB. Cross-engine values can differ slightly on
        ties; pass ``materialize=True`` when exact Pandas semantics are required.
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
        """Return a row-filtered Dataset, preferring native engine ops.

        Parameters
        ----------
        mask:
            Boolean mask aligned to current row order.
        materialize:
            When True, result is Pandas-only.
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
        """Filter rows with an engine-native SQL/expression predicate when supported.

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
            When True, force a Pandas-only result.

        Notes
        -----
        Prefer this over a Python boolean mask when the predicate can run as a
        native engine expression so the source table is not fully collected
        first. Complex SQL (joins, window functions, engine-only builtins)
        remains engine-specific.
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
        """Materialize a Pandas copy for sklearn or other in-memory consumers.

        Soft gates disclose when the frame exceeds ~250 MiB. Hard gates refuse
        when ``hard_limit_bytes`` or ``BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES``
        is set. Prefer native project/filter/sample first, then materialize only
        the design matrix needed at the estimator boundary.
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
        """Convert the current data into a native engine table.

        When ``engine`` matches the attached native handle, returns that handle
        without a Pandas round-trip. DuckDB conversions reuse an existing
        ``DuckDBTable`` connection when present.
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
        """Write the dataset to a Parquet file.

        Parameters
        ----------
        path:
            Destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_pandas().to_parquet(destination, index=False)
        return destination

    def set_roles(self, mapping: dict[str, str | ColumnRole]) -> None:
        """Assign semantic roles to columns.

        Parameters
        ----------
        mapping:
            Column name → role.

        Raises
        ------
        ValidationError
            If a column or role is invalid.
        """
        validate_column_names(mapping.keys(), self.columns)
        resolved: dict[str, ColumnRole] = {}
        for column, role in mapping.items():
            resolved[column] = validate_role_name(role)
        # Keep previous roles for columns not mentioned.
        self.roles.update(resolved)

    def role_columns(self, role: str | ColumnRole) -> list[str]:
        """Return columns assigned to a role."""
        target = validate_role_name(role)
        return [name for name, value in self.roles.items() if value == target]

    def require_target(self) -> str:
        """Return the single target column or raise."""
        targets = self.role_columns(ColumnRole.TARGET)
        if len(targets) != 1:
            raise ValidationError(
                f"Expected exactly one target column, found {len(targets)}: {targets}"
            )
        return targets[0]

    def metadata(self) -> dict[str, Any]:
        """Serializable dataset metadata (no row payload)."""
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
