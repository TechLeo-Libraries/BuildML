"""Session mixin: data domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import data_ops
from buildml.session.mixins._shared import *  # noqa: F403


class DataSessionMixin:
    """Public Session methods for the data domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _ingest_report: Any
        _reattach_result: Any
        _split_plan: Any

    def close_native(self) -> None:
        """Close the DuckDB connection this session owns, if it has one.

        Session facade over :func:`buildml.session.data_ops.close_native`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        See Also
        --------
        :func:`buildml.session.data_ops.close_native`
            Canonical documentation for parameters, raises, and examples.
        """
        return data_ops.close_native(self)

    @classmethod
    def ingest(
        cls,
        source: pd.DataFrame | str | Path,
        *,
        mode: DataMode | str | None = None,
        engine: EngineName | str | None = None,
        dry_run: bool = False,
        mock_byte_estimate: int | None = None,
        read_nrows: int | None = None,
    ) -> Session:
        """Create a session by loading a table, and inspect it while loading.

        Session facade over :func:`buildml.session.data_ops.ingest_session`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            A new session. It carries a dataset unless ``dry_run=True`` (or a

        See Also
        --------
        :func:`buildml.session.data_ops.ingest_session`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.ingest_session(
            cls,
            source=source,
            mode=mode,
            engine=engine,
            dry_run=dry_run,
            mock_byte_estimate=mock_byte_estimate,
            read_nrows=read_nrows,
        ))

    @property
    def ingest_report(self) -> IngestReport | None:
        """What the loader found when reading the source.

        Session-held result for ``ingest_report``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("IngestReport | None", self._ingest_report)

    @property
    def split_plan(self) -> SplitPlan | None:
        """Which rows belong to train, validation, and test.

        Session-held result for ``split_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SplitPlan | None", self._split_plan)

    @property
    def reattach_result(self) -> ReattachResult | None:
        """Whether restored checkpoint data still matched what was expected.

        Session-held result for ``reattach_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ReattachResult | None", self._reattach_result)

    def set_roles(self, mapping: dict[str, str | ColumnRole]) -> Session:
        """Declare what each column means, so later steps can act on it.

        Session facade over :func:`buildml.session.data_ops.set_roles`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        See Also
        --------
        :func:`buildml.session.data_ops.set_roles`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.set_roles(self, mapping=mapping))

    def split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        random_state: int | None = 42,
        stratify: bool = False,
    ) -> Session:
        """Randomly hold back rows so you can measure honest performance.

        Session facade over :func:`buildml.session.data_ops.split`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        See Also
        --------
        :func:`buildml.session.data_ops.split`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.split(
            self,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        ))

    def inject_split(
        self,
        *,
        train_indices: list[int] | tuple[int, ...],
        test_indices: list[int] | tuple[int, ...],
        validation_indices: list[int] | tuple[int, ...] | None = None,
    ) -> Session:
        """Adopt a split that was decided outside BuildML.

        Session facade over :func:`buildml.session.data_ops.inject_split`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        See Also
        --------
        :func:`buildml.session.data_ops.inject_split`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.inject_split(
            self,
            train_indices=train_indices,
            test_indices=test_indices,
            validation_indices=validation_indices,
        ))

    def group_split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        random_state: int | None = 42,
        group_column: str | None = None,
    ) -> Session:
        """Split by entity, so no customer appears on both sides.

        Session facade over :func:`buildml.session.data_ops.group_split`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        See Also
        --------
        :func:`buildml.session.data_ops.group_split`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.group_split(
            self,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            group_column=group_column,
        ))

    def time_split(
        self,
        *,
        test_size: float | int = 0.2,
        validation_size: float | int | None = None,
        time_column: str | None = None,
    ) -> Session:
        """Train on the past and test on the future, as deployment will.

        Session facade over :func:`buildml.session.data_ops.time_split`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into preprocessing.

        See Also
        --------
        :func:`buildml.session.data_ops.time_split`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.time_split(
            self, test_size=test_size, validation_size=validation_size, time_column=time_column
        ))

    def partition(
        self,
        name: PartitionName | Literal["train", "validation", "test"],
    ) -> pd.DataFrame:
        """Pull out the rows belonging to one partition, as a DataFrame.

        Session facade over :func:`buildml.session.data_ops.partition`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pandas.DataFrame
            A copy of those rows with all current columns, reflecting every

        See Also
        --------
        :func:`buildml.session.data_ops.partition`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("pd.DataFrame", data_ops.partition(self, name=name))

    def assert_can_fit(self, partition: PartitionName = "train") -> Session:
        """Refuse to continue unless fitting is confined to the train rows.

        Session facade over :func:`buildml.session.data_ops.assert_can_fit`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so the guard sits inline in a chain.

        See Also
        --------
        :func:`buildml.session.data_ops.assert_can_fit`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.assert_can_fit(self, partition=partition))

    def to_engine(self, engine: EngineName | str | None = None) -> Any:
        """Hand back the data as the chosen engine's own object.

        Session facade over :func:`buildml.session.data_ops.to_engine`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        object
            A ``pandas.DataFrame``, ``polars.DataFrame``, or DuckDB relation,

        See Also
        --------
        :func:`buildml.session.data_ops.to_engine`
            Canonical documentation for parameters, raises, and examples.
        """
        return data_ops.to_engine(self, engine=engine)

    def checkpoint_save(
        self,
        path: str | Path,
        *,
        sidecar_partition_rows: int | None = None,
        sidecar_compression: str | None = None,
        sidecar_layout: str | None = None,
    ) -> Path:
        """Save the whole session so you can stop and pick up where you left off.

        Session facade over :func:`buildml.session.data_ops.checkpoint_save`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            The checkpoint directory that was written.

        See Also
        --------
        :func:`buildml.session.data_ops.checkpoint_save`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", data_ops.checkpoint_save(
            self,
            path=path,
            sidecar_partition_rows=sidecar_partition_rows,
            sidecar_compression=sidecar_compression,
            sidecar_layout=sidecar_layout,
        ))

    @classmethod
    def checkpoint_load(cls, path: str | Path, *, data_only: bool = False, trusted: bool = False) -> Session:
        """Restore a saved session and check the data still matches.

        Session facade over :func:`buildml.session.data_ops.checkpoint_load_session`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            A new session holding the restored state.

        See Also
        --------
        :func:`buildml.session.data_ops.checkpoint_load_session`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.checkpoint_load_session(cls, path=path, data_only=data_only, trusted=trusted))

    def reattach(
        self, path: str | Path, *, data_only: bool = False, trusted: bool = False
    ) -> Session:
        """Replace this session's state from a checkpoint, in place.

        Session facade over :func:`buildml.session.data_ops.reattach`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, now holding the restored state.

        See Also
        --------
        :func:`buildml.session.data_ops.reattach`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.reattach(
            self, path=path, data_only=data_only, trusted=trusted
        ))

    def to_pandas(self) -> pd.DataFrame:
        """Take the data out as a plain Pandas DataFrame.

        Session facade over :func:`buildml.session.data_ops.to_pandas`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pandas.DataFrame
            A copy of the current data with every transform applied so far.

        See Also
        --------
        :func:`buildml.session.data_ops.to_pandas`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("pd.DataFrame", data_ops.to_pandas(self))

    def to_parquet(self, path: str | Path) -> Path:
        """Write the current data to a Parquet file.

        Session facade over :func:`buildml.session.data_ops.to_parquet`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Where the file was written.

        See Also
        --------
        :func:`buildml.session.data_ops.to_parquet`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", data_ops.to_parquet(self, path=path))

    def head(self, n: int = 5) -> pd.DataFrame:
        """Look at the first few rows.

        Session facade over :func:`buildml.session.data_ops.head`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows with all current columns.

        See Also
        --------
        :func:`buildml.session.data_ops.head`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("pd.DataFrame", data_ops.head(self, n=n))

    def with_mode(self, mode: DataMode | str) -> Session:
        """Set whether data is held in memory or kept lazy.

        Session facade over :func:`buildml.session.data_ops.with_mode`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains.

        See Also
        --------
        :func:`buildml.session.data_ops.with_mode`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.with_mode(self, mode=mode))

    def with_engine(self, engine: EngineName | str) -> Session:
        """Switch the compute engine backing the data.

        Session facade over :func:`buildml.session.data_ops.with_engine`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains.

        See Also
        --------
        :func:`buildml.session.data_ops.with_engine`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.with_engine(self, engine=engine))

    def sync_native(self) -> Session:
        """Rebuild the engine's table from the current Pandas frame.

        Session facade over :func:`buildml.session.data_ops.sync_native`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains.

        See Also
        --------
        :func:`buildml.session.data_ops.sync_native`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", data_ops.sync_native(self))

    def metadata(self) -> dict[str, Any]:
        """Take a serialisable snapshot of everything the session knows.

        Session facade over :func:`buildml.session.data_ops.metadata`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        dict
            Whether a dataset is attached, the ingest report, the split plan,

        See Also
        --------
        :func:`buildml.session.data_ops.metadata`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", data_ops.metadata(self))
