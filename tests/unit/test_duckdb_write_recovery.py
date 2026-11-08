"""Real fallback writes preserve data and release borrowed connection state."""
import pandas as pd
import pytest

from buildml.data.engines.duckdb_engine import DuckDBEngine, DuckDBTable, close_duckdb_connection


class _ArrowOnlyRelation:
    def __init__(self, relation):
        self.relation = relation

    def __getattr__(self, name):
        if name == 'write_parquet':
            raise AttributeError(name)
        return getattr(self.relation, name)


class _FailingCopyConnection:
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, query):
        if query.startswith('COPY '):
            raise OSError('Injected SQL COPY failure')
        return self.connection.execute(query)


@pytest.mark.parametrize('fail_copy', [False, True])
def test_fallback_write_preserves_user_registration_and_cleans_temporary_source(fail_copy, tmp_path):
    duckdb = pytest.importorskip('duckdb')
    pytest.importorskip('pyarrow')
    connection = duckdb.connect()
    engine = DuckDBEngine()
    frame = pd.DataFrame({'x': [1., 3., 5.], 'label': ['a', 'b', 'c']})
    try:
        connection.register('_bml_write_src', pd.DataFrame({'sentinel': [731]}))
        table = engine.from_pandas(frame, connection=connection)
        before = connection.execute('SHOW TABLES').fetchall()
        handle = DuckDBTable(_ArrowOnlyRelation(table.relation),
                             _FailingCopyConnection(connection) if fail_copy else connection)
        path = engine.write_parquet(handle, tmp_path / "user's export.parquet", row_group_size=2048)
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
        assert connection.execute('SHOW TABLES').fetchall() == before
        assert connection.execute('SELECT sentinel FROM _bml_write_src').fetchone() == (731,)
        assert engine._last_pushdown == ('arrow_write_parquet' if fail_copy else 'copy_parquet')
    finally:
        close_duckdb_connection(connection)
