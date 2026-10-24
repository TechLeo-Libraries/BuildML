"""Real DuckDB must preserve quoted paths and dataframe column identities."""
import pandas as pd
import pytest

from buildml.data.engines.duckdb_engine import DuckDBEngine, close_duckdb_connection


@pytest.fixture
def relation():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    engine = DuckDBEngine()
    frame = pd.DataFrame({'group"key': ["a", "a", "b", "b"], 'value"測定': [1., 3., 5., 7.]})
    table = engine.from_pandas(frame)
    try:
        yield engine, table, frame
    finally:
        close_duckdb_connection(table.connection)


def test_projection_preserves_embedded_quotes(relation):
    engine, table, frame = relation
    result = engine.select_columns(table, ['value"測定'])
    pd.testing.assert_frame_equal(engine.to_pandas(result), frame[['value"測定']])
    assert result.connection is table.connection


def test_seeded_sampling_preserves_quoted_names_and_is_repeatable(relation):
    engine, table, frame = relation
    first = engine.to_pandas(engine.sample_rows(table, 2, random_state=7))
    second = engine.to_pandas(engine.sample_rows(table, 2, random_state=7))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2
    assert set(first['value"測定']).issubset(frame['value"測定'])


def test_grouped_aggregation_preserves_quoted_keys_and_known_means(relation):
    engine, table, frame = relation
    result = engine.aggregate(table, {'value"測定': "mean"}, by=['group"key'])
    actual = engine.to_pandas(result).sort_values('group"key').reset_index(drop=True)
    assert actual['group"key'].tolist() == ["a", "b"]
    assert actual['value"測定_mean'].tolist() == [2., 6.]


def test_parquet_directory_with_apostrophe_retains_all_shards(relation, tmp_path):
    engine, table, frame = relation
    directory = tmp_path / "customer's data"
    directory.mkdir()
    frame.iloc[:2].to_parquet(directory / "one.parquet", index=False)
    frame.iloc[2:].to_parquet(directory / "two.parquet", index=False)
    restored = engine.from_parquet(directory, connection=table.connection)
    actual = engine.to_pandas(restored).sort_values('value"測定').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, frame)
