"""Engine operations preserve row identity, nulls and stored schemas."""
import numpy as np
import pandas as pd
import pytest

from buildml.data.engines import get_engine


@pytest.mark.parametrize('name', ['pandas', 'polars', 'duckdb'])
def test_masks_and_parquet_roundtrip_preserve_source_rows(name, tmp_path):
    if name != 'pandas':
        pytest.importorskip(name)
    pytest.importorskip('pyarrow')
    engine = get_engine(name)
    frame = pd.DataFrame({'row_id': np.arange(37), 'amount': np.arange(37) * .5,
                          'label': ['quoted value' if i % 3 else None for i in range(37)]})
    native = engine.from_pandas(frame)
    try:
        first_mask = [i % 2 == 0 for i in range(37)]
        second_mask = [i % 3 == 0 for i in range(37)]
        first = engine.filter_rows(native, first_mask)
        second = engine.filter_rows(native, second_mask)
        for filtered, mask in [(second, second_mask), (first, first_mask)]:
            expected = frame.loc[mask].reset_index(drop=True)
            actual = engine.to_pandas(filtered).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected)
        destination = tmp_path / "nested's directory" / 'filtered.parquet'
        if name == "pandas":
            destination.parent.mkdir(parents=True)
            engine.to_pandas(first).to_parquet(destination, index=False)
        else:
            engine.write_parquet(first, destination)
        restored = pd.read_parquet(destination)
        pd.testing.assert_frame_equal(restored.reset_index(drop=True), frame.loc[first_mask].reset_index(drop=True))
        assert engine.n_rows(native) == len(frame)
        pd.testing.assert_frame_equal(engine.to_pandas(native).reset_index(drop=True), frame)
    finally:
        connection = getattr(native, 'connection', None)
        if connection is not None:
            connection.close()


@pytest.mark.parametrize('name', ['pandas', 'polars', 'duckdb'])
def test_empty_mask_returns_schema_preserving_empty_table(name):
    if name != 'pandas':
        pytest.importorskip(name)
    engine = get_engine(name)
    frame = pd.DataFrame({'row_id': [0, 1], 'amount': [1.25, 2.5]})
    native = engine.from_pandas(frame)
    try:
        result = engine.filter_rows(native, [False, False])
        pd.testing.assert_frame_equal(engine.to_pandas(result).reset_index(drop=True), frame.iloc[:0])
    finally:
        connection = getattr(native, 'connection', None)
        if connection is not None:
            connection.close()
