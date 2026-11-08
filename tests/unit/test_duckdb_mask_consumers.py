"""Real DuckDB positional-mask consumers preserve user columns and lazy plans."""
import pandas as pd
import pytest

from buildml.data.engines import get_engine

pytest.importorskip('duckdb')
pytest.importorskip('pyarrow')


@pytest.mark.parametrize('column', ['_bml_rn', '_BML_RN', 'row number', 'quote"column'])
@pytest.mark.parametrize('mask', [[True, False, True, False], [False]*4, [True]*4])
def test_mask_preserves_user_schema_and_positions(column, mask):
    engine = get_engine('duckdb')
    frame = pd.DataFrame({column: [91, 17, 44, 8], 'value': ['a', 'b', 'c', 'd']})
    source = engine.from_pandas(frame)
    selected = engine.filter_rows(source, mask)
    later = engine.filter_rows(source, [False, True, False, True])
    pd.testing.assert_frame_equal(engine.to_pandas(selected), frame.loc[mask].reset_index(drop=True))
    pd.testing.assert_frame_equal(engine.to_pandas(later), frame.iloc[[1, 3]].reset_index(drop=True))
    pd.testing.assert_frame_equal(engine.to_pandas(source), frame)
    assert engine._bridge == 'sql'
