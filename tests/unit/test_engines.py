import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.core.types import EngineName
from buildml.data.engines import engine_roundtrip_pandas, get_engine
from buildml.ingest.detect import available_engines


def test_pandas_engine_roundtrip() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = engine_roundtrip_pandas(frame, "pandas")
    pd.testing.assert_frame_equal(out.reset_index(drop=True), frame)


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_engine_roundtrip() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    engine = get_engine("polars")
    native = engine.from_pandas(frame)
    assert engine.n_rows(native) == 3
    assert engine.columns(native) == ["a", "b"]
    out = engine.to_pandas(native)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), frame)


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_engine_roundtrip() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    engine = get_engine("duckdb")
    native = engine.from_pandas(frame)
    assert engine.n_rows(native) == 3
    assert engine.columns(native) == ["a", "b"]
    out = engine.to_pandas(native)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), frame)


def test_session_with_engine_missing_polars() -> None:
    if EngineName.POLARS in available_engines():
        pytest.skip("polars installed")
    session = Session.ingest(pd.DataFrame({"a": [1]}))
    with pytest.raises(MissingExtraError, match="buildml\\[polars\\]"):
        session.with_engine("polars")
