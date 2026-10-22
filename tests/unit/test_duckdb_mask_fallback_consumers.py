"""Real DuckDB consumers agree with positional selection across bridges."""
import builtins
import dis
import inspect

import numpy as np
import pandas as pd
import pytest

from buildml.data.engines import get_engine

pytest.importorskip("duckdb")


@pytest.mark.parametrize("without_arrow", [False, True])
@pytest.mark.parametrize("mask", [
    [True, False, True, False],
    (False, True, False, True),
    [np.bool_(True), np.bool_(False), np.bool_(False), np.bool_(True)],
])
def test_mask_bridge_matches_independent_positions(monkeypatch, without_arrow, mask):
    engine = get_engine("duckdb")
    frame = pd.DataFrame({"_bml_rn": [90, 10, 30, 70], "text": ["a", None, "c", "d"]})
    source = engine.from_pandas(frame)
    try:
        if without_arrow:
            original_import = builtins.__import__

            def import_without_adapter_arrow(name, globals=None, *args, **kwargs):
                caller = inspect.currentframe().f_back
                direct_import = dis.opname[caller.f_code.co_code[caller.f_lasti]] == "IMPORT_NAME"
                if name == "pyarrow" and direct_import and (globals or {}).get("__name__") == "buildml.data.engines.duckdb_engine":
                    raise ImportError("Injected adapter Arrow import failure")
                return original_import(name, globals, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", import_without_adapter_arrow)
        selected = engine.filter_rows(source, mask)
        expected = frame.iloc[[i for i, keep in enumerate(mask) if keep]].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected.relation.df(), expected)
        assert engine._last_pushdown == ("pandas_filter_fallback" if without_arrow else "sql_mask")
        pd.testing.assert_frame_equal(source.relation.df(), frame)
        # Chaining must apply positions to the selected relation, not the source.
        chained = engine.filter_rows(selected, [False, True])
        pd.testing.assert_frame_equal(chained.relation.df(), expected.iloc[[1]].reset_index(drop=True))
    finally:
        source.connection.close()


@pytest.mark.parametrize("bad", ["False", 1, 0.0, None, float("nan")])
def test_non_boolean_mask_rejected_before_selection(bad):
    engine = get_engine("duckdb")
    source = engine.from_pandas(pd.DataFrame({"x": [10, 20]}))
    try:
        with pytest.raises(ValueError, match="boolean"):
            engine.filter_rows(source, [True, bad])
    finally:
        source.connection.close()
