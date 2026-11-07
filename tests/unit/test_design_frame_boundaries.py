"""Projection and sampling disclosures match what reaches the materialization boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.core.types import EngineName
from buildml.data.dataset import Dataset
from buildml.data.engines import get_engine, prepare_design_frame


def _frame():
    data = {"x": np.arange(40), "y": 2 * np.arange(40)}
    data.update({f"unneeded_{i}": ["wide text payload"] * 40 for i in range(12)})
    return pd.DataFrame(data)


@pytest.mark.parametrize("engine_name", ["polars", "duckdb"])
def test_one_shot_native_prep_converts_only_requested_columns(engine_name, monkeypatch):
    pytest.importorskip(engine_name)
    frame = _frame()
    dataset = Dataset.from_pandas(frame, engine=EngineName(engine_name), attach_native=False)
    engine_type = type(get_engine(engine_name))
    original = engine_type.from_pandas
    converted = []

    def from_pandas(engine, narrow):
        converted.append(tuple(narrow.columns))
        return original(engine, narrow)

    monkeypatch.setattr(engine_type, "from_pandas", from_pandas)
    result = prepare_design_frame(dataset, ["y", "x", "x"], sample_rows=7, random_state=81)
    repeated = prepare_design_frame(dataset, ["y", "x"], sample_rows=7, random_state=81)
    assert converted == [("y", "x"), ("y", "x")]
    assert result.columns_requested == ("y", "x")
    assert result.columns_materialized == ("y", "x")
    assert result.n_rows_source == 40 and result.n_rows_materialized == 7
    assert result.sampled and not result.used_native_handle
    np.testing.assert_array_equal(result.frame.y, 2 * result.frame.x)
    assert set(result.frame.x).issubset(set(frame.x))
    pd.testing.assert_frame_equal(result.frame, repeated.frame)
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), frame)
    assert dataset.native is None
    assert any("one-shot conversion" in note for note in result.limitations)
    assert any("not enable out-of-core training" in note for note in result.limitations)
    assert any("changes the empirical distribution" in note for note in result.limitations)


@pytest.mark.parametrize("engine_name", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("sample_rows", [0, -1])
def test_design_prep_refuses_nonpositive_sample_budget(engine_name, sample_rows):
    if engine_name != "pandas":
        pytest.importorskip(engine_name)
    dataset = Dataset.from_pandas(_frame(), engine=EngineName(engine_name), attach_native=False)
    with pytest.raises(ValidationError, match="sample_rows must be >= 1"):
        prepare_design_frame(dataset, ["x"], sample_rows=sample_rows)
    assert dataset.n_rows == 40


@pytest.mark.parametrize("columns, message", [([], "at least one column"), (["absent"], "missing")])
def test_design_prep_refuses_an_empty_or_incompatible_feature_contract(columns, message):
    dataset = Dataset.from_pandas(_frame())
    with pytest.raises(ValidationError, match=message):
        prepare_design_frame(dataset, columns)


def test_design_prep_honors_the_explicit_hard_memory_gate():
    dataset = Dataset.from_pandas(_frame())
    with pytest.raises(ValidationError, match="hard"):
        prepare_design_frame(dataset, ["x", "y"], hard_limit_bytes=1)
    assert dataset.n_rows == 40


@pytest.mark.parametrize("engine_name, attached", [
    ("pandas", False), ("polars", False), ("polars", True),
    ("duckdb", False), ("duckdb", True),
])
def test_repeated_design_columns_are_materialized_once_in_requested_order(engine_name, attached):
    if engine_name != "pandas":
        pytest.importorskip(engine_name)
    dataset = Dataset.from_pandas(_frame(), engine=EngineName(engine_name), attach_native=attached)
    try:
        result = prepare_design_frame(dataset, ["y", "x", "y", "x"])
        assert result.columns_requested == ("y", "x")
        assert result.columns_materialized == ("y", "x")
        pd.testing.assert_frame_equal(result.frame.reset_index(drop=True), _frame()[["y", "x"]])
    finally:
        dataset.close_native()
