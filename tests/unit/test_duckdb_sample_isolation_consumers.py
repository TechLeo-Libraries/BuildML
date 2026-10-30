"""Lazy random samples retain their source, shared views and saved-model consumers."""

import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.data.engines.duckdb_engine import DuckDBEngine, DuckDBTable, close_duckdb_connection
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
ORIGINS = ("pandas", "parquet", "sql-view")
ROW = 'row"id'


def _frame(offset):
    ids = np.arange(offset, offset+64)
    return pd.DataFrame({ROW: ids, "bucket": ids//2, "value": 2*ids+1,
        "x": ids/10., "z": np.sin(ids), "target": 3+2*ids/10.-.5*np.sin(ids)})


@pytest.fixture
def tables(request, tmp_path):
    engine, con = DuckDBEngine(), duckdb.connect(":memory:")
    origin = request.param
    frames = [_frame(0), _frame(1024)]
    handles = []
    try:
        for name, frame in zip(("left", "right"), frames, strict=True):
            if origin == "pandas":
                handle = engine.from_pandas(frame, connection=con)
            elif origin == "parquet":
                path = tmp_path / (name + "'s shard.parquet")
                frame.to_parquet(path, index=False)
                handle = engine.from_parquet(path, connection=con)
            else:
                con.register(name + "_input", frame)
                handle = DuckDBTable(con.sql("SELECT * FROM " + name + "_input"), con)
            handles.append(handle)
        yield engine, con, handles, frames
    finally:
        close_duckdb_connection(con)


def _transform(engine, table, frame, operation):
    if operation == "project":
        columns = [ROW, "value"]
        return engine.select_columns(table, columns), frame[columns], ROW
    if operation == "filter":
        return engine.filter_expr(table, '"value" % 4 = 1'), frame.loc[frame["value"] % 4 == 1], ROW
    if operation == "aggregate":
        expected = frame.groupby("bucket", as_index=False)["value"].agg(["sum", "mean"])
        expected = expected.rename(columns={"sum": "value_sum", "mean": "value_mean"})
        return engine.aggregate(table, {"value": ["sum", "mean"]}, by=["bucket"]), expected, "bucket"
    return table, frame, ROW


def _assert_subset(actual, expected, key, count):
    assert len(actual) == min(count, len(expected))
    assert actual[key].is_unique
    assert set(actual[key]) <= set(expected[key])
    matched = expected.set_index(key).loc[actual[key]].reset_index()[list(actual.columns)]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), matched, check_dtype=False)


@pytest.mark.parametrize("tables", ORIGINS, indirect=True)
@pytest.mark.parametrize("operation", ("identity", "project", "filter", "aggregate"))
@pytest.mark.parametrize("seed", (None, 17), ids=("unseeded", "seeded"))
@pytest.mark.parametrize("count", (1, 7, 100))
def test_sibling_lazy_samples_keep_disjoint_source_rows_and_values(tables, operation, seed, count):
    engine, con, handles, frames = tables
    left, expected_left, key = _transform(engine, handles[0], frames[0], operation)
    right, expected_right, _ = _transform(engine, handles[1], frames[1], operation)
    first = engine.sample_rows(left, count, random_state=seed)
    second = engine.sample_rows(right, count, random_state=seed)
    assert first.connection is second.connection is con
    # Do not demand identical random draws: source support and row content must hold.
    for _ in range(2):
        _assert_subset(engine.to_pandas(first), expected_left, key, count)
        _assert_subset(engine.to_pandas(second), expected_right, key, count)
    pd.testing.assert_frame_equal(engine.to_pandas(handles[0]), frames[0])
    pd.testing.assert_frame_equal(engine.to_pandas(handles[1]), frames[1])


@pytest.mark.parametrize("tables", ORIGINS, indirect=True)
@pytest.mark.parametrize("seed", (None, 7, 44), ids=("unseeded", "seed7", "seed44"))
@pytest.mark.parametrize("depth", (2, 4))
def test_nested_lazy_samples_remain_executable_after_unrelated_draws(tables, seed, depth):
    engine, _, handles, frames = tables
    current = handles[0]
    for level, count in enumerate((23, 13, 7, 3)[:depth]):
        engine.sample_rows(handles[1], 5)
        current = engine.sample_rows(current, count, random_state=None if seed is None else seed+level)
    _assert_subset(engine.to_pandas(current), frames[0], ROW, count)
    # The shared connection remains usable for subsequent deterministic queries.
    pd.testing.assert_frame_equal(engine.to_pandas(handles[1]), frames[1])


@pytest.mark.parametrize("tables", ORIGINS, indirect=True)
@pytest.mark.parametrize("seed", (None, 17), ids=("unseeded", "seeded"))
@pytest.mark.parametrize("count", (0, 1, 7))
def test_sampling_does_not_replace_a_user_registered_s_view(tables, seed, count):
    engine, con, handles, frames = tables
    sentinel = pd.DataFrame({"marker": [901, 902], "payload": ["left", "right"]})
    con.register("s", sentinel)
    selected = engine.sample_rows(handles[0], count, random_state=seed)
    pd.testing.assert_frame_equal(con.sql("SELECT * FROM s").df(), sentinel)
    _assert_subset(engine.to_pandas(selected), frames[0], ROW, count)


@pytest.mark.parametrize("tables", ORIGINS, indirect=True)
@pytest.mark.parametrize("seed", (None, 17), ids=("unseeded", "seeded"))
@pytest.mark.parametrize("nested", (False, True))
def test_sampled_training_and_saved_ridge_match_independent_linear_solves(tables, seed, nested, tmp_path):
    engine, _, handles, frames = tables
    selected = engine.sample_rows(handles[0], 24, random_state=seed)
    if nested:
        selected = engine.sample_rows(selected, 18, random_state=seed)
    engine.sample_rows(handles[1], 5)
    # Snapshot a single executed draw for fitting; unseeded plans may redraw later.
    frame = engine.to_pandas(selected)
    _assert_subset(frame, frames[0], ROW, 18 if nested else 24)
    train_end = len(frame)-6
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target",
        ROW: "ignore", "bucket": "ignore", "value": "ignore"}).inject_split(
        train_indices=list(range(train_end)), validation_indices=list(range(train_end, train_end+3)),
        test_indices=list(range(train_end+3, len(frame))))
    x, y = frame[["x", "z"]].to_numpy(), frame["target"].to_numpy()
    center, mean_y = x[:train_end].mean(axis=0), y[:train_end].mean()
    centered = x[:train_end]-center
    coefficient = np.linalg.solve(centered.T@centered+.7*np.eye(2), centered.T@(y[:train_end]-mean_y))
    expected = mean_y+(x[train_end+3:]-center)@coefficient
    session.fit(Ridge(alpha=.7), task="regression")
    np.testing.assert_allclose(session.predict(partition="test"), expected, rtol=1e-10, atol=1e-10)
    path = tmp_path / "trusted-ridge"
    session.save_pipeline(path, evaluate_partition=None)
    bundle = load_pipeline_bundle(path, trusted=True)
    weights = joblib.hash(bundle)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    source = session.partition("test").copy(deep=True)
    for stored in (path, bundle):
        result = predict_from_pipeline(stored, source, trusted=True)
        np.testing.assert_allclose(result.predictions, expected, rtol=1e-10, atol=1e-10)
        pd.testing.assert_index_equal(result.predictions.index, source.index)
    assert joblib.hash(bundle) == weights
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()} == files
