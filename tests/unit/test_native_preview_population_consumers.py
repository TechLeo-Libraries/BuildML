"""Native previews preserve positional prefixes and downstream fitted populations."""

import hashlib
import math
from contextlib import contextmanager
from decimal import Decimal

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.engines import get_engine
from buildml.data.engines.duckdb_engine import DuckDBTable
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline

BACKINGS = [
    "pandas-memory",
    "pandas-parquet",
    "polars-eager",
    "polars-lazy",
    "polars-warm",
    "duckdb-arrow",
    "duckdb-parquet",
    "duckdb-warm",
]
COUNTS = [0, 5, np.int64(7), -3, np.int64(-36), 10**80, -(10**80)]
COUNT_IDS = ["empty", "prefix", "numpy", "drop-tail", "drop-all", "huge", "huge-negative"]
BAD = [True, np.bool_(False), 2.5, "2", None, np.nan, 1 + 0j]
BAD_IDS = ["bool", "numpy-bool", "fraction", "text", "none", "nan", "complex"]
ROLES = {
    "row_id": "id",
    "x": "feature",
    "z": "feature",
    "target": "target",
    "group": "ignore",
    "money": "ignore",
    'text"測定': "ignore",
}


def _frame():
    rng = np.random.default_rng(929)
    x, z = rng.normal(size=(2, 36))
    return pd.DataFrame(
        {
            "row_id": np.arange(36),
            "x": x,
            "z": z,
            "target": 1.7 * x - 0.4 * z + 0.1 * np.cos(np.arange(36)),
            "group": np.arange(36) % 4,
            "money": [Decimal(i) / 100 if i % 5 else None for i in range(36)],
            'text"測定': [f"row '{i}'" for i in range(36)],
        },
        index=np.repeat([8, 2, 9, 1], 9),
    )


def _ordered_duckdb(table):
    # The documented forwarded relation method drops the wrapper; retain its owner.
    return DuckDBTable(table.order('"row_id"'), table.connection)


@contextmanager
def _population(backing, directory):
    engine, state = backing.split("-")
    if engine != "pandas":
        pytest.importorskip(engine)
    frame = _frame()
    path = directory / "source's population.parquet"
    frame.to_parquet(path, index=False)
    if backing == "duckdb-arrow":
        import pyarrow as pa

        native = get_engine(engine).from_arrow(pa.Table.from_pandas(frame, preserve_index=False))
        dataset = Dataset.from_native(_ordered_duckdb(native), engine=engine, source="arrow")
        owner = Session(dataset=dataset)
    else:
        source = frame if state in {"memory", "eager"} else path
        owner = Session.ingest(
            source,
            engine=engine,
            mode="memory" if engine == "pandas" or state == "eager" else "lazy",
        )
        dataset = owner.dataset
        if engine == "duckdb":
            dataset.native = _ordered_duckdb(dataset.native)
    owner.set_roles(ROLES)
    if state == "warm":
        owner.to_pandas()
    if backing != "pandas-memory":
        frame = frame.reset_index(drop=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        yield owner, dataset, frame, path, digest
    finally:
        owner.close_native()


def _positions(n, size):
    end = min(int(n), size) if n >= 0 else max(0, size + int(n))
    return list(range(end))


def _same(actual, expected):
    # Arrow/native providers normalize decimal/null dtypes; values and columns must agree.
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )


def _derive(source, frame, workflow):
    if workflow == "identity":
        return source, frame
    if workflow == "project":
        columns = ["row_id", "x", "target", 'text"測定', "money"]
        return source.project(columns), frame[columns]
    if workflow == "filter":
        mask = [i % 3 != 1 for i in range(len(frame))]
        return source.filter_rows(mask), frame.iloc[[i for i in range(len(frame)) if i % 3 != 1]]
    derived = source.aggregate({"x": ["sum", "mean"], "row_id": ["count"]}, by=["group"])
    records = []
    for group in range(4):
        positions = [i for i in range(len(frame)) if i % 4 == group]
        total = math.fsum(float(frame.x.iloc[i]) for i in positions)
        records.append(
            {
                "group": group,
                "x_sum": total,
                "x_mean": total / len(positions),
                "row_id_count": len(positions),
            }
        )
    # Group order is otherwise unspecified; establish it before asking for a prefix.
    if derived.has_native:
        if str(derived.engine.value) == "duckdb":
            table = derived.native
            derived.native = DuckDBTable(table.order('"group"'), table.connection)
        else:
            derived.native = derived.native.sort("group")
    else:
        derived.frame = derived.frame.sort_values("group").reset_index(drop=True)
    return derived, pd.DataFrame(records)


def _check_owner(source, frame, path, digest, native, roles):
    assert source.native is native
    assert source.roles == roles
    actual = get_engine(source.engine).to_pandas(native) if source.has_native else source.frame
    _same(actual, frame)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("workflow", ["identity", "project", "filter", "aggregate"])
@pytest.mark.parametrize("n", COUNTS, ids=COUNT_IDS)
def test_preview_prefixes_match_independent_positions_after_native_workflows(
    backing, workflow, n, tmp_path
):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        native, roles = source.native, dict(source.roles)
        derived, expected = _derive(source, frame, workflow)
        cached, stale = derived.frame.copy(deep=True), derived.pandas_stale
        handle, derived_roles = derived.native, dict(derived.roles)
        try:
            selected = derived.head(n)
            wanted = expected.iloc[_positions(n, len(expected))]
            _same(selected, wanted)
            assert len(selected) == len(_positions(n, len(expected)))
            assert list(selected.columns) == list(wanted.columns)
            pd.testing.assert_frame_equal(derived.frame, cached)
            assert derived.pandas_stale is stale
            assert derived.native is handle and derived.roles == derived_roles
            if backing.startswith("pandas"):
                pd.testing.assert_index_equal(selected.index, wanted.index)
            destination = tmp_path / "preview.parquet"
            selected.to_parquet(destination, index=False)
            _same(pd.read_parquet(destination), wanted)
            # A returned preview must not be a writable view into the source/cache.
            if len(selected):
                selected.iloc[0, 0] = -901
            _same(derived.head(n), wanted)
        finally:
            if derived is not source:
                derived.close_native()
        _check_owner(source, frame, path, digest, native, roles)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("n", BAD, ids=BAD_IDS)
def test_invalid_preview_counts_refuse_before_native_or_cache_changes(backing, n, tmp_path):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        cached, stale = source.frame.copy(deep=True), source.pandas_stale
        native, roles = source.native, dict(source.roles)
        with pytest.raises(ValidationError, match="integer"):
            source.head(n)
        pd.testing.assert_frame_equal(source.frame, cached)
        assert source.pandas_stale is stale
        _same(source.head(3), frame.iloc[[0, 1, 2]])
        _check_owner(source, frame, path, digest, native, roles)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("n", [0, 7, -3])
def test_empty_native_population_keeps_preview_schema(backing, n, tmp_path):
    with _population(backing, tmp_path) as (_, source, frame, _, _):
        empty = source.filter_rows([False] * len(frame))
        try:
            result = empty.head(n)
            _same(result, frame.iloc[[]])
            assert result.empty and result.columns.tolist() == frame.columns.tolist()
            assert source.n_rows == len(frame)
        finally:
            empty.close_native()


def _prediction(train, query):
    x, y = train[["x", "z"]].to_numpy(), train.target.to_numpy()
    center, mean_y = x.mean(axis=0), y.mean()
    centered = x - center
    beta = np.linalg.solve(centered.T @ centered + 0.6 * np.eye(2), centered.T @ (y - mean_y))
    return (query[["x", "z"]].to_numpy() - center) @ beta + mean_y


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("n", [24, np.int64(-12)], ids=["prefix", "omit-tail"])
def test_preview_training_and_saved_ridge_match_independent_prefix_design(backing, n, tmp_path):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        native, roles = source.native, dict(source.roles)
        wanted = frame.iloc[list(range(24))].reset_index(drop=True)
        observed = source.head(n)
        train_end = len(observed) - 6
        fitted = (
            Session.ingest(observed)
            .set_roles(ROLES)
            .inject_split(
                train_indices=list(range(train_end)),
                validation_indices=[train_end, train_end + 1],
                test_indices=list(range(train_end + 2, len(observed))),
            )
        )
        fitted.classical.fit(Ridge(alpha=0.6, solver="svd"), task="regression")
        query = pd.DataFrame({"x": [-1.0, 0.2, 2.5], "z": [0.5, -0.7, 1.1]}, index=[9, 9, 2])
        expected = _prediction(wanted.iloc[:18], query)
        destination = fitted.save_pipeline(tmp_path / "preview-ridge", evaluate_partition=None)
        bundle = load_pipeline_bundle(destination, trusted=True)
        checksum = joblib.hash(bundle)
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.iterdir()
            if p.is_file()
        }
        for artifact in (destination, bundle):
            result = predict_from_pipeline(artifact, query, trusted=True)
            np.testing.assert_allclose(result.predictions, expected, rtol=1e-11, atol=1e-11)
            pd.testing.assert_index_equal(result.predictions.index, query.index)
        _same(observed, wanted)
        with pytest.raises(ValidationError, match="integer"):
            source.head(True)
        np.testing.assert_allclose(
            predict_from_pipeline(bundle, query, trusted=True).predictions,
            expected,
            rtol=1e-11,
            atol=1e-11,
        )
        assert joblib.hash(bundle) == checksum
        assert files == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.iterdir()
            if p.is_file()
        }
        _check_owner(source, frame, path, digest, native, roles)


@contextmanager
def _native(engine):
    if engine != "pandas":
        pytest.importorskip(engine)
    adapter, frame = get_engine(engine), _frame().reset_index(drop=True)
    if engine == "duckdb":
        import duckdb
        import pyarrow as pa

        con = duckdb.connect(":memory:")
        native = _ordered_duckdb(
            adapter.from_arrow(pa.Table.from_pandas(frame, preserve_index=False), connection=con)
        )
    else:
        native = adapter.from_pandas(frame)
    try:
        yield adapter, native, frame
    finally:
        if engine == "duckdb":
            con.close()


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize(
    "n", [0, 2, np.int64(3), -2, np.int64(-2), -36, -100, 100, 10**80, -(10**80)]
)
def test_direct_adapter_preview_values_match_independent_positions(engine, n):
    with _native(engine) as (adapter, native, frame):
        wanted = frame.iloc[_positions(n, len(frame))]
        result = adapter.head(native, n)
        _same(result, wanted)
        if len(result):
            result.iloc[0, 0] = -902
        _same(adapter.to_pandas(native), frame)


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("n", BAD, ids=BAD_IDS)
def test_direct_adapters_refuse_noninteger_preview_counts(engine, n):
    with _native(engine) as (adapter, native, frame):
        with pytest.raises(ValueError, match="integer"):
            adapter.head(native, n)
        _same(adapter.to_pandas(native), frame)
