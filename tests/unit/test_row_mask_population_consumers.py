"""Positional boolean masks preserve native populations and fitted consumers."""

import hashlib
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.engines import get_engine
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline

BACKINGS = [
    "pandas-memory",
    "pandas-parquet",
    "polars-cold",
    "polars-warm",
    "duckdb-cold",
    "duckdb-warm",
]
BAD = ["False", 1, 0.0, None, np.nan, pd.NA]
BAD_IDS = ["string", "integer", "float", "none", "nan", "nullable-missing"]
ROLES = {"row_id": "id", "x": "feature", "z": "feature", "target": "target"}


def _frame():
    rng = np.random.default_rng(811)
    x, z = rng.normal(size=(2, 36))
    return pd.DataFrame(
        {
            "row_id": np.arange(36),
            "x": x,
            "z": z,
            "target": 2.3 * x - 0.6 * z + 0.1 * np.sin(np.arange(36)),
        },
        index=np.repeat([9, 3, 7, 1], 9),
    )


@contextmanager
def _population(backing, directory):
    engine, state = backing.split("-")
    if engine != "pandas":
        pytest.importorskip(engine)
    raw = _frame()
    path = directory / "owner's population.parquet"
    raw.to_parquet(path, index=False)
    source = raw if state == "memory" else path
    owner = Session.ingest(source, engine=engine, mode="memory" if engine == "pandas" else "lazy")
    owner.set_roles(ROLES)
    if state == "warm":
        owner.to_pandas()
    dataset = owner.dataset
    # Native materialization here does not populate Dataset's Pandas cache.
    expected = (
        get_engine(engine).to_pandas(dataset.native)
        if dataset.has_native
        else dataset.frame.copy(deep=True)
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        yield owner, dataset, expected, path, digest
    finally:
        owner.close_native()


def _mask(pattern, n, form):
    values = (
        [True] * n
        if pattern == "all"
        else [False] * n
        if pattern == "none"
        else [i % 3 != 1 for i in range(n)]
    )
    if form == "numpy":
        return np.array(values, dtype=np.bool_)
    if form == "nullable-series":
        # The contract is positional; these labels deliberately disagree with the data index.
        return pd.Series(values, index=np.arange(n)[::-1] + 1000, dtype="boolean")
    return values


def _check_owner(dataset, expected, path, digest, native, roles):
    assert dataset.native is native
    assert dataset.roles == roles
    pd.testing.assert_frame_equal(
        dataset.to_pandas().reset_index(drop=True), expected.reset_index(drop=True)
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("pattern", ["all", "none", "subset"])
@pytest.mark.parametrize("form", ["numpy", "nullable-series"])
def test_valid_boolean_populations_survive_cache_modes_export_and_chaining(
    backing, materialize, pattern, form, tmp_path
):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        native, roles = source.native, dict(source.roles)
        mask = _mask(pattern, len(frame), form)
        values = list(mask)
        positions = [i for i in range(len(frame)) if values[i]]
        expected = frame.iloc[positions]
        selected = source.filter_rows(mask, materialize=materialize)
        chained = None
        try:
            pd.testing.assert_frame_equal(
                selected.to_pandas().reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=not expected.empty,
            )
            assert selected.n_rows == len(positions)
            assert selected.roles == roles
            if backing.startswith("pandas"):
                pd.testing.assert_index_equal(selected.to_pandas().index, expected.index)
            if backing.startswith("duckdb") and not materialize:
                assert selected.native.connection is native.connection
            second_mask = [i % 2 == 0 for i in range(len(expected))]
            chained = selected.filter_rows(second_mask)
            remaining = [i for i in range(len(expected)) if i % 2 == 0]
            wanted = expected.iloc[remaining]
            pd.testing.assert_frame_equal(
                chained.to_pandas().reset_index(drop=True),
                wanted.reset_index(drop=True),
                check_dtype=not wanted.empty,
            )
            exported = chained.to_parquet(tmp_path / "filtered.parquet")
            pd.testing.assert_frame_equal(
                pd.read_parquet(exported).reset_index(drop=True),
                wanted.reset_index(drop=True),
                check_dtype=not wanted.empty,
            )
            assert list(mask) == values
        finally:
            if chained is not None:
                chained.close_native()
            selected.close_native()
        _check_owner(source, frame, path, digest, native, roles)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("bad", BAD, ids=BAD_IDS)
def test_nonboolean_masks_refuse_before_cache_or_native_population_changes(
    backing, materialize, bad, tmp_path
):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        native, roles = source.native, dict(source.roles)
        cached = source.frame.copy(deep=True)
        stale = source.pandas_stale
        mask = [True] * len(frame)
        mask[7] = bad
        with pytest.raises(ValidationError, match="boolean"):
            source.filter_rows(mask, materialize=materialize)
        assert source.pandas_stale is stale
        pd.testing.assert_frame_equal(source.frame, cached)
        _check_owner(source, frame, path, digest, native, roles)
        # A rejected request must leave this same source usable for a valid one.
        reused = source.filter_rows([i == 7 for i in range(len(frame))])
        try:
            assert reused.to_pandas().row_id.tolist() == [frame.row_id.iloc[7]]
        finally:
            reused.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("offset", [-1, 1])
def test_wrong_length_masks_preserve_source_and_refuse_row_misalignment(
    backing, materialize, offset, tmp_path
):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        native, roles = source.native, dict(source.roles)
        with pytest.raises(ValidationError, match="length"):
            source.filter_rows([True] * (len(frame) + offset), materialize=materialize)
        _check_owner(source, frame, path, digest, native, roles)


def _expected_prediction(train, query):
    x, y = train[["x", "z"]].to_numpy(), train.target.to_numpy()
    mean_x, mean_y = x.mean(axis=0), y.mean()
    centered = x - mean_x
    beta = np.linalg.solve(centered.T @ centered + 0.75 * np.eye(2), centered.T @ (y - mean_y))
    return (query[["x", "z"]].to_numpy() - mean_x) @ beta + mean_y


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("materialize", [False, True])
def test_selected_native_population_drives_saved_ridge_and_survives_bad_mask(
    backing, materialize, tmp_path
):
    with _population(backing, tmp_path) as (_, source, frame, path, digest):
        mask = [i % 3 != 1 for i in range(len(frame))]
        positions = [i for i in range(len(frame)) if i % 3 != 1]
        expected = frame.iloc[positions].reset_index(drop=True)
        selected = source.filter_rows(mask, materialize=materialize)
        fitted = Session(dataset=selected)
        try:
            n = len(expected)
            fitted.inject_split(
                train_indices=list(range(n - 6)),
                validation_indices=[n - 6, n - 5],
                test_indices=list(range(n - 4, n)),
            )
            fitted.classical.fit(Ridge(alpha=0.75, solver="svd"), task="regression")
            destination = fitted.save_pipeline(tmp_path / "ridge", evaluate_partition=None)
            bundle = load_pipeline_bundle(destination, trusted=True)
            model_hash = joblib.hash(bundle)
            files = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.iterdir()
                if p.is_file()
            }
            query = pd.DataFrame({"x": [-2.0, 0.5, 2.0], "z": [0.2, -1.0, 0.7]}, index=[9, 9, 1])
            prediction = _expected_prediction(expected.iloc[: n - 6], query)
            for artifact in (destination, bundle):
                result = predict_from_pipeline(artifact, query, trusted=True)
                np.testing.assert_allclose(result.predictions, prediction, atol=1e-11, rtol=1e-11)
                pd.testing.assert_index_equal(result.predictions.index, query.index)
            invalid = [True] * selected.n_rows
            invalid[3] = "False"
            with pytest.raises(ValidationError, match="boolean"):
                selected.filter_rows(invalid, materialize=materialize)
            assert joblib.hash(bundle) == model_hash
            assert files == {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.iterdir()
                if p.is_file()
            }
            np.testing.assert_allclose(
                predict_from_pipeline(bundle, query, trusted=True).predictions,
                prediction,
                atol=1e-11,
            )
            assert source.n_rows == len(frame)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        finally:
            fitted.close_native()


@contextmanager
def _native(engine):
    if engine != "pandas":
        pytest.importorskip(engine)
    adapter = get_engine(engine)
    frame = _frame().reset_index(drop=True)
    native = adapter.from_pandas(frame)
    try:
        yield adapter, native, frame
    finally:
        if engine == "duckdb":
            native.connection.close()


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("bad", BAD, ids=BAD_IDS)
def test_direct_adapters_refuse_nonboolean_mask_values(engine, bad):
    with _native(engine) as (adapter, native, frame):
        mask = [True] * len(frame)
        mask[7] = bad
        with pytest.raises(ValueError, match="boolean"):
            adapter.filter_rows(native, mask)
        pd.testing.assert_frame_equal(adapter.to_pandas(native), frame)


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("offset", [-1, 1])
def test_direct_adapter_length_contract_refuses_misaligned_masks(engine, offset):
    with _native(engine) as (adapter, native, frame):
        with pytest.raises(ValueError, match="length"):
            adapter.filter_rows(native, [True] * (len(frame) + offset))
        pd.testing.assert_frame_equal(adapter.to_pandas(native), frame)


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("pattern", ["all", "none", "subset", "numpy-values"])
def test_direct_adapter_valid_boolean_positions_remain_exact(engine, pattern):
    with _native(engine) as (adapter, native, frame):
        values = [i % 3 != 1 for i in range(len(frame))]
        mask = (
            tuple(np.bool_(v) for v in values)
            if pattern == "numpy-values"
            else tuple(_mask(pattern, len(frame), "list"))
        )
        expected = frame.iloc[[i for i in range(len(frame)) if mask[i]]].reset_index(drop=True)
        selected = adapter.filter_rows(native, mask)
        pd.testing.assert_frame_equal(
            adapter.to_pandas(selected).reset_index(drop=True),
            expected,
            check_dtype=not expected.empty,
        )
        pd.testing.assert_frame_equal(adapter.to_pandas(native), frame)
