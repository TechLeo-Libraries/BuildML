"""Live native rows, rather than their Pandas cache, feed saved inference."""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.engines import get_engine
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline

BACKINGS = ["frame", "pandas", "polars-memory", "polars-lazy", "polars-view",
            "duckdb-memory", "duckdb-lazy", "duckdb-view"]


def _fitted(tmp_path, kind, preparation):
    rng = np.random.default_rng(917)
    values = rng.normal(size=(84, 2))
    signal = 1.4 * values[:, 0] - .6 * values[:, 1]
    target = signal + .07 * np.sin(np.arange(len(values)))
    if kind == "binary":
        target = np.where(signal > 0, "010", "001")
    elif kind == "multiclass":
        target = np.array(["001", "010", "100"])[np.digitize(signal, [-.65, .65])]
    frame = pd.DataFrame({"x": values[:, 0], "z": values[:, 1], "target": target})
    if preparation != "none":
        frame.loc[::9, "x"] = np.nan
    session = Session.ingest(frame).set_roles({"target": "target"}).split(
        test_size=.2, validation_size=.2, random_state=13)
    train = frame.iloc[list(session.split_plan.train_indices)][["x", "z"]].to_numpy()
    median, center, divisor = np.zeros(2), np.zeros(2), np.ones(2)
    if preparation != "none":
        median = np.nanmedian(train, axis=0)
        filled = np.where(np.isnan(train), median, train)
        center, divisor = filled.mean(axis=0), filled.std(axis=0)
        session.impute(columns=["x", "z"]).scale(columns=["x", "z"])
        np.testing.assert_allclose(list(session.impute_plan.statistics_.values()), median)
        np.testing.assert_allclose(session.scale_plan.scaler.mean_, center)
        np.testing.assert_allclose(session.scale_plan.scaler.scale_, divisor)
    session.fit(Ridge(alpha=.8) if kind == "regression" else LogisticRegression(C=.9),
                task="regression" if kind == "regression" else "classification")
    path = session.save_pipeline(tmp_path / "pipeline", evaluate_partition=None)
    return path, load_pipeline_bundle(path, trusted=True), (median, center, divisor)


def _incoming(preparation, constants):
    frame = pd.DataFrame({"row_id": [f"{i:04}" for i in range(9)],
                          "z": [2., -1., 3.4, .5, -4., 1., 8., 0., -2.],
                          "x": [-2., .3, 4., 1.5, -1., 0., 7., 2., -.3],
                          "extra": ["unused"] * 9}, index=["same", "same", "x", "z", "x", "a", "b", "c", "a"])
    if preparation != "none":
        frame.loc[frame.index == "same", "x"] = np.nan
    numeric = frame[["x", "z"]].to_numpy()
    median, center, divisor = constants
    transformed = (np.where(np.isnan(numeric), median, numeric) - center) / divisor
    if preparation == "prepared":
        frame[["x", "z"]] = transformed
    elif preparation == "raw":
        # Coercion must reach native consumers without changing their source values.
        frame["z"] = frame["z"].map(str)
    return frame, transformed


def _source(tmp_path, frame, backing):
    if backing == "frame":
        return frame, None, frame.copy(), np.arange(len(frame))
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith("lazy"):
        file = tmp_path / "incoming.parquet"
        frame.to_parquet(file, index=False)
        owner = Session.ingest(file, engine=engine, mode="lazy")
    else:
        owner = Session.ingest(frame, engine=engine)
    source = owner.dataset
    positions = np.arange(len(frame))
    expected = frame.copy()
    if backing.endswith("view"):
        mask = [i % 3 != 1 for i in range(len(frame))]
        positions = positions[np.array(mask, dtype=bool)]
        source = source.filter_rows(mask).project([c for c in ["extra", "z", "row_id", "x"] if c in frame])
        expected = frame.iloc[positions].loc[:, source.columns].reset_index(drop=True)
    elif backing.endswith("lazy"):
        expected = expected.reset_index(drop=True)
    return source, owner, expected, positions


def _assert_input(source, owner, expected, native):
    if isinstance(source, pd.DataFrame):
        pd.testing.assert_frame_equal(source, expected)
    else:
        assert source.native is native
        # Empty object columns have no scalar type; Arrow/DuckDB may infer NULL/INTEGER.
        pd.testing.assert_frame_equal(source.to_pandas(), expected,
                                      check_dtype=not (source.has_native and expected.empty))
        if source.has_native:
            actual = get_engine(source.engine).to_pandas(source.native)
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    if owner is not None:
        assert owner.fit_result is None and owner.impute_plan is None and owner.scale_plan is None


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("preparation", ["none", "raw", "prepared"])
@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("route", ["loaded", "path", "session"])
def test_native_pipeline_predictions_match_scalar_model(kind, preparation, backing, route, tmp_path):
    path, bundle, constants = _fitted(tmp_path, kind, preparation)
    frame, transformed = _incoming(preparation, constants)
    source, owner, expected_frame, positions = _source(tmp_path, frame, backing)
    before_bundle = hashlib.sha256(pickle.dumps(bundle)).hexdigest()
    native = None if isinstance(source, pd.DataFrame) else source.native
    source_roles = None if isinstance(source, pd.DataFrame) else dict(source.roles)
    scorer = None
    result = None
    try:
        if route == "session":
            scorer = Session.ingest(source) if isinstance(source, pd.DataFrame) else Session(dataset=source)
            history = len(scorer.history)
            result = scorer.predict_from_pipeline(path, apply_plans=preparation != "prepared",
                                                   return_proba=True, trusted=True)
            assert len(scorer.history) == history + 1
            assert scorer.fit_result is None and scorer.impute_plan is None
        else:
            result = predict_from_pipeline(path if route == "path" else bundle, source,
                                           apply_plans=preparation != "prepared", return_proba=True,
                                           trusted=route == "path")
        model = bundle.fit_result.estimator
        design = transformed[positions]
        if kind == "regression":
            values = design @ model.coef_ + model.intercept_
            np.testing.assert_allclose(result.predictions, values, atol=2e-12)
            assert result.probabilities is None
            assert any("no predict_proba" in w for w in result.warnings)
        else:
            logits = design @ model.coef_.T + model.intercept_
            if kind == "binary":
                positive = 1 / (1 + np.exp(-logits.ravel()))
                probabilities = np.column_stack([1-positive, positive])
            else:
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities = exp / exp.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(result.probabilities, probabilities, atol=2e-12)
            np.testing.assert_array_equal(result.predictions, model.classes_[probabilities.argmax(axis=1)])
            assert list(result.probabilities) == [f"proba_{c}" for c in model.classes_]
            assert result.probabilities.index.equals(expected_frame.index)
        assert result.predictions.index.equals(expected_frame.index)
        assert result.n_rows == len(positions) and result.feature_columns == ("x", "z")
        assert (result.apply_result is not None) == (preparation == "raw")
        assert hashlib.sha256(pickle.dumps(bundle)).hexdigest() == before_bundle
        if source_roles is not None:
            assert source.roles == source_roles
        _assert_input(source, owner, expected_frame, native)
    finally:
        if result is not None and result.apply_result is not None and result.apply_result.dataset is not source:
            result.apply_result.dataset.close_native()
        if scorer is not None:
            scorer.close_native()
        if owner is not None:
            owner.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("invalid", ["empty", "missing", "nonnumeric"])
@pytest.mark.parametrize("route", ["loaded", "path"])
def test_invalid_native_score_rows_preserve_source_and_saved_model(backing, invalid, route, tmp_path):
    path, bundle, constants = _fitted(tmp_path, "binary", "raw")
    frame, _ = _incoming("raw", constants)
    if invalid == "empty":
        frame = frame.iloc[:0]
    elif invalid == "missing":
        frame = frame.drop(columns="x")
    else:
        frame["z"] = "invalid-numeric"
    source, owner, expected, _ = _source(tmp_path, frame, backing)
    before = hashlib.sha256(pickle.dumps(bundle)).hexdigest()
    native = None if isinstance(source, pd.DataFrame) else source.native
    try:
        with pytest.raises(ValidationError, match={"empty": "empty", "missing": "missing|Missing", "nonnumeric": "coercion|dtype|type"}[invalid]):
            predict_from_pipeline(path if route == "path" else bundle, source, trusted=route == "path")
        assert hashlib.sha256(pickle.dumps(bundle)).hexdigest() == before
        _assert_input(source, owner, expected, native)
    finally:
        if owner is not None:
            owner.close_native()
