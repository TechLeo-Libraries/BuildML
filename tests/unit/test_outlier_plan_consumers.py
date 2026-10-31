"""Real-valued outlier plans retain their meaning through native saved pipelines."""

import copy
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline, save_pipeline_bundle
from buildml.preprocess.apply import apply_preprocess_plans
from buildml.preprocess.outliers import apply_outlier_plan, fit_outlier_plan

FEATURES = ["x", "constant", "keep"]
BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")


def _frame():
    return pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0, -100.0, 2.0, 100.0, 3.0],
            "constant": [2.0] * 10 + [5.0, 2.0, 5.0, 2.0],
            "keep": np.cos(np.arange(14) / 3),
            "target": 1 + np.arange(14) / 7,
            "record": [f"row-{i}" for i in range(14)],
        },
        index=[i // 2 for i in range(14)],
    )


def _session(frame, backing, directory):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "population.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    return session.set_roles(
        {c: "feature" for c in FEATURES} | {"target": "target", "record": "id"}
    ).inject_split(
        train_indices=list(range(10)), validation_indices=[10, 11], test_indices=[12, 13]
    )


def _quantile(values, probability):
    values = sorted(float(x) for x in values)
    point = (len(values) - 1) * probability
    a, b = math.floor(point), math.ceil(point)
    return values[a] + (point - a) * (values[b] - values[a])


def _fences(values, method):
    if method == "iqr":
        a, b = _quantile(values, 0.25), _quantile(values, 0.75)
        return a - 1.25 * (b - a), b + 1.25 * (b - a)
    mean = sum(float(v) for v in values) / len(values)
    sd = math.sqrt(sum((float(v) - mean) ** 2 for v in values) / len(values))
    return mean - 1.75 * sd, mean + 1.75 * sd


def _capped(frame, bounds):
    result = frame.copy(deep=True)
    for col, (lo, hi) in bounds.items():
        result[col] = [min(hi, max(lo, float(v))) for v in frame[col]]
    return result


def _prediction(train, query):
    x, y = train[FEATURES].to_numpy(), train.target.to_numpy()
    center, mean_y = x.mean(axis=0), y.mean()
    x0 = x - center
    beta = np.linalg.solve(x0.T @ x0 + 0.5 * np.eye(len(FEATURES)), x0.T @ (y - mean_y))
    return (query[FEATURES].to_numpy() - center) @ beta + mean_y


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("method", ["iqr", "zscore"])
@pytest.mark.parametrize("action", ["detect", "cap", "drop"])
@pytest.mark.parametrize("scalar", [float, np.float32], ids=["python", "numpy32"])
def test_fences_partition_rewrites_and_saved_ridge_match_independent_values(
    backing, method, action, scalar, tmp_path
):
    session = _session(_frame(), backing, tmp_path)
    raw = session.dataset.to_pandas().copy(deep=True)
    original_split = session.split_plan
    bounds = {c: _fences(raw[c].iloc[:10], method) for c in ("x", "constant")}
    flags = [
        any(float(row[c]) < lo or float(row[c]) > hi for c, (lo, hi) in bounds.items())
        for _, row in raw.iterrows()
    ]
    params = dict(
        columns=["x", "constant"],
        method=method,
        action=action,
        iqr_multiplier=scalar(1.25),
        zscore_threshold=scalar(1.75),
    )
    plan = fit_outlier_plan(session.dataset, original_split, **params)
    for c, (lo, hi) in bounds.items():
        assert float(plan.lower_[c]) == pytest.approx(lo, abs=1e-12, rel=1e-12)
        assert float(plan.upper_[c]) == pytest.approx(hi, abs=1e-12, rel=1e-12)
    assert plan.n_flagged_train == sum(flags[:10])
    exported = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    assert exported["lower_"] == {c: float(plan.lower_[c]) for c in bounds}
    session.preprocess.handle_outliers(**params)
    actual = session.dataset.to_pandas()
    if action == "drop":
        kept = [i for i, flag in enumerate(flags) if not flag]
        expected = raw.iloc[kept].reset_index(drop=True)
        remap = {old: new for new, old in enumerate(kept)}
        for name in ("train", "validation", "test"):
            assert session.split_plan.indices_for(name) == tuple(
                remap[i] for i in original_split.indices_for(name) if i in remap
            )
        assert session.preprocess.outlier_plan.n_dropped == len(raw) - len(kept)
    else:
        expected = _capped(raw, bounds) if action == "cap" else raw
        assert session.split_plan is original_split
    pd.testing.assert_frame_equal(actual, expected)
    train = expected.iloc[list(session.split_plan.train_indices)]
    session.classical.fit(Ridge(alpha=0.5, solver="svd"), task="regression")
    path = tmp_path / "pipeline"
    session.save_pipeline(path, evaluate_partition=None)
    bundle = load_pipeline_bundle(path, trusted=True)
    state = joblib.hash(bundle)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()
    }
    score = pd.DataFrame(
        {"x": [-100.0, 0.0, 100.0, 3.4], "constant": [2.0] * 4, "keep": [0.2, 0.5, -0.4, 0.9]},
        index=[9, 9, 3, 1],
    )
    score_before = score.copy(deep=True)
    # A saved drop plan caps at score time without a SplitPlan; row count is preserved.
    prepared = score if action == "detect" else _capped(score, bounds)
    expected_prediction = _prediction(train, prepared)
    for source in (path, bundle):
        result = predict_from_pipeline(source, score, trusted=True)
        np.testing.assert_allclose(result.predictions, expected_prediction, atol=1e-10, rtol=1e-10)
        pd.testing.assert_index_equal(result.predictions.index, score.index)
        if action == "drop":
            assert any("cap" in w and "drop" in w for w in result.warnings)
        if action != "detect":
            complex_score = score.copy(deep=True)
            complex_score["x"] = complex_score.x.astype(complex) + 2j
            unchanged = complex_score.copy(deep=True)
            with pytest.raises(ValidationError, match="[Rr]eal|[Cc]omplex"):
                predict_from_pipeline(source, complex_score, trusted=True)
            pd.testing.assert_frame_equal(complex_score, unchanged)
    assert joblib.hash(bundle) == state
    assert files == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()
    }
    pd.testing.assert_frame_equal(score, score_before)


@pytest.mark.parametrize("parameter", ["method", "action"])
@pytest.mark.parametrize("bad", ["bad", True, [], {}, None, np.nan])
def test_bad_policies_do_not_replace_a_prior_outlier_plan(parameter, bad, tmp_path):
    session = _session(_frame(), "pandas", tmp_path)
    session.preprocess.handle_outliers(action="detect")
    previous, result = session.preprocess.outlier_plan, session.preprocess.last_preprocess
    before, history, split = (
        session.dataset.to_pandas().copy(deep=True),
        copy.deepcopy(session.history),
        session.split_plan,
    )
    with pytest.raises(ValidationError, match="[Oo]utlier"):
        session.preprocess.handle_outliers(**{parameter: bad})
    assert (
        session.preprocess.outlier_plan is previous and session.preprocess.last_preprocess is result
    )
    assert session.split_plan is split and session.history == history
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


@pytest.mark.parametrize("parameter", ["iqr_multiplier", "zscore_threshold"])
@pytest.mark.parametrize("bad", [10**400, np.nan, np.inf, None, True, "1.5", 1 + 2j])
def test_bad_thresholds_raise_validation_errors_before_state_publication(parameter, bad, tmp_path):
    session = _session(_frame(), "pandas", tmp_path)
    history = copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="iqr_multiplier|zscore_threshold"):
        session.preprocess.handle_outliers(**{parameter: bad})
    assert session.preprocess.outlier_plan is None and session.history == history


@pytest.mark.parametrize("method", ["iqr", "zscore"])
@pytest.mark.parametrize("action", ["detect", "cap", "drop"])
@pytest.mark.parametrize("storage", ["complex", "object"])
def test_complex_training_observations_are_refused(method, action, storage, tmp_path):
    frame = _frame()
    frame["x"] = frame.x.astype(complex) + 2j
    if storage == "object":
        frame["x"] = frame.x.astype(object)
    session = _session(frame, "pandas", tmp_path)
    before, history = session.dataset.to_pandas().copy(deep=True), copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="[Rr]eal|[Cc]omplex|numeric"):
        session.preprocess.handle_outliers(columns=["x"], method=method, action=action)
    assert session.preprocess.outlier_plan is None and session.history == history
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


@pytest.mark.parametrize("action", ["cap", "drop"])
@pytest.mark.parametrize("storage", ["complex", "object"])
@pytest.mark.parametrize("surface", ["direct", "score", "score-with-split"])
def test_complex_replay_observations_are_refused_without_changing_the_plan(
    action, storage, surface, tmp_path
):
    session = _session(_frame(), "pandas", tmp_path)
    plan = fit_outlier_plan(session.dataset, session.split_plan, action=action, columns=["x"])
    frame = pd.DataFrame({"x": np.array([1, 2, 3, 1, 2, 3], dtype=complex) + 2j})
    if storage == "object":
        frame["x"] = frame.x.astype(object)
    before, state = frame.copy(deep=True), joblib.hash(plan)
    split = SplitPlan("injected", None, None, 0, None, (0, 1, 2), (), (3, 4, 5))
    with pytest.raises(ValidationError, match="[Rr]eal|[Cc]omplex"):
        if surface == "direct":
            apply_outlier_plan(Session.ingest(frame).dataset, split, plan)
        else:
            apply_preprocess_plans(
                frame,
                split_plan=split if surface == "score-with-split" else None,
                plans={"outlier_plan": plan},
            )
    pd.testing.assert_frame_equal(frame, before)
    assert joblib.hash(plan) == state


def test_real_legacy_numpy_fences_export_and_resave_without_refitting(tmp_path):
    path = Path(__file__).parents[1] / "fixtures" / "outlier-plan-numpy-fences-2a4ff11.joblib"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["producer_source"] == "2a4ff11d36ce65261a09eed13ba8386879cc6ffd"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["fixture_sha256"]
    plan = joblib.load(path)
    assert isinstance(plan.lower_["x"], np.float32)
    state = joblib.hash(plan)
    assert json.loads(json.dumps(plan.to_dict(), allow_nan=False))["lower_"] == meta["lower"]
    bounds = {c: (float(plan.lower_[c]), float(plan.upper_[c])) for c in plan.columns}
    frame = _capped(_frame(), bounds)
    session = _session(frame, "pandas", tmp_path)
    session.classical.fit(Ridge(alpha=0.5, solver="svd"), task="regression")
    stored = save_pipeline_bundle(
        tmp_path / "legacy-pipeline", fit_result=session._fit_result, outlier_plan=plan
    )
    bundle = load_pipeline_bundle(stored, trusted=True)
    score = _frame().iloc[10:][FEATURES]
    result = predict_from_pipeline(bundle, score)
    expected = _prediction(frame.iloc[:10], _capped(score, bounds))
    np.testing.assert_allclose(result.predictions, expected, atol=1e-10, rtol=1e-10)
    assert joblib.hash(plan) == state
    assert (
        json.loads((stored / "meta.json").read_text(encoding="utf-8"))["outlier_plan"]["lower_"]
        == meta["lower"]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["fixture_sha256"]


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_absent_training_measurements_cannot_define_fences(method, tmp_path):
    frame = _frame()
    frame["x"] = np.nan
    session = _session(frame, "pandas", tmp_path)
    with pytest.raises(ValidationError, match="no finite train values"):
        session.preprocess.handle_outliers(columns=["x"], method=method)


@pytest.mark.parametrize("action", ["detect", "cap", "drop"])
def test_missing_replay_column_keeps_the_fitted_plan_usable(action, tmp_path):
    session = _session(_frame(), "pandas", tmp_path)
    raw = session.dataset.to_pandas().copy(deep=True)
    plan = fit_outlier_plan(session.dataset, session.split_plan, columns=["x"], action=action)
    missing = raw.drop(columns="x")
    before, state = missing.copy(deep=True), joblib.hash(plan)
    with pytest.raises(ValidationError, match="columns missing"):
        apply_outlier_plan(Session.ingest(missing).dataset, session.split_plan, plan)
    pd.testing.assert_frame_equal(missing, before)
    bounds = {"x": _fences(raw.x.iloc[:10], "iqr")}
    # The original plan used its default multiplier, independently recover its fences.
    a, b = _quantile(raw.x.iloc[:10], 0.25), _quantile(raw.x.iloc[:10], 0.75)
    bounds["x"] = a - 1.5 * (b - a), b + 1.5 * (b - a)
    replay, _, _, _ = apply_outlier_plan(session.dataset, session.split_plan, plan)
    if action == "drop":
        lo, hi = bounds["x"]
        expected = raw.iloc[[i for i, value in enumerate(raw.x) if lo <= value <= hi]].reset_index(
            drop=True
        )
    else:
        expected = _capped(raw, bounds) if action == "cap" else raw
    pd.testing.assert_frame_equal(replay.to_pandas(), expected)
    assert joblib.hash(plan) == state


@pytest.mark.parametrize("empty", ["train", "test"])
def test_drop_refusal_preserves_previous_plan_and_positional_split(empty):
    x = (
        [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0]
        if empty == "train"
        else [-1.0, 0.0, 1.0, 0.0, 100.0, -100.0]
    )
    threshold = 0.01 if empty == "train" else 3.0
    session = (
        Session.ingest(pd.DataFrame({"x": x, "y": np.arange(6, dtype=float)}))
        .set_roles({"x": "feature", "y": "target"})
        .inject_split(train_indices=[0, 1, 2, 3], test_indices=[4, 5])
    )
    session.preprocess.handle_outliers(method="zscore", action="detect", zscore_threshold=threshold)
    plan, result, split = (
        session.preprocess.outlier_plan,
        session.preprocess.last_preprocess,
        session.split_plan,
    )
    before, history = session.dataset.to_pandas().copy(deep=True), copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="entire train or test"):
        session.preprocess.handle_outliers(
            method="zscore", action="drop", zscore_threshold=threshold
        )
    assert session.preprocess.outlier_plan is plan and session.preprocess.last_preprocess is result
    assert session.split_plan is split and session.history == history
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)
