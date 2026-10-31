"""Rare-category thresholds and replay agree with independent count arithmetic."""

import copy
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor
from buildml.preprocess.encode import fit_encoder, transform_encoder

BUCKET = "__infrequent__"
COLS = ("category", "region")
VALID = [(0.25, 4), (np.float32(.25), 4), (np.float64(.25), 4),
         (4, 4), (np.int32(4), 4), (np.int64(4), 4), (1, 1), (20, 20), (10**100, 10**100)]
BAD = [True, False, np.bool_(True), np.bool_(False), 0, -1, 0., 1., -.25, 1.25,
       np.nan, np.inf, -np.inf, "2", "0.25", None, .25+0j, [2], np.array([2])]


def source_frame():
    rows = np.arange(16, dtype=float)
    train = pd.DataFrame({"x": rows, "category": ["common"]*8+["boundary"]*4+["rare"]*3+["single"],
                          "region": ["north"]*8+["south"]*7+["other"],
                          "y": 2*rows + np.sin(rows)})
    holdout = pd.DataFrame({"x": [20., 21., 22., 23.],
                            "category": ["rare", "rare", "novel", "single"],
                            "region": ["new", "other", "south", "new"], "y": [1e6]*4})
    frame = pd.concat([train, holdout], ignore_index=True)
    frame.index = [f"row-{index//2}" for index in range(len(frame))]
    return frame


def make_session(frame=None):
    return (Session.ingest(source_frame() if frame is None else frame)
            .set_roles({"x": "feature", "category": "feature", "region": "feature", "y": "target"})
            .inject_split(train_indices=list(range(16)), test_indices=list(range(16, 20))))


def manual_features(train, score, threshold):
    out = pd.DataFrame({"x": score.x.to_numpy()}, index=score.index)
    maps = {}
    for column in COLS:
        counts = Counter(str(value) for value in train[column])
        rare = {value for value, count in counts.items() if count < threshold}
        maps[column] = rare
        levels = sorted({BUCKET if value in rare else value for value in counts})
        values = [BUCKET if str(value) in rare else str(value) for value in score[column]]
        for level in levels:
            out[f"{column}_{level}"] = np.asarray([value == level for value in values], dtype=float)
    return out, maps


def ridge_predictions(fit_x, fit_y, score_x):
    design = np.column_stack([np.ones(len(fit_x)), fit_x])
    penalty = np.diag([0.]+[.5]*fit_x.shape[1])
    coefficients = np.linalg.solve(design.T @ design+penalty, design.T @ np.asarray(fit_y))
    return np.column_stack([np.ones(len(score_x)), score_x]) @ coefficients


@pytest.mark.parametrize("route", ["session", "fold"])
@pytest.mark.parametrize("value", BAD)
def test_invalid_rarity_controls_refuse_without_mutating_source_or_fitted_session(route, value):
    frame = source_frame()
    original = frame.copy(deep=True)
    if route == "session":
        session = make_session(frame).fit(DummyRegressor(), task="regression")
        prediction = session.predict().copy()
        fitted, history = session.fit_result, copy.deepcopy(session.history)
        with pytest.raises(ValidationError, match="min_frequency"):
            session.encode(columns=list(COLS), method="infrequent", min_frequency=value)
        assert session.fit_result is fitted and session.encode_plan is None
        assert session.history == history
        pd.testing.assert_frame_equal(session.dataset.frame, original)
        np.testing.assert_allclose(session.predict(), prediction)
    else:
        with pytest.raises(ValidationError, match="min_frequency"):
            build_fold_preprocessor(frame.iloc[:16][["x", *COLS]], PreprocessRecipe(
                impute=None, encode="infrequent", encode_columns=COLS, min_frequency=value))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("route", ["session", "fold"])
@pytest.mark.parametrize("value,threshold", VALID)
def test_supported_thresholds_use_only_training_counts_and_keep_boundary_levels(route, value, threshold):
    frame = source_frame()
    train = frame.iloc[:16]
    expected, maps = manual_features(train, frame, threshold)
    original = frame.copy(deep=True)
    if route == "session":
        session = make_session(frame).encode(columns=list(COLS), method="infrequent", min_frequency=value)
        actual = session.dataset.frame[list(expected)]
        plan = session.encode_plan
        assert {key: set(levels) for key, levels in plan.infrequent_maps_.items()} == maps
        assert type(plan.min_frequency) in (int, float)
        json.dumps(plan.to_dict(), allow_nan=False)
        assert any("Unseen" in text and "all-zero" in text for text in session.last_preprocess.limitations)
        pd.testing.assert_series_equal(session.dataset.frame.y, original.y)
    else:
        recipe = PreprocessRecipe(impute=None, encode="infrequent", encode_columns=COLS, min_frequency=value)
        prep = build_fold_preprocessor(train[["x", *COLS]], recipe)
        before = joblib.hash(prep)
        actual = prep.transform(frame[["x", *COLS]])
        assert joblib.hash(prep) == before
        assert {key: set(levels) for key, levels in prep._infrequent_maps.items()} == maps
        assert type(recipe.to_dict()["min_frequency"]) in (int, float)
        json.dumps(recipe.to_dict(), allow_nan=False)
    pd.testing.assert_frame_equal(actual, expected)
    assert all("novel" not in name and "new" not in name for name in actual)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("where", ["train", "holdout"])
@pytest.mark.parametrize("route", ["session", "fold"])
def test_reserved_literal_cannot_be_confused_with_the_internal_rare_bucket(where, route):
    frame = source_frame()
    frame.iloc[:8 if where == "train" else 0, frame.columns.get_loc("category")] = BUCKET
    if where == "holdout":
        frame.iloc[18, frame.columns.get_loc("category")] = BUCKET
    original = frame.copy(deep=True)
    if route == "session":
        session = make_session(frame)
        history = copy.deepcopy(session.history)
        with pytest.raises(ValidationError, match="reserved.*__infrequent__"):
            session.encode(columns=list(COLS), method="infrequent", min_frequency=4)
        assert session.encode_plan is None and session.history == history
        pd.testing.assert_frame_equal(session.dataset.frame, original)
    else:
        recipe = PreprocessRecipe(impute=None, encode="infrequent", encode_columns=COLS, min_frequency=4)
        if where == "train":
            with pytest.raises(ValidationError, match="reserved.*__infrequent__"):
                build_fold_preprocessor(frame.iloc[:16][["x", *COLS]], recipe)
        else:
            prep = build_fold_preprocessor(frame.iloc[:16][["x", *COLS]], recipe)
            before = joblib.hash(prep)
            with pytest.raises(ValidationError, match="reserved.*__infrequent__"):
                prep.transform(frame.iloc[16:][["x", *COLS]])
            assert joblib.hash(prep) == before
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("route", ["session", "fold"])
@pytest.mark.parametrize("frequency", [1, 4])
def test_reserved_literal_remains_a_real_category_when_its_column_has_no_pool(route, frequency):
    frame = source_frame()
    frame["category"] = BUCKET
    if route == "session":
        session = make_session(frame).encode(columns=list(COLS), method="infrequent", min_frequency=frequency)
        actual = session.dataset.frame
    else:
        prep = build_fold_preprocessor(frame.iloc[:16][["x", *COLS]], PreprocessRecipe(
            impute=None, encode="infrequent", encode_columns=COLS, min_frequency=frequency))
        actual = prep.transform(frame[["x", *COLS]])
    np.testing.assert_array_equal(actual[f"category_{BUCKET}"], np.ones(len(frame)))


@pytest.mark.parametrize("route", ["session", "fold"])
def test_literal_that_is_itself_a_fitted_rare_category_keeps_normal_pooling(route):
    frame = source_frame()
    frame.iloc[0, frame.columns.get_loc("category")] = BUCKET
    frame.iloc[18, frame.columns.get_loc("category")] = BUCKET
    expected, _ = manual_features(frame.iloc[:16], frame, 4)
    if route == "session":
        session = make_session(frame).encode(columns=list(COLS), method="infrequent", min_frequency=4)
        actual = session.dataset.frame[list(expected)]
    else:
        prep = build_fold_preprocessor(frame.iloc[:16][["x", *COLS]], PreprocessRecipe(
            impute=None, encode="infrequent", encode_columns=COLS, min_frequency=4))
        actual = prep.transform(frame[["x", *COLS]])
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("value", [.25, np.float32(.25), np.int64(4)])
def test_fold_cv_metrics_match_manual_train_only_vocabulary_and_ridge(value):
    session = make_session()
    train = session.partition("train")
    expected = []
    for fit_pos, score_pos in KFold(4, shuffle=True, random_state=42).split(train):
        fit, score = train.iloc[fit_pos], train.iloc[score_pos]
        threshold = 4 if isinstance(value, np.integer) else .25*len(fit)
        fit_x, _ = manual_features(fit, fit, threshold)
        score_x, _ = manual_features(fit, score, threshold)
        prediction = ridge_predictions(fit_x, fit.y, score_x)
        error = score.y.to_numpy()-prediction
        mse = np.mean(error**2)
        expected.append({"mae": np.mean(abs(error)), "mse": mse, "rmse": np.sqrt(mse),
                         "r2": 1-mse/np.mean((score.y-score.y.mean())**2)})
    result = session.cv_score(Ridge(alpha=.5), task="regression", cv=4, cv_strategy="kfold",
                              preprocess=PreprocessRecipe(impute=None, encode="infrequent",
                                                          encode_columns=COLS, min_frequency=value))
    for actual, metrics in zip(result.folds, expected, strict=True):
        assert actual.metrics == pytest.approx(metrics)
    for key in expected[0]:
        values = [metrics[key] for metrics in expected]
        assert result.mean_metrics[key] == pytest.approx(np.mean(values))
        assert result.std_metrics[key] == pytest.approx(np.std(values, ddof=1))
    json.dumps(result.to_dict(), allow_nan=False)
    pd.testing.assert_frame_equal(session.dataset.frame, source_frame())


@pytest.fixture(params=[.25, np.float32(.25), np.int64(4)])
def encoded_pipeline(request, tmp_path):
    raw = source_frame()
    session = make_session(raw).encode(columns=list(COLS), method="infrequent", min_frequency=request.param)
    session.fit(Ridge(alpha=.5), task="regression")
    path = tmp_path / "rare-pipeline"
    session.save_pipeline(path, evaluate_partition=None)
    return path, load_pipeline_bundle(path, trusted=True), raw


@pytest.mark.parametrize("from_path", [False, True])
def test_saved_pipeline_replays_frozen_rare_maps_and_independent_predictions(encoded_pipeline, from_path):
    path, bundle, raw = encoded_pipeline
    score = raw.iloc[16:][["x", *COLS]].copy()
    original = score.copy(deep=True)
    fit_x, _ = manual_features(raw.iloc[:16], raw.iloc[:16], 4)
    score_x, _ = manual_features(raw.iloc[:16], score, 4)
    expected = ridge_predictions(fit_x, raw.iloc[:16].y, score_x)
    before = joblib.hash(bundle)
    result = predict_from_pipeline(path if from_path else bundle, score, trusted=from_path)
    np.testing.assert_allclose(result.predictions, expected, rtol=1e-11, atol=1e-11)
    assert result.predictions.index.equals(score.index)
    assert result.feature_columns == tuple(fit_x)
    assert joblib.hash(bundle) == before
    pd.testing.assert_frame_equal(score, original)


def test_saved_pipeline_refuses_ambiguous_reserved_literal_without_mutating_bundle(encoded_pipeline):
    _, bundle, raw = encoded_pipeline
    score = raw.iloc[16:][["x", *COLS]].copy()
    score.iloc[0, score.columns.get_loc("category")] = BUCKET
    before = joblib.hash(bundle)
    with pytest.raises(ValidationError, match="reserved.*__infrequent__"):
        predict_from_pipeline(bundle, score)
    assert joblib.hash(bundle) == before


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
@pytest.mark.parametrize("route", ["session", "fold"])
def test_other_methods_continue_to_ignore_unused_frequency_knob(method, route):
    session = make_session()
    if route == "session":
        plan = fit_encoder(session.dataset, session.split_plan, columns=list(COLS),
                           method=method, min_frequency="unused")
        dataset, _ = transform_encoder(session.dataset, plan)
        assert len(dataset.frame) == 20
    else:
        train = session.partition("train")[["x", *COLS]]
        prep = build_fold_preprocessor(train, PreprocessRecipe(impute=None, encode=method,
                                                               encode_columns=COLS, min_frequency="unused"))
        assert len(prep.transform(train)) == 16
