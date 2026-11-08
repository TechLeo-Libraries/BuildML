"""OOF target features use only the labels available to each fitting fold."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor

TRAIN = [9, 2, 5, 0, 11, 7, 3, 15, 17, 1, 19, 4, 8, 6, 13, 10, 14, 12, 18, 16]
COLS = ("category", "region")
BAD_SMOOTHING = [True, np.bool_(False), -.1, np.nan, np.inf, -np.inf,
                 "1", None, 1+0j, [1], np.array([1]), 10**400]
BAD_FOLDS = [True, np.bool_(True), False, 0, 1, -1, 2.5, 3., np.nan, np.inf,
             "3", None, 3+0j, [3], np.array([3])]


def raw_frame(target="continuous", kind="mixed"):
    x = np.arange(24, dtype=float)
    frame = pd.DataFrame({"x": x, "category": ["a", "b"]*12,
                          "region": ["east", "east", "west", "west"]*6,
                          "y": .3 + 2*x + np.cos(x)})
    if kind == "mixed":
        frame.loc[0, "category"] = "singleton"
        frame.loc[1, "region"] = "rare"
    if kind == "unique":
        frame["category"] = [f"level-{i}" for i in range(len(frame))]
    if target == "binary":
        frame["y"] = [0, 1]*12
    elif target == "labels":
        frame["y"] = ["approved", "rejected"]*12
    frame.loc[22:, "category"] = "unseen"
    frame.index = [f"row-{i//2}" for i in range(len(frame))]
    return frame


def make_session(frame):
    return (Session.ingest(frame).set_roles({"x": "feature", "category": "feature",
                                             "region": "feature", "y": "target"})
            .inject_split(train_indices=TRAIN, test_indices=[20, 21, 22, 23]))


def target_values(frame):
    return np.asarray([{"approved": 0., "rejected": 1.}.get(value, value) for value in frame.y], dtype=float)


def means_reference(fit, score, smoothing, columns=COLS):
    """Use the defining shrinkage equation, independently of BuildML helpers."""
    y = target_values(fit)
    prior = np.mean(y)
    out = pd.DataFrame({"x": score.x.to_numpy()}, index=score.index)
    for column in columns:
        values = []
        for category in score[column]:
            members = y[np.asarray(fit[column] == category)]
            values.append((np.sum(members)+smoothing*prior)/(len(members)+smoothing)
                          if len(members) else prior)
        out[f"{column}_target"] = values
    return out


def oof_reference(raw, folds, smoothing, target="continuous"):
    train = raw.iloc[TRAIN]
    expected = means_reference(train, raw, smoothing)
    if target == "continuous":
        splits = KFold(folds, shuffle=True, random_state=7).split(train)
    else:
        splits = StratifiedKFold(folds, shuffle=True, random_state=7).split(train, target_values(train))
    for fit_pos, score_pos in splits:
        fit = train.iloc[fit_pos]
        score = train.iloc[score_pos]
        values = means_reference(fit, score, smoothing)
        global_positions = [TRAIN[i] for i in score_pos]
        expected.iloc[global_positions] = values.to_numpy()
    return expected


@pytest.mark.parametrize("folds", [2, 3, 5])
@pytest.mark.parametrize("smoothing", [0., .5, 10.])
@pytest.mark.parametrize("target", ["continuous", "binary", "labels"])
def test_session_oof_and_holdout_features_match_independent_fold_priors(folds, smoothing, target):
    raw = raw_frame(target)
    session = make_session(raw)
    session.encode(columns=list(COLS), method="target", n_folds=folds, smoothing=smoothing, random_state=7)
    expected = oof_reference(raw, folds, smoothing, target)
    pd.testing.assert_frame_equal(session.dataset.frame[list(expected)], expected)
    assert session.encode_plan.target_prior_ == pytest.approx(np.mean(target_values(raw.iloc[TRAIN])))
    pd.testing.assert_series_equal(session.dataset.frame.y, raw.y)
    assert session.dataset.frame.index.equals(raw.index)


@pytest.mark.parametrize("folds", [2, 3, 5])
@pytest.mark.parametrize("smoothing", [0., .5, 10.])
@pytest.mark.parametrize("kind", ["known", "mixed", "unique"])
def test_a_held_out_folds_labels_cannot_change_any_of_its_own_features(folds, smoothing, kind):
    raw = raw_frame(kind=kind)
    train = raw.iloc[TRAIN]
    _, held_out = next(KFold(folds, shuffle=True, random_state=7).split(train))
    positions = [TRAIN[i] for i in held_out]
    changed = raw.copy(deep=True)
    changed.iloc[positions, changed.columns.get_loc("y")] += 1000.
    results = []
    for frame in (raw, changed):
        session = make_session(frame).encode(columns=list(COLS), method="target", n_folds=folds,
                                             smoothing=smoothing, random_state=7)
        results.append(session.dataset.frame[[f"{column}_target" for column in COLS]])
    pd.testing.assert_frame_equal(results[0].iloc[positions], results[1].iloc[positions])
    pd.testing.assert_frame_equal(raw, raw_frame(kind=kind))


@pytest.mark.parametrize("smoothing", [0., .5, 10.])
def test_test_labels_cannot_influence_fitted_maps_prior_or_any_encoded_value(smoothing):
    raw = raw_frame()
    changed = raw.copy(deep=True)
    changed.iloc[20:, changed.columns.get_loc("y")] = 1e8
    sessions = [make_session(frame).encode(columns=list(COLS), method="target", n_folds=3,
                                          smoothing=smoothing, random_state=7)
                for frame in (raw, changed)]
    assert sessions[0].encode_plan.to_dict() == sessions[1].encode_plan.to_dict()
    pd.testing.assert_frame_equal(sessions[0].dataset.frame.drop(columns="y"),
                                  sessions[1].dataset.frame.drop(columns="y"))


@pytest.mark.parametrize("value", BAD_FOLDS)
def test_invalid_oof_counts_raise_validation_before_mutating_a_fitted_session(value):
    raw = raw_frame()
    session = make_session(raw).fit(DummyRegressor(), task="regression")
    before = session.predict().copy()
    fit, history = session.fit_result, copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="n_folds"):
        session.encode(columns=list(COLS), method="target", n_folds=value)
    assert session.fit_result is fit and session.encode_plan is None and session.history == history
    pd.testing.assert_frame_equal(session.dataset.frame, raw)
    np.testing.assert_array_equal(session.predict(), before)


@pytest.mark.parametrize("value", BAD_SMOOTHING)
@pytest.mark.parametrize("route", ["session", "fold"])
def test_invalid_target_smoothing_refuses_before_feature_or_model_mutation(value, route):
    raw = raw_frame()
    if route == "session":
        session = make_session(raw).fit(DummyRegressor(), task="regression")
        before = session.predict().copy()
        fit, history = session.fit_result, copy.deepcopy(session.history)
        with pytest.raises(ValidationError, match="smoothing"):
            session.encode(columns=list(COLS), method="target", smoothing=value, n_folds=3)
        assert session.fit_result is fit and session.encode_plan is None and session.history == history
        pd.testing.assert_frame_equal(session.dataset.frame, raw)
        np.testing.assert_array_equal(session.predict(), before)
    else:
        with pytest.raises(ValidationError, match="smoothing"):
            build_fold_preprocessor(raw.drop(columns="y").iloc[TRAIN], PreprocessRecipe(
                impute=None, encode="target", encode_columns=COLS, target_smoothing=value), raw.y.iloc[TRAIN])
    pd.testing.assert_frame_equal(raw, raw_frame())


@pytest.mark.parametrize("folds,smoothing", [(np.int32(3), np.float32(.5)),
                                          (np.int64(5), np.float64(10)), (3, 0)])
def test_valid_numpy_controls_retain_numerical_semantics_and_json_metadata(folds, smoothing):
    raw = raw_frame()
    session = make_session(raw).encode(columns=list(COLS), method="target", n_folds=folds,
                                       smoothing=smoothing, random_state=7)
    expected = oof_reference(raw, int(folds), float(smoothing))
    pd.testing.assert_frame_equal(session.dataset.frame[list(expected)], expected)
    assert type(session.encode_plan.n_folds) is int
    assert type(session.encode_plan.smoothing) is float
    json.dumps(session.encode_plan.to_dict(), allow_nan=False)
    recipe = PreprocessRecipe(impute=None, encode="target", encode_columns=COLS, target_smoothing=smoothing)
    prep = build_fold_preprocessor(raw.drop(columns="y").iloc[TRAIN], recipe, raw.y.iloc[TRAIN])
    result = prep.transform(raw.drop(columns="y"))
    pd.testing.assert_frame_equal(result, means_reference(raw.iloc[TRAIN], raw, float(smoothing)))
    assert type(recipe.to_dict()["target_smoothing"]) is float
    json.dumps(recipe.to_dict(), allow_nan=False)


def ridge_oracle(fit_x, y, score_x):
    design = np.column_stack([np.ones(len(fit_x)), fit_x])
    coefficients = np.linalg.solve(design.T @ design + np.diag([0.] + [.75]*fit_x.shape[1]),
                                   design.T @ y)
    return np.column_stack([np.ones(len(score_x)), score_x]) @ coefficients


@pytest.mark.parametrize("from_path", [False, True])
@pytest.mark.parametrize("smoothing", [0., .5, 10.])
def test_saved_prediction_uses_full_train_maps_with_a_model_fit_on_independent_oof_features(
    from_path, smoothing, tmp_path
):
    raw = raw_frame()
    session = make_session(raw).encode(columns=list(COLS), method="target", n_folds=3,
                                       smoothing=smoothing, random_state=7)
    expected_features = oof_reference(raw, 3, smoothing)
    session.fit(Ridge(alpha=.75), task="regression")
    path = tmp_path / "oof-encoded"
    session.save_pipeline(path, evaluate_partition=None)
    bundle = load_pipeline_bundle(path, trusted=True)
    before = joblib.hash(bundle)
    score = raw.drop(columns="y").iloc[20:]
    expected_x = means_reference(raw.iloc[TRAIN], score, smoothing)
    expected = ridge_oracle(expected_features.iloc[TRAIN], raw.y.iloc[TRAIN].to_numpy(), expected_x)
    result = predict_from_pipeline(path if from_path else bundle, score, trusted=from_path)
    np.testing.assert_allclose(result.predictions, expected, rtol=1e-10, atol=1e-10)
    assert joblib.hash(bundle) == before
    pd.testing.assert_frame_equal(score, raw.drop(columns="y").iloc[20:])


@pytest.mark.parametrize("method", ["onehot", "ordinal", "infrequent"])
def test_other_methods_keep_ignoring_unused_target_controls(method):
    raw = raw_frame()
    session = make_session(raw).encode(columns=list(COLS), method=method, smoothing="unused", n_folds="unused")
    assert len(session.dataset.frame) == len(raw)
    prep = build_fold_preprocessor(raw.drop(columns="y").iloc[TRAIN], PreprocessRecipe(
        impute=None, encode=method, encode_columns=COLS, target_smoothing="unused"))
    assert len(prep.transform(raw.drop(columns="y"))) == len(raw)


def test_two_training_rows_use_the_other_label_even_when_stratification_cannot_split():
    raw = pd.DataFrame({"x": [0., 1., 2., 3.], "category": ["a", "b", "a", "unseen"],
                        "y": [0, 1, 0, 1]})
    session = (Session.ingest(raw).set_roles({"x": "feature", "category": "feature", "y": "target"})
               .inject_split(train_indices=[0, 1], test_indices=[2, 3]))
    session.encode(columns=["category"], method="target", n_folds=5, smoothing=10, random_state=7)
    np.testing.assert_allclose(session.dataset.frame.category_target, [1., 0., 5/11, .5])


def test_a_single_training_row_refuses_oof_encoding_without_destroying_a_fitted_model():
    raw = raw_frame()
    session = (Session.ingest(raw).set_roles({"x": "feature", "category": "feature", "y": "target"})
               .inject_split(train_indices=[0], test_indices=list(range(1, len(raw)))))
    session.fit(DummyRegressor(), task="regression")
    fit, history = session.fit_result, copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="at least two training rows"):
        session.encode(columns=["category"], method="target")
    assert session.fit_result is fit and session.history == history and session.encode_plan is None
    pd.testing.assert_frame_equal(session.dataset.frame, raw)
