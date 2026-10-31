"""Encoding preserves distinct feature meanings through fitting and saved replay."""

import copy
from dataclasses import replace

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
from buildml.preprocess.apply import apply_preprocess_plans
from buildml.preprocess.encode import fit_encoder, transform_encoder

METHODS = ("onehot", "infrequent", "target")
ROLES = ("feature", "target", "id", "group", "time", "weight", "ignore")


def raw_frame():
    x = np.arange(16, dtype=float)
    frame = pd.DataFrame({"x": x, "category": ["a", "b"]*8,
                          "y": .3 + 2*x + np.cos(x)})
    frame.index = [f"row-{i//2}" for i in range(len(frame))]
    return frame


def session_for(frame):
    return (Session.ingest(frame).set_roles({"x": "feature", "category": "feature", "y": "target"})
            .inject_split(train_indices=list(range(12)), test_indices=list(range(12, 16))))


def output_name(method):
    return "category_target" if method == "target" else "category_a"


def recipe_for(method, columns=("category",)):
    return PreprocessRecipe(impute=None, encode=method, encode_columns=columns,
                            min_frequency=1, target_smoothing=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("role", ROLES)
def test_session_fit_refuses_retained_name_in_every_role_and_preserves_model(method, role):
    frame = raw_frame()
    name = output_name(method)
    frame[name] = 101.3 + np.arange(len(frame))
    session = session_for(frame).set_roles({name: role})
    if role == "target":
        session.set_roles({"y": "ignore"})
    session.fit(DummyRegressor(), task="regression")
    expected = session.predict().copy()
    history = copy.deepcopy(session.history)
    fitted = session.fit_result
    with pytest.raises(ValidationError, match="Encoding output.*" + name):
        session.encode(columns=["category"], method=method, min_frequency=1, n_folds=3)
    assert session.history == history and session.encode_plan is None
    assert session.fit_result is fitted
    np.testing.assert_array_equal(session.predict(), expected)
    pd.testing.assert_frame_equal(session.dataset.frame, frame)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("route", ["fit_plan", "fold_fit", "plan_replay", "fold_replay", "apply_frozen"])
def test_colliding_retained_columns_refuse_at_fit_and_replay_without_mutation(method, route):
    raw = raw_frame()
    session = session_for(raw)
    x = raw.drop(columns="y")
    plan = fit_encoder(session.dataset, session.split_plan, columns=["category"], method=method,
                       min_frequency=1, smoothing=0, n_folds=3)
    prep = build_fold_preprocessor(x.iloc[:12], recipe_for(method), raw.y.iloc[:12])
    name = output_name(method)
    colliding = raw.assign(**{name: 900. + raw.x})
    original = colliding.copy(deep=True)
    before = joblib.hash((plan, prep))
    with pytest.raises(ValidationError, match="Encoding output.*" + name):
        if route == "fit_plan":
            conflict = session_for(colliding)
            fit_encoder(conflict.dataset, conflict.split_plan, columns=["category"], method=method,
                        min_frequency=1, n_folds=3)
        elif route == "fold_fit":
            build_fold_preprocessor(colliding.drop(columns="y").iloc[:12], recipe_for(method), raw.y.iloc[:12])
        elif route == "plan_replay":
            conflict = session_for(colliding)
            transform_encoder(conflict.dataset, plan, conflict.split_plan)
        elif route == "fold_replay":
            prep.transform(colliding.drop(columns="y"))
        else:
            apply_preprocess_plans(colliding.drop(columns="y"), {"encode_plan": plan})
    assert joblib.hash((plan, prep)) == before
    pd.testing.assert_frame_equal(colliding, original)
    pd.testing.assert_frame_equal(raw, raw_frame())


@pytest.mark.parametrize("method", ["onehot", "infrequent"])
@pytest.mark.parametrize("route", ["session", "fold"])
def test_category_separators_cannot_create_two_outputs_with_one_name(method, route):
    frame = raw_frame().assign(category=["a_b", "other"]*8, category_a=["b", "else"]*8)
    session = session_for(frame)
    with pytest.raises(ValidationError, match="Encoding output.*category_a_b"):
        if route == "session":
            session.encode(columns=["category", "category_a"], method=method, min_frequency=1)
        else:
            build_fold_preprocessor(frame.drop(columns="y"), recipe_for(method, ("category", "category_a")))
    pd.testing.assert_frame_equal(session.dataset.frame, frame)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("route", ["plan_replay", "apply_frozen"])
def test_duplicate_names_in_restored_plans_refuse_before_feature_assignment(method, route):
    frame = raw_frame().assign(region=["north", "south"]*8)
    session = session_for(frame)
    plan = fit_encoder(session.dataset, session.split_plan, columns=["category", "region"],
                       method=method, min_frequency=1, n_folds=3)
    broken = replace(plan, feature_names_=tuple("duplicated" for _ in plan.feature_names_))
    before = joblib.hash(broken)
    with pytest.raises(ValidationError, match="Encoding output.*duplicated"):
        if route == "plan_replay":
            transform_encoder(session.dataset, broken, session.split_plan)
        else:
            apply_preprocess_plans(frame.drop(columns="y"), {"encode_plan": broken})
    assert joblib.hash(broken) == before
    pd.testing.assert_frame_equal(session.dataset.frame, frame)


def target_means(fit, score, column):
    sums, counts = {}, {}
    for category, y in zip(fit[column], fit.y, strict=True):
        sums[category] = sums.get(category, 0.) + y
        counts[category] = counts.get(category, 0) + 1
    return np.asarray([sums[value]/counts[value] if value in counts else fit.y.mean()
                       for value in score[column]])


def manual_features(fit, score, method, columns):
    out = pd.DataFrame({"x": score.x.to_numpy()}, index=score.index)
    for column in columns:
        if method == "target":
            out[f"{column}_target"] = target_means(fit, score, column)
        elif method == "ordinal":
            levels = sorted(set(fit[column]))
            out[column] = [float(levels.index(value)) if value in levels else -1. for value in score[column]]
        else:
            for level in sorted(set(fit[column])):
                out[f"{column}_{level}"] = np.asarray(score[column] == level, dtype=float)
    return out


def ridge_oracle(fit_x, y, score_x):
    design = np.column_stack([np.ones(len(fit_x)), fit_x])
    penalty = np.diag([0.] + [.5]*fit_x.shape[1])
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(y))
    return np.column_stack([np.ones(len(score_x)), score_x]) @ beta


@pytest.mark.parametrize("method", [*METHODS, "ordinal"])
def test_fold_cv_predictions_keep_all_independent_features_with_manual_ridge(method):
    frame = raw_frame().assign(category_audit=100.+np.arange(16)**2)
    session = session_for(frame).set_roles({"category_audit": "feature"})
    train = frame.iloc[:12]
    expected = []
    for fit_pos, score_pos in KFold(3, shuffle=True, random_state=42).split(train):
        fit, score = train.iloc[fit_pos], train.iloc[score_pos]
        fit_x = manual_features(fit, fit, method, ("category",))
        score_x = manual_features(fit, score, method, ("category",))
        # Preserve the original numeric feature in the same native output order.
        fit_x.insert(1, "category_audit", fit.category_audit.to_numpy())
        score_x.insert(1, "category_audit", score.category_audit.to_numpy())
        pred = ridge_oracle(fit_x, fit.y, score_x)
        mse = np.mean((score.y-pred)**2)
        expected.append({"mse": mse, "rmse": np.sqrt(mse), "mae": np.mean(abs(score.y-pred)),
                         "r2": 1-mse/np.mean((score.y-score.y.mean())**2)})
    result = session.cv_score(Ridge(alpha=.5), task="regression", cv=3, cv_strategy="kfold",
                              preprocess=recipe_for(method))
    for fold, metrics in zip(result.folds, expected, strict=True):
        assert fold.metrics == pytest.approx(metrics, abs=1e-9)
    pd.testing.assert_frame_equal(session.dataset.frame, frame)


@pytest.mark.parametrize("method", METHODS)
def test_public_cv_refuses_collision_before_estimator_fit(method, monkeypatch):
    frame = raw_frame()
    frame[output_name(method)] = frame.x + 101.
    session = session_for(frame).set_roles({output_name(method): "feature"})

    def forbidden_fit(self, X, y, sample_weight=None):
        pytest.fail("Estimator must not fit a matrix with ambiguous features")

    monkeypatch.setattr(Ridge, "fit", forbidden_fit)
    with pytest.raises(ValidationError, match="Encoding output"):
        session.cv_score(Ridge(), task="regression", cv=3, cv_strategy="kfold",
                         preprocess=recipe_for(method))
    pd.testing.assert_frame_equal(session.dataset.frame, frame)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("from_path", [False, True])
def test_saved_pipeline_valid_features_and_predictions_match_independent_arithmetic(method, from_path, tmp_path):
    raw = raw_frame()
    columns = ("category",)
    session = session_for(raw).encode(columns=list(columns), method=method,
                                      min_frequency=1, smoothing=0, n_folds=3, random_state=7)
    train, score = raw.iloc[:12], raw.iloc[12:].drop(columns="y")
    fit_x = manual_features(train, train, method, columns)
    if method == "target":
        for fit_pos, pred_pos in KFold(3, shuffle=True, random_state=7).split(train):
            for column in columns:
                fit_x.iloc[pred_pos, fit_x.columns.get_loc(f"{column}_target")] = target_means(
                    train.iloc[fit_pos], train.iloc[pred_pos], column)
    expected_score = manual_features(train, score, method, columns)
    pd.testing.assert_frame_equal(session.dataset.frame[list(fit_x)].iloc[:12], fit_x)
    session.fit(Ridge(alpha=.5), task="regression")
    path = tmp_path / "encoded"
    session.save_pipeline(path, evaluate_partition=None)
    bundle = load_pipeline_bundle(path, trusted=True)
    before = joblib.hash(bundle)
    actual = predict_from_pipeline(path if from_path else bundle, score, trusted=from_path)
    np.testing.assert_allclose(actual.predictions, ridge_oracle(fit_x, train.y, expected_score),
                               atol=1e-10, rtol=1e-10)
    assert actual.feature_columns == tuple(fit_x)
    assert actual.predictions.index.equals(score.index)
    assert joblib.hash(bundle) == before
    pd.testing.assert_frame_equal(raw_frame(), raw.drop(columns="category_target", errors="ignore"))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("from_path", [False, True])
def test_saved_pipeline_refuses_new_input_collision_and_preserves_bundle(method, from_path, tmp_path):
    raw = raw_frame()
    session = session_for(raw).encode(columns=["category"], method=method,
                                      min_frequency=1, smoothing=0, n_folds=3)
    session.fit(Ridge(alpha=.5), task="regression")
    path = tmp_path / "encoded"
    session.save_pipeline(path, evaluate_partition=None)
    bundle = load_pipeline_bundle(path, trusted=True)
    before = joblib.hash(bundle)
    score = raw.iloc[12:].drop(columns="y").assign(**{output_name(method): [700., 800., 900., 1000.]})
    original = score.copy(deep=True)
    with pytest.raises(ValidationError, match="Encoding output.*" + output_name(method)):
        predict_from_pipeline(path if from_path else bundle, score, trusted=from_path)
    assert joblib.hash(bundle) == before
    pd.testing.assert_frame_equal(score, original)


@pytest.mark.parametrize("method", ["onehot", "infrequent"])
@pytest.mark.parametrize("route", ["session", "fold", "frozen"])
def test_outputs_can_reuse_removed_source_names_when_the_final_schema_is_unique(method, route):
    raw = raw_frame().assign(category_a=["c", "d"]*8)
    columns = ("category", "category_a")
    expected = manual_features(raw.iloc[:12], raw, method, columns)
    session = session_for(raw)
    if route == "session":
        session.encode(columns=list(columns), method=method, min_frequency=1)
        actual = session.dataset.frame.drop(columns="y")
    elif route == "fold":
        prep = build_fold_preprocessor(raw.drop(columns="y").iloc[:12], recipe_for(method, columns))
        actual = prep.transform(raw.drop(columns="y"))
    else:
        plan = fit_encoder(session.dataset, session.split_plan, columns=list(columns),
                           method=method, min_frequency=1)
        actual = apply_preprocess_plans(raw.drop(columns="y"), {"encode_plan": plan}).dataset.frame
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("route", ["session", "fold", "frozen"])
def test_target_outputs_can_reuse_removed_source_names_without_reading_overwritten_values(route):
    raw = raw_frame().assign(category_target=["north", "south", "south", "north"]*4)
    columns = ("category", "category_target")
    expected = manual_features(raw.iloc[:12], raw.drop(columns="y"), "target", columns)
    session = session_for(raw)
    if route == "session":
        session.encode(columns=list(columns), method="target", smoothing=0, n_folds=3, random_state=7)
        for fit_pos, pred_pos in KFold(3, shuffle=True, random_state=7).split(raw.iloc[:12]):
            for column in columns:
                expected.iloc[pred_pos, expected.columns.get_loc(f"{column}_target")] = target_means(
                    raw.iloc[fit_pos], raw.iloc[pred_pos], column)
        actual = session.dataset.frame.drop(columns="y")
    elif route == "fold":
        prep = build_fold_preprocessor(raw.drop(columns="y").iloc[:12], recipe_for("target", columns), raw.y.iloc[:12])
        actual = prep.transform(raw.drop(columns="y"))
    else:
        plan = fit_encoder(session.dataset, session.split_plan, columns=list(columns), method="target",
                           smoothing=0, n_folds=3)
        actual = apply_preprocess_plans(raw.drop(columns="y"), {"encode_plan": plan}).dataset.frame
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("route", ["session", "fold", "frozen"])
def test_a_pooled_indicator_cannot_replace_an_existing_literal_bucket_named_column(route):
    raw = raw_frame()
    raw["category"] = ["a"]*10 + ["rare", "tail", "a", "rare", "unknown", "tail"]
    raw["category___infrequent__"] = "retained audit label"
    session = session_for(raw)
    with pytest.raises(ValidationError, match="Encoding output.*category___infrequent__"):
        if route == "session":
            session.encode(columns=["category"], method="infrequent", min_frequency=2)
        elif route == "fold":
            build_fold_preprocessor(raw.drop(columns="y").iloc[:12], PreprocessRecipe(
                impute=None, encode="infrequent", encode_columns=("category",), min_frequency=2))
        else:
            clean = session_for(raw.drop(columns="category___infrequent__"))
            plan = fit_encoder(clean.dataset, clean.split_plan, columns=["category"],
                               method="infrequent", min_frequency=2)
            apply_preprocess_plans(raw.drop(columns="y"), {"encode_plan": plan})
    pd.testing.assert_frame_equal(session.dataset.frame, raw)


@pytest.mark.parametrize("route", ["session", "fold", "frozen"])
def test_ordinal_source_replacement_keeps_original_columns_and_unknown_values(route):
    raw = raw_frame()
    score = raw.copy()
    score.iloc[13, score.columns.get_loc("category")] = "unseen"
    expected = manual_features(raw.iloc[:12], score, "ordinal", ("category",))
    session = session_for(raw)
    if route == "session":
        session = session_for(score).encode(columns=["category"], method="ordinal")
        actual = session.dataset.frame.drop(columns="y")
    elif route == "fold":
        prep = build_fold_preprocessor(raw.drop(columns="y").iloc[:12], recipe_for("ordinal"))
        actual = prep.transform(score.drop(columns="y"))
    else:
        plan = fit_encoder(session.dataset, session.split_plan, columns=["category"], method="ordinal")
        actual = apply_preprocess_plans(score.drop(columns="y"), {"encode_plan": plan}).dataset.frame
    pd.testing.assert_frame_equal(actual, expected)
