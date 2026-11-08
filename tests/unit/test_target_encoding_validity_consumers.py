"""Target encoding validates training outcomes before learning or replacing plans."""
from __future__ import annotations

import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor
from buildml.preprocess.encode import fit_encoder

TRAIN = [9, 2, 5, 0, 11, 7, 3, 15, 17, 1, 19, 4, 8, 6, 13, 10, 14, 12, 18, 16, 23, 20, 21, 22]
BACKINGS = ["pandas", "polars-memory", "polars-lazy", "duckdb-memory", "duckdb-lazy"]
FAMILIES = ["float64", "float32", "integer", "boolean", "labels_object", "labels_string", "labels_category"]
BAD = ["null_object", "nan_object", "null_string", "null_category", "null_float",
       "null_integer", "null_boolean", "inf64", "neginf64", "inf32", "inf_nullable",
       "complex64", "complex128", "complex_real"]
COLS = ["category", "region"]


def _frame(family="float64"):
    i = np.arange(36)
    x = i / 10 - 1.5
    target = pd.Series(.3 + 1.7*x + np.cos(i), dtype="float32" if family == "float32" else "float64")
    if family == "integer":
        target = pd.Series(np.where(i % 2, 10, 2), dtype="int64")
    elif family == "boolean":
        target = pd.Series(i % 2 == 1)
    elif family.startswith("labels"):
        dtype = {"labels_object": object, "labels_string": "string", "labels_category": "category"}[family]
        target = pd.Series(np.array(["None", "nan", "<NA>"])[i % 3], dtype=dtype)
    frame = pd.DataFrame({"x": x, "category": ["A", "B", "C"]*12,
                          "region": ["east", "west"]*18, "target": target,
                          "row_id": [f"{n:04}" for n in i]})
    frame.loc[5, "category"] = "rare"
    frame.loc[30:, "category"] = "unseen"
    frame.index = [f"same-{n//2}" for n in i]
    return frame


def _session(frame, tmp_path, backing="pandas"):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    expected = frame.copy()
    if backing.endswith("lazy"):
        path = tmp_path / "input.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="lazy")
        expected = expected.reset_index(drop=True)
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles({"x": "feature", "category": "feature", "region": "feature",
                       "target": "target", "row_id": "id"})
    session.inject_split(train_indices=TRAIN, validation_indices=list(range(24, 30)), test_indices=list(range(30, 36)))
    return session, expected


def _targets(series, family):
    if family.startswith("labels"):
        return np.array([{"<NA>": 0., "None": 1., "nan": 2.}[value] for value in series])
    return series.to_numpy(dtype=float)


def _assert_target_values(actual, expected):
    """Native files may change category storage and null sentinels, not values."""
    assert actual.name == expected.name and actual.index.equals(expected.index)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_array_equal(actual[actual.notna()].to_numpy(), expected[expected.notna()].to_numpy())


def _means(fit, score, family, smoothing):
    y = _targets(fit.target, family)
    prior = float(np.mean(y))
    result = pd.DataFrame({"x": score.x.to_numpy()}, index=score.index)
    for column in COLS:
        values = []
        for category in score[column]:
            members = y[np.array(fit[column] == category)]
            values.append((sum(members) + smoothing*prior)/(len(members)+smoothing) if len(members) else prior)
        result[column+"_target"] = values
    return result


def _oof(raw, family, smoothing):
    train = raw.iloc[TRAIN]
    y = _targets(train.target, family)
    result = _means(train, raw, family, smoothing)
    splitter = KFold(3, shuffle=True, random_state=7) if family.startswith("float") else StratifiedKFold(3, shuffle=True, random_state=7)
    for fit, score in splitter.split(train, y):
        values = _means(train.iloc[fit], train.iloc[score], family, smoothing)
        result.iloc[[TRAIN[i] for i in score]] = values.to_numpy()
    return result


@pytest.fixture(scope="module")
def control_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("target-validity-control")
    session, _ = _session(_frame(), root)
    session.encode(columns=COLS, method="target", n_folds=3, random_state=7).fit(Ridge(), task="regression")
    return session.save_pipeline(root / "pipeline", evaluate_partition=None)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("smoothing", [0., 2.5])
def test_valid_target_maps_oof_features_and_saved_predictions(family, backing, smoothing, tmp_path):
    session, raw = _session(_frame(family), tmp_path, backing)
    try:
        session.encode(columns=COLS, method="target", n_folds=3, smoothing=smoothing, random_state=7)
        expected = _oof(raw, family, smoothing)
        pd.testing.assert_frame_equal(session.to_pandas()[list(expected)], expected, check_dtype=False)
        assert session.encode_plan.target_prior_ == pytest.approx(_targets(raw.iloc[TRAIN].target, family).mean())
        recipe = PreprocessRecipe(impute=None, encode="target", encode_columns=tuple(COLS), target_smoothing=smoothing)
        prep = build_fold_preprocessor(raw[["x", *COLS]].iloc[TRAIN], recipe, raw.target.iloc[TRAIN])
        pd.testing.assert_frame_equal(prep.transform(raw[["x", *COLS]]), _means(raw.iloc[TRAIN], raw, family, smoothing), check_dtype=False)
        regression = family.startswith("float")
        session.fit(Ridge(alpha=.8) if regression else LogisticRegression(C=.7, max_iter=500),
                    task="regression" if regression else "classification")
        path = session.save_pipeline(tmp_path / "pipeline", evaluate_partition=None)
        bundle = load_pipeline_bundle(path, trusted=True)
        before = joblib.hash(bundle)
        score = raw.drop(columns="target").iloc[[25, 30, 31, 28]].copy()
        matrix = _means(raw.iloc[TRAIN], score, family, smoothing).to_numpy()
        model = bundle.fit_result.estimator
        if regression:
            reference = matrix @ model.coef_ + model.intercept_
        else:
            logits = matrix @ model.coef_.T + model.intercept_
            if logits.shape[1] == 1:
                positive = 1/(1+np.exp(-logits.ravel()))
                probability = np.column_stack([1-positive, positive])
            else:
                exp = np.exp(logits-logits.max(axis=1, keepdims=True))
                probability = exp/exp.sum(axis=1, keepdims=True)
            reference = model.classes_[probability.argmax(axis=1)]
        for source in [bundle, path]:
            result = predict_from_pipeline(source, score, trusted=source == path, return_proba=True)
            if regression:
                np.testing.assert_allclose(result.predictions, reference, atol=2e-10)
            else:
                np.testing.assert_array_equal(result.predictions, reference)
                np.testing.assert_allclose(result.probabilities, probability, atol=2e-10)
                assert list(result.probabilities) == [f"proba_{value}" for value in model.classes_]
            assert result.predictions.index.equals(score.index)
        assert joblib.hash(bundle) == before
        _assert_target_values(session.to_pandas().target, raw.target)
    finally:
        session.close_native()


def _bad_frame(kind):
    raw = _frame("labels_object" if "object" in kind else "float64")
    if kind in {"null_string", "null_category"}:
        raw = _frame("labels_string" if kind == "null_string" else "labels_category")
    elif kind in {"null_integer", "null_boolean"}:
        raw["target"] = pd.Series(np.arange(len(raw)) % 2, index=raw.index, dtype="Int64" if kind == "null_integer" else "boolean")
    elif kind == "inf_nullable":
        raw["target"] = raw.target.astype("Float64")
    elif kind == "inf32":
        raw["target"] = raw.target.astype("float32")
    elif kind.startswith("complex"):
        raw["target"] = raw.target.astype("complex64" if kind == "complex64" else "complex128")
    value = np.nan
    if kind == "null_object":
        value = None
    elif kind in {"null_string", "null_integer", "null_boolean"}:
        value = pd.NA
    elif kind == "null_category":
        value = None
    elif kind.startswith("inf"):
        value = np.inf
    elif kind == "neginf64":
        value = -np.inf
    elif kind in {"complex64", "complex128"}:
        value = 2+3j
    elif kind == "complex_real":
        value = 2+0j
    raw.iloc[TRAIN[5], raw.columns.get_loc("target")] = value
    return raw


@pytest.mark.parametrize("kind", BAD)
@pytest.mark.parametrize("route", ["session", "fit_encoder", "fold"])
def test_invalid_training_targets_refuse_before_replacing_fitted_state(kind, route, control_pipeline, tmp_path):
    raw = _bad_frame(kind)
    session, _ = _session(raw, tmp_path)
    session.load_pipeline(control_pipeline, trusted=True)
    fitted, plan = session.fit_result, session.encode_plan
    before_fit, before_plan = joblib.hash(fitted), joblib.hash(plan)
    history, split = copy.deepcopy(session.history), session.split_plan.to_dict()
    with pytest.raises(ValidationError, match="[Tt]arget|non-null|finite|real"):
        if route == "session":
            session.encode(columns=COLS, method="target", n_folds=3)
        elif route == "fit_encoder":
            fit_encoder(session.dataset, session.split_plan, columns=COLS, method="target", n_folds=3)
        else:
            build_fold_preprocessor(raw[["x", *COLS]].iloc[TRAIN], PreprocessRecipe(
                impute=None, encode="target", encode_columns=tuple(COLS)), raw.target.iloc[TRAIN])
    assert session.history == history and session.split_plan.to_dict() == split
    assert session.fit_result is fitted and session.encode_plan is plan
    assert joblib.hash(fitted) == before_fit and joblib.hash(plan) == before_plan
    pd.testing.assert_frame_equal(session.to_pandas(), raw)


@pytest.mark.parametrize("kind", ["null_object", "inf64", "complex128"])
@pytest.mark.parametrize("method", ["onehot", "ordinal", "infrequent"])
@pytest.mark.parametrize("route", ["session", "fold"])
def test_unsupervised_encoders_ignore_unused_outcomes(kind, method, route, tmp_path):
    raw = _bad_frame(kind)
    if route == "session":
        session, _ = _session(raw, tmp_path)
        session.encode(columns=COLS, method=method)
        assert session.dataset.n_rows == len(raw)
    else:
        prep = build_fold_preprocessor(raw[["x", *COLS]].iloc[TRAIN], PreprocessRecipe(
            impute=None, encode=method, encode_columns=tuple(COLS)), raw.target.iloc[TRAIN])
        assert len(prep.transform(raw[["x", *COLS]])) == len(raw)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("poison", ["float_null", "float_inf", "label_null"])
def test_unused_holdout_outcomes_do_not_change_fitted_maps_or_features(backing, poison, tmp_path):
    raw = _frame("labels_object" if poison == "label_null" else "float64")
    expected = _oof(raw, "labels_object" if poison == "label_null" else "float64", 2.5)
    changed = raw.copy()
    changed.iloc[24:, changed.columns.get_loc("target")] = None if poison == "label_null" else (np.inf if poison == "float_inf" else np.nan)
    session, normalized = _session(changed, tmp_path, backing)
    try:
        session.encode(columns=COLS, method="target", n_folds=3, smoothing=2.5, random_state=7)
        if backing.endswith("lazy"):
            expected = expected.reset_index(drop=True)
        pd.testing.assert_frame_equal(session.to_pandas()[list(expected)], expected, check_dtype=False)
        _assert_target_values(session.to_pandas().target, normalized.target)
    finally:
        session.close_native()
