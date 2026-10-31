"""Saved raw category contracts preserve labels before numeric feature replay."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline, save_pipeline_bundle
from buildml.pipeline.contract import (
    build_schema_contract,
    coerce_score_frame,
    input_columns_from_plans,
    validate_score_frame,
)

LAYOUTS = [("ordinal", False), ("target", True), ("onehot", True), ("infrequent", True)]


def raw_data(dtype, labels):
    x = np.arange(16, dtype=float)
    values = (["01", "1"] if labels == "numeric_text" else ["north", "south"])*8
    frame = pd.DataFrame({"x": x, "category": pd.Series(values, dtype=dtype),
                          "y": .3 + 2*x + np.cos(x)})
    frame.index = [f"row-{i//2}" for i in range(len(frame))]
    return frame


def features(fit, score, method, columns):
    out = pd.DataFrame({"x": score.x.to_numpy()}, index=score.index)
    for column in columns:
        train_values = list(fit[column].astype(str))
        values = list(score[column].astype(str))
        levels = sorted(set(train_values))
        if method == "ordinal":
            out[column] = [float(levels.index(value)) if value in levels else -1. for value in values]
        elif method == "target":
            means = {level: np.mean([y for category, y in zip(train_values, fit.y, strict=True)
                                    if category == level]) for level in levels}
            out[f"{column}_target"] = [means.get(value, fit.y.mean()) for value in values]
        else:
            for level in levels:
                out[f"{column}_{level}"] = np.asarray([value == level for value in values], dtype=float)
    return out


def ridge_oracle(fit_x, y, score_x):
    design = np.column_stack([np.ones(len(fit_x)), fit_x])
    coefficients = np.linalg.solve(design.T @ design + np.diag([0.] + [.5]*fit_x.shape[1]),
                                   design.T @ np.asarray(y))
    return np.column_stack([np.ones(len(score_x)), score_x]) @ coefficients


@pytest.fixture(params=LAYOUTS, ids=[method for method, _ in LAYOUTS])
def saved_category_model(request, tmp_path):
    method, overlap = request.param

    def make(dtype, labels):
        raw = raw_data(dtype, labels)
        columns = ["category"]
        if overlap:
            second = "category_target" if method == "target" else "category_" + str(raw.category.iloc[0])
            raw[second] = pd.Series(["east", "west", "west", "east"]*4,
                                    index=raw.index, dtype=dtype)
            columns.append(second)
        source_dtypes = {column: str(raw[column].dtype) for column in columns}
        session = (Session.ingest(raw).set_roles({"x": "feature", "category": "feature", "y": "target"})
                   .inject_split(train_indices=list(range(12)), test_indices=list(range(12, 16))))
        session.encode(columns=columns, method=method, min_frequency=1,
                       smoothing=0, n_folds=3, random_state=7)
        train, score = raw.iloc[:12], raw.iloc[12:].drop(columns="y")
        fit_x = features(train, train, method, columns)
        if method == "target":
            for fit_pos, pred_pos in KFold(3, shuffle=True, random_state=7).split(train):
                values = features(train.iloc[fit_pos], train.iloc[pred_pos], method, columns)
                fit_x.iloc[pred_pos] = values.to_numpy()
        score_x = features(train, score, method, columns)
        expected = ridge_oracle(fit_x, train.y, score_x)
        session.fit(Ridge(alpha=.5), task="regression")
        np.testing.assert_allclose(session.predict(), expected, atol=1e-10)
        path = tmp_path / f"{method}-{dtype}-{labels}"
        session.save_pipeline(path, evaluate_partition=None)
        return dict(path=path, bundle=load_pipeline_bundle(path, trusted=True), raw=raw,
                    score=score, expected=expected, feature_frame=score_x, columns=columns,
                    source_dtypes=source_dtypes, method=method, fit_x=fit_x, session=session)

    return make


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize("labels", ["numeric_text", "ordinary_text"])
@pytest.mark.parametrize("from_path", [False, True])
def test_saved_input_and_feature_stages_match_independent_predictions(saved_category_model, dtype, labels, from_path):
    case = saved_category_model(dtype, labels)
    bundle, score = case["bundle"], case["score"]
    before = joblib.hash(bundle)
    original = score.copy(deep=True)
    result = predict_from_pipeline(case["path"] if from_path else bundle, score, trusted=from_path)
    np.testing.assert_allclose(result.predictions, case["expected"], atol=1e-10, rtol=1e-10)
    assert result.predictions.index.equals(score.index)
    contract = bundle.schema_contract
    assert set(case["columns"]).issubset(contract.columns)
    for column in case["columns"]:
        assert contract.dtype_families[column] in {"string", "categorical"}
    for column in bundle.encode_plan.feature_names_:
        assert column in contract.encoded_numeric_columns
    pd.testing.assert_frame_equal(score, original)
    assert joblib.hash(bundle) == before
    # Prepared features still use a numeric contract, even under the same names.
    features_result = predict_from_pipeline(bundle, case["feature_frame"], apply_plans=False)
    np.testing.assert_allclose(features_result.predictions, case["expected"], atol=1e-10)
    assert bundle.encode_plan.to_dict()["input_dtypes_"] == case["source_dtypes"]
    json.dumps(bundle.encode_plan.to_dict(), allow_nan=False)


@pytest.mark.parametrize("as_dict", [False, True])
def test_plan_input_derivation_keeps_source_names_that_are_also_generated(saved_category_model, as_dict):
    case = saved_category_model("object", "numeric_text")
    plan = case["bundle"].encode_plan
    plans = {"encode_plan": plan.to_dict() if as_dict else plan}
    assert input_columns_from_plans(plans) == case["columns"]


def test_missing_reused_raw_source_is_detected_at_input_stage(saved_category_model):
    case = saved_category_model("string", "ordinary_text")
    missing = case["columns"][-1]
    score = case["score"].drop(columns=missing)
    validation = validate_score_frame(score, case["bundle"].schema_contract, stage="input")
    assert not validation.ok and missing in validation.missing_columns
    with pytest.raises(ValidationError, match="missing"):
        predict_from_pipeline(case["bundle"], score)


@pytest.mark.parametrize("stage", ["input", "features"])
def test_encoded_feature_contract_does_not_destroy_raw_type_information(stage):
    contract = build_schema_contract(schema={"fields": [
        {"name": "category", "dtype": "string", "nullable": False}]},
        roles={"category": "feature"}, feature_columns=["category"], target_column=None,
        input_columns=["category"], encoded_numeric_columns=["category"])
    assert contract.dtype_families["category"] == "string"
    frame = pd.DataFrame({"category": ["01", "1"]})
    converted, validation = coerce_score_frame(frame, contract, stage=stage)
    assert validation.ok
    if stage == "input":
        assert list(converted.category) == ["01", "1"]
    else:
        assert list(converted.category) == [1, 1]
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"category": ["01", "1"]}))


def test_non_numeric_prepared_features_still_refuse_after_raw_contract_repair(saved_category_model):
    case = saved_category_model("object", "ordinary_text")
    broken = case["feature_frame"].copy()
    column = case["columns"][0] if case["method"] == "ordinal" else case["bundle"].encode_plan.feature_names_[0]
    broken[column] = "not-an-encoded-number"
    with pytest.raises(ValidationError, match="wrong-type"):
        predict_from_pipeline(case["bundle"], broken, apply_plans=False)


@pytest.mark.parametrize("unknown", ["001", "unseen"])
def test_saved_replay_keeps_unknown_label_semantics_without_refitting(saved_category_model, unknown):
    case = saved_category_model("object", "numeric_text")
    score = case["score"].copy()
    score["category"] = unknown
    expected_x = features(case["raw"].iloc[:12], score, case["method"], case["columns"])
    expected = ridge_oracle(case["fit_x"], case["raw"].y.iloc[:12], expected_x)
    bundle = case["bundle"]
    before = joblib.hash(bundle)
    result = predict_from_pipeline(bundle, score)
    np.testing.assert_allclose(result.predictions, expected, atol=1e-10)
    assert joblib.hash(bundle) == before


def test_older_plan_without_raw_type_metadata_can_be_saved_with_explicit_raw_schema(saved_category_model, tmp_path):
    case = saved_category_model("string", "ordinary_text")
    old_plan = case["bundle"].encode_plan
    del old_plan.input_dtypes_
    assert "input_dtypes_" not in old_plan.to_dict()
    raw_dataset = Session.ingest(case["raw"]).dataset
    path = tmp_path / "legacy-with-known-raw-schema"
    save_pipeline_bundle(path, fit_result=case["bundle"].fit_result, encode_plan=old_plan,
                         dataset_schema=raw_dataset.schema.to_dict(), roles={"y": "target"})
    loaded = load_pipeline_bundle(path, trusted=True)
    assert "input_dtypes_" not in loaded.encode_plan.to_dict()
    result = predict_from_pipeline(loaded, case["score"])
    np.testing.assert_allclose(result.predictions, case["expected"], atol=1e-10)


@pytest.mark.parametrize("method", ["ordinal", "onehot", "infrequent", "target"])
@pytest.mark.parametrize("dtype", ["object", "string", "category"])
@pytest.mark.parametrize("as_dataset", [False, True])
def test_raw_null_category_representations_match_fit_time_semantics(method, dtype, as_dataset, tmp_path):
    raw = raw_data(dtype, "numeric_text")
    for pos, value in enumerate([None, np.nan, pd.NA]):
        raw.iloc[pos, raw.columns.get_loc("category")] = value
    for pos, value in enumerate([None, np.nan, pd.NA, "1"], start=12):
        raw.iloc[pos, raw.columns.get_loc("category")] = value
    score = raw.iloc[12:].drop(columns="y").copy()
    columns = ["category"]
    session = (Session.ingest(raw).set_roles({"x": "feature", "category": "feature", "y": "target"})
               .inject_split(train_indices=list(range(12)), test_indices=list(range(12, 16))))
    session.encode(columns=columns, method=method, min_frequency=1,
                   smoothing=0, n_folds=3, random_state=7).fit(
                       make_pipeline(SimpleImputer(strategy="constant", fill_value=-3), Ridge(alpha=.5)),
                       task="regression")
    expected = session.predict().to_numpy()
    path = tmp_path / "null-categories"
    session.save_pipeline(path, evaluate_partition=None)
    bundle = load_pipeline_bundle(path, trusted=True)
    source = Session.ingest(score).dataset if as_dataset else score
    original = score.copy(deep=True)
    result = predict_from_pipeline(bundle, source)
    np.testing.assert_allclose(result.predictions, expected, atol=1e-10)
    pd.testing.assert_frame_equal(score, original)
    if as_dataset:
        pd.testing.assert_frame_equal(source.frame, original)


def test_checkpoint_preserves_raw_type_metadata_before_refitting_and_export(saved_category_model, tmp_path):
    case = saved_category_model("string", "numeric_text")
    checkpoint = tmp_path / "checkpoint"
    case["session"].checkpoint_save(checkpoint)
    restored = Session.checkpoint_load(checkpoint, trusted=True)
    assert restored.fit_result is None
    assert restored.encode_plan.input_dtypes_ == case["source_dtypes"]
    restored.fit(Ridge(alpha=.5), task="regression")
    path = tmp_path / "exported-after-resume"
    restored.save_pipeline(path, evaluate_partition=None)
    result = predict_from_pipeline(path, case["score"], trusted=True)
    np.testing.assert_allclose(result.predictions, case["expected"], atol=1e-10)


@pytest.mark.parametrize("as_dataset", [False, True])
def test_native_category_policy_cannot_hide_malformed_numeric_input(saved_category_model, as_dataset):
    case = saved_category_model("object", "numeric_text")
    score = case["score"].copy()
    score["x"] = ["1", "not-a-number", "3", "4"]
    original = score.copy(deep=True)
    source = Session.ingest(score).dataset if as_dataset else score
    before = joblib.hash(case["bundle"])
    with pytest.raises(ValidationError, match="x"):
        predict_from_pipeline(case["bundle"], source)
    assert joblib.hash(case["bundle"]) == before
    pd.testing.assert_frame_equal(score, original)


@pytest.mark.parametrize("stage", ["input", "features"])
@pytest.mark.parametrize("values,valid", [(["01", "1"], True), (["1", "not-a-number"], False)])
def test_native_string_policy_does_not_override_numeric_contracts(stage, values, valid):
    contract = build_schema_contract(schema={"fields": [
        {"name": "x", "dtype": "float64", "nullable": False}]},
        roles={"x": "feature"}, feature_columns=["x"], target_column=None,
        input_columns=["x"], encoded_numeric_columns=["x"])
    raw = pd.DataFrame({"x": values})
    result, validation = coerce_score_frame(raw, contract, stage=stage, stringify_columns=["x"])
    assert validation.ok is valid
    if valid:
        np.testing.assert_array_equal(result.x, [1, 1])
        assert pd.api.types.is_numeric_dtype(result.x)
    else:
        pd.testing.assert_frame_equal(result, raw)
