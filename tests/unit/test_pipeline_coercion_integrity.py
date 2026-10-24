"""Score-time coercion preserves values or refuses the incoming column."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline
from buildml.pipeline.contract import SchemaContract, coerce_score_frame, raise_for_contract


def contract(family):
    return SchemaContract(columns=("x",), feature_columns=("x",), dtype_families={"x": family})


@pytest.mark.parametrize("stage", ["input", "features"])
@pytest.mark.parametrize("family,values", [("numeric", ["1", "invalid"]), ("numeric", ["invalid", "bad"]), ("bool", ["0", "invalid"]), ("bool", ["invalid", "bad"]), ("bool", [0, 2]), ("datetime", ["2026-01-01", "invalid"]), ("datetime", ["invalid", "bad"])])
def test_invalid_values_are_not_silently_replaced_with_null(family, values, stage):
    frame = pd.DataFrame({"x": values}, index=["row-a", "row-a"])
    original = frame.copy(deep=True)
    coerced, result = coerce_score_frame(frame, contract(family), stage=stage)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(coerced, original)
    assert not result.ok
    assert "x" not in result.coerced_columns
    with pytest.raises(ValidationError, match="x"):
        raise_for_contract(result)


@pytest.mark.parametrize("stage", ["input", "features"])
@pytest.mark.parametrize("family,values,expected", [("numeric", ["1.5", None, "2"], [1.5, np.nan, 2.]), ("bool", ["0", None, "1"], [False, pd.NA, True]), ("datetime", ["2026-01-01", None, "2026-01-03"], list(pd.to_datetime(["2026-01-01", None, "2026-01-03"]))), ("string", [1, None, 2], ["1", pd.NA, "2"]), ("categorical", [1, None, 2], ["1", pd.NA, "2"])])
def test_lossless_coercion_preserves_rows_and_existing_nulls(family, values, expected, stage):
    frame = pd.DataFrame({"x": values}, index=[7, 7, 3], dtype=object)
    out, result = coerce_score_frame(frame, contract(family), stage=stage)
    assert result.ok
    assert out.index.equals(frame.index)
    assert out.x.isna().tolist() == frame.x.isna().tolist()
    for actual, wanted in zip(out.x, expected, strict=True):
        if pd.isna(wanted):
            assert pd.isna(actual)
        else:
            assert actual == wanted


@pytest.fixture
def saved(tmp_path):
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 40)})
    frame["y"] = frame.x ** 2 + frame.x
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(test_size=.2, random_state=3).fit(HistGradientBoostingRegressor(max_iter=3, min_samples_leaf=2, random_state=4), task="regression")
    path = tmp_path / "scoring"
    session.save_pipeline(path, evaluate_partition=None)
    return path


@pytest.mark.parametrize("values", [["1.0", "bad"], ["bad", "invalid"]])
def test_model_that_accepts_nulls_cannot_hide_bad_scoring_values(saved, values):
    bundle = load_pipeline_bundle(saved, trusted=True)
    # This model can score actual missing values; the contract must reject
    # malformed supplied values before they become a valid missing-data input.
    assert np.isfinite(bundle.fit_result.estimator.predict(pd.DataFrame({"x": [np.nan]}))).all()
    with pytest.raises(ValidationError, match="schema contract"):
        predict_from_pipeline(saved, pd.DataFrame({"x": values}), apply_plans=False, trusted=True)


def test_persisted_pipeline_numeric_coercion_matches_estimator_and_index(saved):
    incoming = pd.DataFrame({"x": ["1.0", None, "-0.25"]}, index=["duplicate", "duplicate", "other"])
    original = incoming.copy(deep=True)
    bundle = load_pipeline_bundle(saved, trusted=True)
    expected = bundle.fit_result.estimator.predict(pd.DataFrame({"x": [1., np.nan, -.25]}, index=incoming.index))
    actual = predict_from_pipeline(saved, incoming, apply_plans=False, trusted=True)
    np.testing.assert_allclose(actual.predictions, expected)
    assert actual.predictions.index.equals(incoming.index)
    pd.testing.assert_frame_equal(incoming, original)
