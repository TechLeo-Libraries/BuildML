"""Saved inference replays frozen statistics and preserves raw record identity."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline


@pytest.fixture(params=["regression", "classification"])
def frozen_bundle(request, tmp_path):
    rng = np.random.default_rng(562)
    x = rng.normal(size=(70, 2))
    y = 1.7 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(0, 0.3, len(x))
    if request.param == "classification":
        y = np.where(y > 0, "approved", "declined")
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": y})
    frame.loc[::7, "x"] = np.nan
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
    s.split(test_size=0.2, validation_size=0.2, random_state=4)
    s.impute(columns=["x", "z"], strategy="median").scale(columns=["x", "z"])
    s.fit(
        Ridge(alpha=0.6) if request.param == "regression" else LogisticRegression(C=0.8),
        task=request.param,
    )
    path = tmp_path / "pipeline"
    s.save_pipeline(path, evaluate_partition=None)
    return path, load_pipeline_bundle(path, trusted=True)


@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("from_path", [False, True])
@pytest.mark.parametrize("apply_plans", [False, True])
@pytest.mark.parametrize("legacy_contract", [False, True])
def test_saved_pipeline_replay_matches_manual_transform_and_affine_model(
    frozen_bundle, as_dataset, from_path, apply_plans, legacy_contract
):
    path, bundle = frozen_bundle
    raw = pd.DataFrame(
        {"z": [0.4, -3.0, 2.0, 8.0], "x": [None, -2.0, 4.0, None], "extra": [999, 999, 999, 999]},
        index=["same", "same", "later", "first"],
    )
    filled = raw[["x", "z"]].fillna(bundle.impute_plan.statistics_)
    scaler = bundle.scale_plan.scaler
    transformed = (filled.to_numpy() - scaler.mean_) / scaler.scale_
    model = bundle.fit_result.estimator
    if hasattr(model, "named_steps"):
        model = model.named_steps["model"]
    if bundle.fit_result.task == "regression":
        expected = transformed @ model.coef_ + model.intercept_
    else:
        logits = (transformed @ model.coef_.T + model.intercept_).ravel()
        positive = 1 / (1 + np.exp(-logits))
        expected = model.classes_[(positive >= 0.5).astype(int)]
    incoming = (
        raw.copy()
        if apply_plans
        else pd.DataFrame(transformed, columns=["x", "z"], index=raw.index)
    )
    if legacy_contract:
        (path / "schema_contract.json").unlink()
        bundle = load_pipeline_bundle(path, trusted=True)
    original = incoming.copy(deep=True)
    source = Session.ingest(incoming).dataset if as_dataset else incoming
    before = joblib.hash(bundle)
    actual = predict_from_pipeline(
        path if from_path else bundle,
        source,
        apply_plans=apply_plans,
        return_proba=True,
        trusted=from_path,
    )
    if bundle.fit_result.task == "regression":
        np.testing.assert_allclose(actual.predictions, expected, atol=1e-12)
        assert actual.probabilities is None
        assert any("no predict_proba" in w for w in actual.warnings)
    else:
        np.testing.assert_array_equal(actual.predictions, expected)
        np.testing.assert_allclose(
            actual.probabilities, np.column_stack([1 - positive, positive]), atol=1e-12
        )
        assert list(actual.probabilities) == [f"proba_{c}" for c in model.classes_]
        assert actual.probabilities.index.equals(incoming.index)
    assert actual.predictions.index.equals(incoming.index)
    assert actual.n_rows == len(raw)
    assert actual.feature_columns == ("x", "z")
    assert (actual.apply_result is not None) == apply_plans
    assert joblib.hash(bundle) == before
    pd.testing.assert_frame_equal(incoming, original)
    if as_dataset:
        pd.testing.assert_frame_equal(source.frame, original)


@pytest.mark.parametrize("from_path", [False, True])
@pytest.mark.parametrize("apply_plans", [False, True])
@pytest.mark.parametrize("invalid", ["missing_feature", "bad_numeric", "empty", "not_a_frame"])
def test_saved_pipeline_refuses_bad_requests_without_changing_fitted_state(
    frozen_bundle, from_path, apply_plans, invalid
):
    from buildml.core.errors import ValidationError

    path, bundle = frozen_bundle
    frame = pd.DataFrame({"x": [1.0, 2.0], "z": [-1.0, 3.0]})
    if invalid == "missing_feature":
        frame = frame.drop(columns="x")
    elif invalid == "bad_numeric":
        frame["x"] = ["bad", "1.0"]
    elif invalid == "empty":
        frame = frame.iloc[:0]
    original = frame.copy(deep=True)
    before = joblib.hash(bundle)
    with pytest.raises(ValidationError):
        predict_from_pipeline(
            path if from_path else bundle,
            [[1.0, 2.0]] if invalid == "not_a_frame" else frame,
            trusted=from_path,
            apply_plans=apply_plans,
        )
    assert joblib.hash(bundle) == before
    pd.testing.assert_frame_equal(frame, original)
