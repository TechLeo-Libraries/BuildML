"""Persisted score contracts must feed coerced values to the actual estimator."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from buildml import Session
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline


@pytest.fixture
def categorical_bundle(tmp_path):
    frame = pd.DataFrame({"category": pd.Series(["1", "2", "3"] * 20, dtype=object),
                          "target": [0, 1, 1] * 20})
    estimator = Pipeline([("encode", OneHotEncoder(sparse_output=False)),
                          ("model", LogisticRegression(C=2, max_iter=500))])
    session = Session.ingest(frame).set_roles({"category": "feature", "target": "target"}).split(test_size=0.2, random_state=4)
    session.fit(estimator, task="classification")
    path = tmp_path / "categorical-pipeline"
    session.save_pipeline(path, evaluate_partition=None)
    return path, load_pipeline_bundle(path, trusted=True)


@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("apply_plans", [False, True])
@pytest.mark.parametrize("from_path", [False, True])
def test_coerced_categories_reach_saved_estimator(categorical_bundle, as_dataset, apply_plans, from_path):
    path, bundle = categorical_bundle
    incoming = pd.DataFrame({"category": [3, 1, 2], "extra": [10, 20, 30]}, index=["same", "same", "other"])
    original = incoming.copy(deep=True)
    source = Session.ingest(incoming).dataset if as_dataset else incoming
    result = predict_from_pipeline(path if from_path else bundle, source,
                                   return_proba=True, apply_plans=apply_plans, trusted=from_path)
    # Manual one-hot construction and logistic probabilities from learned weights.
    categories = bundle.fit_result.estimator.named_steps["encode"].categories_[0]
    matrix = np.array([[str(value) == category for category in categories] for value in incoming.category], dtype=float)
    classifier = bundle.fit_result.estimator.named_steps["model"]
    logits = (matrix @ classifier.coef_.T + classifier.intercept_).ravel()
    positive = 1 / (1 + np.exp(-logits))
    expected_proba = np.column_stack([1 - positive, positive])
    expected_labels = classifier.classes_[(positive >= 0.5).astype(int)]
    np.testing.assert_array_equal(result.predictions.to_numpy(), expected_labels)
    np.testing.assert_allclose(result.probabilities.to_numpy(), expected_proba, atol=1e-12)
    assert result.predictions.index.equals(incoming.index)
    assert result.probabilities.index.equals(incoming.index)
    assert list(result.probabilities.columns) == [f"proba_{label}" for label in classifier.classes_]
    assert "category" in result.contract_validation.coerced_columns
    assert result.apply_result is None
    pd.testing.assert_frame_equal(incoming, original)
    if as_dataset:
        pd.testing.assert_frame_equal(source.frame, original)
