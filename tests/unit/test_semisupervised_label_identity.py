"""Semi-supervised consumers preserve labels instead of guessing numeric types."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.checkpoint import load_semisupervised_bundle, save_semisupervised_bundle
from buildml.semisupervised.evaluate import evaluate_semisupervised
from buildml.semisupervised.fit import fit_semisupervised
from buildml.semisupervised.predict import predict_semisupervised


def workflow(pair):
    rng = np.random.default_rng(72)
    x = np.r_[rng.normal(-2, 0.3, 30), rng.normal(2, 0.3, 30)]
    labels = pd.Series([pair[0]] * 30 + [pair[1]] * 30, dtype=object)
    labels.iloc[2::7] = None
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=60), "y": labels})
    return Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"}).split(test_size=.2, validation_size=.2, random_state=7)


@pytest.mark.parametrize("pair", [("001", "002"), ("1.0", "2.0"), (-3, 7), (1.25, 2.75), (False, True)])
@pytest.mark.parametrize("method", ["label_propagation", "label_spreading", "self_training"])
def test_raw_labels_metrics_and_bundle_identity(tmp_path, pair, method):
    session = workflow(pair)
    plan, fit = fit_semisupervised(session.dataset, session.split_plan, method=method, kernel="rbf")
    assert plan.classes_ == pair
    assert all(type(label) is type(pair[0]) for label in plan.classes_)
    train = frame_for_partition(session.dataset, session.split_plan, "train")
    test = frame_for_partition(session.dataset, session.split_plan, "test")
    codes = np.array([-1 if pd.isna(value) else pair.index(value) for value in train.y])
    reference = {"label_propagation": LabelPropagation(kernel="rbf"),
                 "label_spreading": LabelSpreading(kernel="rbf", alpha=.2),
                 "self_training": SelfTrainingClassifier(LogisticRegression(max_iter=500, random_state=0))}[method]
    reference.fit(train[["x", "z"]].to_numpy(), codes)
    expected_codes = reference.predict(test[["x", "z"]].to_numpy())
    expected = tuple(pair[int(code)] for code in expected_codes)
    _, predictions = predict_semisupervised(session.dataset, plan, session.split_plan, partition="test")
    assert predictions.predictions == expected
    assert all(type(label) is type(pair[0]) for label in predictions.predictions)
    evaluation = evaluate_semisupervised(session.dataset, plan, session.split_plan, partition="test")
    labeled = test.y.notna().to_numpy()
    true_codes = np.array([pair.index(value) for value in test.y[labeled]])
    assert evaluation.metrics["accuracy"] == pytest.approx(np.mean(expected_codes[labeled] == true_codes))
    json.dumps(plan.to_dict())
    json.dumps(fit.to_dict())
    path = save_semisupervised_bundle(tmp_path / "bundle", plan, fit_result=fit, eval_result=evaluation)
    restored = load_semisupervised_bundle(path, trusted=True)  # Benign fixture written above.
    _, actual = predict_semisupervised(session.dataset, restored, session.split_plan, partition="test")
    assert actual.predictions == expected
    assert all(type(label) is type(pair[0]) for label in actual.predictions)


def test_mixed_labels_cannot_silently_collapse():
    frame = pd.DataFrame({"x": np.arange(60), "y": pd.Series([1, "1", 2] * 20, dtype=object)})
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(test_size=.2, random_state=7)
    with pytest.raises(ValidationError, match="label|class|consistent"):
        fit_semisupervised(session.dataset, session.split_plan)


def test_holdout_numeric_labels_do_not_match_trained_strings():
    session = workflow(("1.0", "2.0"))
    plan, _ = fit_semisupervised(session.dataset, session.split_plan, kernel="rbf")
    changed = session.to_pandas().copy()
    holdout = list(session.split_plan.test_indices)
    changed.loc[holdout, "y"] = [float(value) if pd.notna(value) else None for value in changed.loc[holdout, "y"]]
    dataset = Dataset.from_transformed(session.dataset, changed, schema=schema_from_dataframe(changed), roles=dict(session.dataset.roles))
    with pytest.raises(ValidationError, match="class|label"):
        evaluate_semisupervised(dataset, plan, session.split_plan, partition="test")


@pytest.mark.parametrize("pair", [(-3, 7), (1.25, 2.75), ("bad", "good")])
def test_legacy_string_encoder_retains_compatibility_with_refit_warning(tmp_path, pair):
    session = workflow(pair)
    plan, _ = fit_semisupervised(session.dataset, session.split_plan, kernel="rbf")
    plan.label_encoder_ = LabelEncoder().fit([str(value) for value in pair])
    plan.classes_ = tuple(str(value) for value in pair)
    path = save_semisupervised_bundle(tmp_path / "legacy", plan)
    restored = load_semisupervised_bundle(path, trusted=True)
    _, predictions = predict_semisupervised(session.dataset, restored, session.split_plan, partition="test")
    assert set(predictions.predictions).issubset(set(pair))
    assert all(type(value) is type(pair[0]) for value in predictions.predictions)
    evaluation = evaluate_semisupervised(session.dataset, restored, session.split_plan, partition="test")
    assert any("legacy" in warning.lower() for warning in evaluation.warnings)
