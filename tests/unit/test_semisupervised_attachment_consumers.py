"""Frozen semi-supervised predictions preserve population, targets and consumers."""

import copy
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole


def _frame(labels):
    rng = np.random.default_rng(851)
    positions = np.arange(90)
    x = np.where(positions % 2, 1.5, -1.5) + rng.normal(0, .35, 90)
    truth = pd.Series([labels[int(i % 2)] for i in positions], dtype=object)
    truth.iloc[::7] = None
    frame = pd.DataFrame({"x": x, "z": rng.normal(0, .4, 90), "y": truth, "row_id": positions})
    frame.index = pd.Index((positions * 7) % 13, name="duplicate_label")
    return frame


def _session(frame, *, split=True):
    session = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "y": "target", "row_id": "id"})
    if split:
        session.inject_split(train_indices=list(range(60)),
                             validation_indices=list(range(60, 75)),
                             test_indices=list(range(75, 90)))
    return session


@pytest.fixture(scope="module", params=list(itertools.product(
    ["label_propagation", "label_spreading", "self_training"],
    [("001", "010"), (2, 10), (False, True)])))
def fitted(request):
    method, labels = request.param
    frame = _frame(labels)
    source = _session(frame)
    source.semisupervised.fit(method=method, kernel="rbf", max_self_train_iter=8,
                              prefer_reduce_components=False)
    return source, frame


def _receiver(fitted, path, *, split=True):
    source, frame = fitted
    bundle = source.semisupervised.save_bundle(path)
    receiver = _session(frame, split=split)
    receiver.semisupervised.load_bundle(bundle, trusted=True)
    assert receiver.semisupervised_plan.to_dict() == source.semisupervised_plan.to_dict()
    return receiver


def _oracle(plan, frame):
    """Scalar frozen logits or RBF weights; no estimator prediction calls."""
    model = plan.estimator_
    result = []
    for values in frame[list(plan.columns)].itertuples(index=False, name=None):
        if plan.method == "self_training":
            estimator = model.estimator_
            logit = float(estimator.intercept_[0]) + sum(
                float(a)*float(b) for a, b in zip(values, estimator.coef_[0]))
            code = int(logit > 0)
        else:
            scores = [0., 0.]
            for training, distribution in zip(model.X_, model.label_distributions_):
                squared_distance = sum((float(a)-float(b))**2 for a, b in zip(values, training))
                weight = float(np.exp(-model.gamma * squared_distance))
                for index in range(2):
                    scores[index] += weight*float(distribution[index])
            code = int(scores[1] > scores[0])
        result.append(plan.classes_[code])
    return tuple(result)


def _check_readonly(session, partition):
    before = session.to_pandas()
    dataset = session.dataset
    roles = dict(dataset.roles)
    plan = session.semisupervised_plan
    fingerprint = joblib.hash(plan)
    part = before if partition == "all" else session.partition(partition)
    expected = _oracle(plan, part)
    result = session.semisupervised.predict(partition=partition)
    assert result.predictions == expected
    assert result.n_rows == len(part) and result.partition == partition and not result.attached
    assert all(type(value) is type(plan.classes_[0]) for value in result.predictions)
    assert session.dataset is dataset and session.dataset.roles == roles
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    assert joblib.hash(plan) == fingerprint
    json.dumps(result.to_dict(), allow_nan=False)
    return result


@pytest.mark.parametrize("partition", ["train", "validation", "test", "all"])
def test_partition_and_all_predictions_follow_positions_and_scalar_model(fitted, partition, tmp_path):
    receiver = _receiver(fitted, tmp_path / "owned")
    _check_readonly(receiver, partition)


def test_loaded_plan_predicts_all_without_a_split(fitted, tmp_path):
    receiver = _receiver(fitted, tmp_path / "owned", split=False)
    assert receiver.split_plan is None
    _check_readonly(receiver, "all")


def _check_attachment(receiver, column, destination):
    before = receiver.to_pandas()
    original_dataset = receiver.dataset
    roles = dict(original_dataset.roles)
    split = receiver.split_plan
    plan = receiver.semisupervised_plan
    fingerprint = joblib.hash(plan)
    expected = _oracle(plan, before)
    result = receiver.semisupervised.predict(partition="all", attach=True,
                                            prediction_column=column)
    assert result.attached and result.predictions == expected and result.n_rows == len(before)
    assert result.prediction_column == column
    assert receiver.dataset is not original_dataset
    assert receiver.split_plan is split
    assert receiver.dataset.roles == {**roles, column: ColumnRole.FEATURE}
    actual = receiver.to_pandas()
    pd.testing.assert_frame_equal(actual[before.columns], before)
    pd.testing.assert_frame_equal(original_dataset.to_pandas(), before)
    assert tuple(actual[column]) == expected
    assert all(type(value) is type(plan.classes_[0]) for value in actual[column])
    assert joblib.hash(plan) == fingerprint
    json.dumps(result.to_dict(), allow_nan=False)
    # A real data consumer must see each row, prediction, target and missing value.
    output = receiver.to_parquet(destination)
    restored = pd.read_parquet(output)
    # Parquet omits the index and normalizes object-backed nullable storage.
    # Compare logical values independently, retaining strict prediction types.
    assert restored.columns.tolist() == actual.columns.tolist()
    assert len(restored) == len(actual)
    assert restored.index.equals(pd.RangeIndex(len(actual)))
    for name in actual.columns:
        for value, original in zip(restored[name], actual[name]):
            if pd.isna(original):
                assert pd.isna(value)
            else:
                assert not pd.isna(value) and value == original
                if isinstance(original, str):
                    assert isinstance(value, str)
    assert tuple(restored[column]) == expected
    assert all(type(value) is type(plan.classes_[0]) for value in restored[column])
    assert _check_readonly(receiver, "all").predictions == expected
    return original_dataset


@pytest.mark.parametrize("column", ["predicted", "prediction λ"])
def test_attachment_and_parquet_keep_raw_labels_targets_roles_and_split(fitted, column, tmp_path):
    receiver = _receiver(fitted, tmp_path / "owned")
    _check_attachment(receiver, column, tmp_path / "predictions.parquet")
    # Added feature roles cannot alter the frozen plan's original feature list.
    assert receiver.semisupervised_plan.columns == ("x", "z")
    _check_readonly(receiver, "test")


def _refuses_atomically(receiver, *, match, **kwargs):
    before = receiver.to_pandas()
    prior = receiver.semisupervised.predict(partition="all")
    dataset = receiver.dataset
    split = receiver.split_plan
    plan = receiver.semisupervised_plan
    fingerprint = joblib.hash(plan)
    roles = dict(dataset.roles)
    history = copy.deepcopy(receiver._history)
    with pytest.raises(ValidationError, match=match):
        receiver.semisupervised.predict(**kwargs)
    assert receiver.dataset is dataset and receiver.split_plan is split
    assert receiver._semisupervised_predict_result is prior
    assert receiver._history == history and receiver.dataset.roles == roles
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(receiver.to_pandas(), before)


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_partial_attachment_cannot_replace_working_dataset_or_result(fitted, partition, tmp_path):
    receiver = _receiver(fitted, tmp_path / "owned")
    _refuses_atomically(receiver, match="all", partition=partition, attach=True)


@pytest.mark.parametrize("column", ["x", "y", "row_id"])
def test_attachment_cannot_overwrite_features_targets_or_identifiers(fitted, column, tmp_path):
    receiver = _receiver(fitted, tmp_path / "owned")
    _refuses_atomically(receiver, match="already exists", partition="all", attach=True,
                        prediction_column=column)


def test_holdout_prediction_without_split_refuses_atomically(fitted, tmp_path):
    receiver = _receiver(fitted, tmp_path / "owned", split=False)
    _refuses_atomically(receiver, match="SplitPlan", partition="test")


def test_loaded_plan_refuses_missing_feature_without_dataset_mutation(fitted, tmp_path):
    source, frame = fitted
    bundle = source.semisupervised.save_bundle(tmp_path / "owned")
    receiver = Session.ingest(frame.drop(columns="x"))
    receiver.semisupervised.load_bundle(bundle, trusted=True)
    dataset = receiver.dataset
    history = copy.deepcopy(receiver._history)
    with pytest.raises(ValidationError, match="Missing feature.*x"):
        receiver.semisupervised.predict(partition="all", attach=True)
    assert receiver.dataset is dataset and receiver._history == history
    assert receiver._semisupervised_predict_result is None
    pd.testing.assert_frame_equal(receiver.to_pandas(), frame.drop(columns="x"))


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
@pytest.mark.parametrize("filtered", [False, True])
def test_native_filtered_all_rows_attach_and_export_the_same_population(fitted, engine, filtered, tmp_path):
    pytest.importorskip(engine)
    source, frame = fitted
    path = tmp_path / "input.parquet"
    frame.to_parquet(path, index=False)
    bundle = source.semisupervised.save_bundle(tmp_path / "owned")
    root = Session.ingest(path, engine=engine, mode="lazy").set_roles(dict(source.dataset.roles))
    owner = root.dataset
    receiver = root
    prior = None
    try:
        if filtered:
            expression = 'row_id % 3 != 0'
            receiver = Session(owner.filter_expr(expression))
        receiver.semisupervised.load_bundle(bundle, trusted=True)
        expected_ids = frame.row_id[frame.row_id % 3 != 0] if filtered else frame.row_id
        prior = _check_attachment(receiver, "predicted", tmp_path / "output.parquet")
        assert receiver.to_pandas().row_id.tolist() == expected_ids.tolist()
        assert receiver.dataset.engine.value == engine and receiver.dataset.has_native
    finally:
        receiver.dataset.close_native()
        if prior is not None:
            prior.close_native()
        owner.close_native()


def test_prediction_without_fitted_plan_is_a_validation_error():
    session = _session(_frame((2, 10)))
    history = copy.deepcopy(session._history)
    with pytest.raises(ValidationError, match="No semi-supervised plan"):
        session.semisupervised.predict(partition="all", attach=True)
    assert session._history == history
