"""Multi-output consumers verify numerical predictions and atomic attachment."""

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.multitask.checkpoint import load_multitask_bundle
from buildml.multitask.predict import predict_multitask


@pytest.fixture(scope="module", params=["regression", "strings", "integers", "booleans"])
def fitted(request):
    rng = np.random.default_rng(864)
    x = rng.normal(size=(100, 2))
    y = np.column_stack([2 * x[:, 0] - 0.3 * x[:, 1], -0.8 * x[:, 0] + 1.6 * x[:, 1]])
    if request.param != "regression":
        vocabulary = {"strings": ["001", "010"], "integers": [2, 10], "booleans": [False, True]}[
            request.param
        ]
        y = np.asarray(vocabulary)[(y > 0).astype(int)]
    frame = pd.DataFrame(
        {"x": x[:, 0], "z": x[:, 1], "a": y[:, 0], "b": y[:, 1]}, index=[7, 7, 3, 9] * 25
    )
    s = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "a": "target", "b": "target"}
    )
    s.split(test_size=0.2, validation_size=0.2, random_state=6)
    task = "regression" if request.param == "regression" else "classification"
    s.multitask.fit(
        backend="sklearn",
        method="multi_output",
        task=task,
        base_estimator="ridge" if task == "regression" else "logistic_regression",
        prefer_reduce_components=False,
    )
    return s


def expected_predictions(s, part):
    plan = s.multitask_plan
    x = part[list(plan.columns)].to_numpy()
    out = {}
    for target, est in zip(plan.target_columns, plan.estimator_.estimators_, strict=True):
        if plan.task == "regression":
            train = s.partition("train")
            xt = train[list(plan.columns)].to_numpy()
            centered = xt - xt.mean(axis=0)
            y = train[target].to_numpy()
            coef = np.linalg.solve(centered.T @ centered + np.eye(2), centered.T @ (y - y.mean()))
            out[target] = (x - xt.mean(axis=0)) @ coef + y.mean()
        else:
            logits = (x @ est.coef_.T + est.intercept_).ravel()
            labels = plan.label_encoders_[target].classes_
            out[target] = labels[(logits >= 0).astype(int)]
    return out


@pytest.mark.parametrize("partition", ["train", "validation", "test", "all"])
@pytest.mark.parametrize("reload", [False, True])
def test_multitarget_prediction_matches_matrix_oracle_and_bundle(
    fitted, partition, reload, tmp_path
):
    s = fitted
    plan = s.multitask_plan
    if reload:
        path = s.multitask.save_bundle(tmp_path / "bundle")
        plan = load_multitask_bundle(path, trusted=True)
    original = s.to_pandas().copy(deep=True)
    before = joblib.hash(plan)
    part = original if partition == "all" else s.partition(partition)
    wanted = expected_predictions(s, part)
    attached, result = predict_multitask(s.dataset, plan, s.split_plan, partition=partition)
    assert attached is None and not result.attached
    assert result.n_rows == len(part)
    for target, values in wanted.items():
        if plan.task == "regression":
            np.testing.assert_allclose(result.predictions[target], values, atol=1e-12)
        else:
            np.testing.assert_array_equal(result.predictions[target], values)
    assert joblib.hash(plan) == before
    pd.testing.assert_frame_equal(s.to_pandas(), original)


@pytest.mark.parametrize("prefix", ["consumer", "result"])
@pytest.mark.parametrize("reload", [False, True])
def test_prediction_attachment_preserves_duplicate_rows_and_target_identity(
    fitted, prefix, reload, tmp_path
):
    s = fitted
    plan = s.multitask_plan
    if reload:
        plan = load_multitask_bundle(s.multitask.save_bundle(tmp_path / "bundle"), trusted=True)
    before = joblib.hash(plan)
    original = s.to_pandas().copy(deep=True)
    attached, result = predict_multitask(
        s.dataset, plan, None, partition="all", attach=True, prediction_prefix=prefix
    )
    wanted = expected_predictions(s, original)
    assert result.attached
    assert attached.frame.index.equals(original.index)
    pd.testing.assert_frame_equal(attached.frame[original.columns], original)
    for target, values in wanted.items():
        if plan.task == "regression":
            np.testing.assert_allclose(attached.frame[f"{prefix}_{target}"], values, atol=1e-12)
        else:
            np.testing.assert_array_equal(attached.frame[f"{prefix}_{target}"], values)
        assert str(attached.roles[f"{prefix}_{target}"].value) == "feature"
    assert joblib.hash(plan) == before
    pd.testing.assert_frame_equal(s.to_pandas(), original)


@pytest.mark.parametrize(
    "failure",
    ["holdout_attach", "first_collision", "second_collision", "missing_feature", "missing_split"],
)
def test_attachment_failures_preserve_entire_dataset_and_fitted_plan(fitted, failure):
    s = fitted
    frame = s.to_pandas().copy(deep=True)
    if failure.endswith("collision"):
        frame["output_a" if failure == "first_collision" else "output_b"] = 123
    if failure == "missing_feature":
        frame = frame.drop(columns="x")
    dataset = Dataset.from_pandas(
        frame, roles={k: v for k, v in s.dataset.roles.items() if k in frame}
    )
    before = joblib.hash(s.multitask_plan)
    with pytest.raises(ValidationError):
        predict_multitask(
            dataset,
            s.multitask_plan,
            None if failure == "missing_split" else s.split_plan,
            partition="test" if failure in {"holdout_attach", "missing_split"} else "all",
            attach=True,
            prediction_prefix="output",
        )
    pd.testing.assert_frame_equal(dataset.frame, frame)
    assert joblib.hash(s.multitask_plan) == before
