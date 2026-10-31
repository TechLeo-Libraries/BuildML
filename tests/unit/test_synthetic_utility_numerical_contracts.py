"""TSTR task selection and scores follow frozen training information."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.synthetic.evaluate import evaluate_synthetic


def _frame(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 2))
    y = 3 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(scale=0.15, size=len(x))
    x[::11, 1] = np.nan
    return pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": y}, index=np.arange(900, 1020))


def _session(frame, test_size=0.1):
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=test_size, validation_size=0.2, random_state=17)
    )
    session.synthetic.fit(method="bootstrap", smooth_sigma=0, random_state=4)
    return session


def _design(train, test):
    a, b = train[["x", "z"]].to_numpy(), test[["x", "z"]].to_numpy()
    medians = np.nanmedian(a, axis=0)
    a, b = np.where(np.isnan(a), medians, a), np.where(np.isnan(b), medians, b)
    mean, scale = a.mean(axis=0), a.std(axis=0)
    return (a - mean) / scale, (b - mean) / scale


def _ridge_r2(train, test):
    a, b = _design(train, test)
    y = train.y.to_numpy()
    coefficients = np.linalg.solve(a.T @ a + np.eye(a.shape[1]), a.T @ (y - y.mean()))
    predicted = b @ coefficients + y.mean()
    truth = test.y.to_numpy()
    return 1 - np.square(truth - predicted).sum() / np.square(truth - truth.mean()).sum()


@pytest.mark.parametrize("seed", [3, 29])
@pytest.mark.parametrize("partition", ["validation", "test"])
def test_regression_utility_matches_normal_equations_despite_holdout_cardinality(seed, partition):
    frame = _frame(seed)
    original = _session(frame)
    for poisoned in (False, True):
        changed = frame.copy()
        indices = list(getattr(original.split_plan, f"{partition}_indices"))
        if poisoned:
            changed.iloc[indices, changed.columns.get_loc("y")] = np.arange(len(indices)) % 2
        session = _session(changed)
        train, real = session.partition("train"), session.partition(partition)
        positions = np.random.default_rng(11).integers(0, len(train), size=80)
        synthetic = train.iloc[positions].reset_index(drop=True)
        generator = session.synthesizer_plan.generator_
        before = generator.frame.copy(deep=True)
        sampled = session.synthetic.sample(n=80, random_state=11).frame
        pd.testing.assert_frame_equal(sampled, synthetic)
        actual = session.synthetic.evaluate(
            mode="tstr", partition=partition, n_synthetic=np.int64(80), random_state=11
        )
        assert actual.metrics["task_is_classification"] == 0
        assert actual.metrics["r2"] == pytest.approx(_ridge_r2(synthetic, real), abs=1e-11)
        assert actual.metrics["trtr_r2"] == pytest.approx(_ridge_r2(train, real), abs=1e-11)
        assert actual.metrics["tstr_gap_vs_trtr"] == pytest.approx(
            actual.metrics["trtr_r2"] - actual.metrics["r2"]
        )
        assert any(
            "frozen" in item.lower() and "task" in item.lower() for item in actual.disclosures
        )
        pd.testing.assert_frame_equal(generator.frame, before)
        pd.testing.assert_frame_equal(session.dataset.frame, changed)


@pytest.mark.parametrize("estimator", ["auto", "ridge"])
def test_classification_utility_keeps_training_task_with_new_holdout_classes(estimator):
    frame = _frame(43)
    frame["y"] = (frame.x > 0).astype(int)
    original = _session(frame, test_size=0.3)
    rows = list(original.split_plan.test_indices)
    frame.iloc[rows, frame.columns.get_loc("y")] = np.arange(len(rows))
    session = _session(frame, test_size=0.3)
    train, real = session.partition("train"), session.partition("test")
    synthetic = train.iloc[np.random.default_rng(11).integers(0, len(train), size=80)]
    a, b = _design(synthetic, real)
    model = (
        RidgeClassifier(random_state=11)
        if estimator == "ridge"
        else LogisticRegression(max_iter=400, random_state=11)
    )
    predicted = model.fit(a, synthetic.y.astype(str)).predict(b)
    actual = session.synthetic.evaluate(
        mode="tstr", n_synthetic=80, random_state=11, estimator=estimator
    )
    assert actual.metrics["task_is_classification"] == 1
    assert actual.metrics["accuracy"] == pytest.approx(np.mean(predicted == real.y.astype(str)))
    assert "r2" not in actual.metrics


@pytest.mark.parametrize("n_synthetic", [0, -2, 5.5, True, np.nan, "5"])
@pytest.mark.parametrize("mode", ["fidelity", "tstr"])
def test_explicit_sample_budget_is_a_positive_integer(n_synthetic, mode):
    session = _session(_frame(8))
    with pytest.raises(ValidationError, match="n_synthetic"):
        session.synthetic.evaluate(mode=mode, n_synthetic=n_synthetic, eval_backend="builtin")


def test_missing_frozen_target_metadata_requires_refit_instead_of_holdout_inference():
    session = _session(_frame(8))
    plan = session.synthesizer_plan
    damaged = replace(
        plan, column_specs=tuple(spec for spec in plan.column_specs if spec.name != "y")
    )
    with pytest.raises(ValidationError, match="frozen training schema; refit"):
        evaluate_synthetic(
            session.dataset, session.split_plan, damaged, mode="tstr", n_synthetic=80
        )
