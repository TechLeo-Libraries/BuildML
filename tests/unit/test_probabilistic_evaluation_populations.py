"""Uncertainty reports count only labeled evaluation rows without learning from them."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.probabilistic.evaluate import evaluate_probabilistic
from buildml.probabilistic.predict import predict_interval


def _fitted(task):
    rng = np.random.default_rng(276)
    x = rng.normal(size=180)
    z = rng.normal(size=180)
    outcome = 1.8 * x - 0.4 * z + rng.normal(0, 0.4, len(x))
    if task == "classification":
        outcome = np.where(outcome > 0, "approved", "declined")
    frame = pd.DataFrame({"x": x, "z": z, "outcome": outcome}, index=np.arange(8000, 8180))
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "outcome": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=14)
    )
    session.probabilistic.fit(
        backend="native",
        task=task,
        conformal=True,
        alpha=0.2,
        estimator="bayesian_ridge" if task == "regression" else "gaussian_nb",
    )
    return session


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("population", ["partially_labeled", "unlabeled", "empty"])
def test_frozen_evaluation_handles_label_missingness_without_changing_population(task, population):
    session = _fitted(task)
    plan = session.probabilistic_plan
    frame = session.to_pandas().copy()
    partition = "test"
    split = session.split_plan
    if population == "empty":
        frame = frame.iloc[:0].copy()
        partition, split = "all", None
    else:
        positions = list(split.test_indices)
        missing = positions[::3] if population == "partially_labeled" else positions
        frame.iloc[missing, frame.columns.get_loc("outcome")] = np.nan
    original = frame.copy()
    dataset = Dataset.from_pandas(frame, roles=session.dataset.roles)
    before = joblib.hash(plan)
    result = evaluate_probabilistic(dataset, plan, split, partition=partition)
    assert any("independence" in text for text in result.disclosures)
    if population != "partially_labeled":
        assert result.n_rows == 0 and result.metrics == {}
        assert result.interval_coverage is None
        assert any("empty" in text or "No labeled" in text for text in result.warnings)
    else:
        evaluated = frame.iloc[list(split.test_indices)]
        labeled = evaluated.outcome.notna().to_numpy()
        kept = evaluated.loc[evaluated.outcome.notna()]
        x = kept[list(plan.columns)].to_numpy()
        assert result.n_rows == len(kept)
        assert any("null targets" in text for text in result.warnings)
        if task == "regression":
            predicted = plan.estimator_.predict(x)
            errors = kept.outcome.to_numpy(dtype=float) - predicted
            assert result.metrics["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)))
            assert result.metrics["mae"] == pytest.approx(np.mean(np.abs(errors)))
            bounds = predict_interval(dataset, plan, split, partition="test")
            lower, upper = np.asarray(bounds.lower)[labeled], np.asarray(bounds.upper)[labeled]
            covered = (kept.outcome.to_numpy() >= lower) & (kept.outcome.to_numpy() <= upper)
            assert result.interval_coverage == pytest.approx(covered.mean())
            assert result.mean_interval_width == pytest.approx(np.mean(upper - lower))
        else:
            probabilities = plan.estimator_.predict_proba(x)
            labels = np.asarray(plan.classes_)
            assert labels.tolist() == ["approved", "declined"]
            actual = kept.outcome.to_numpy()
            predicted = labels[probabilities.argmax(axis=1)]
            assert result.metrics["accuracy"] == pytest.approx(np.mean(predicted == actual))
            assert result.metrics["brier"] == pytest.approx(
                np.mean(((actual == "declined").astype(float) - probabilities[:, 1]) ** 2)
            )
            class_positions = {label: i for i, label in enumerate(labels)}
            assigned = probabilities[
                np.arange(len(kept)), [class_positions[label] for label in actual]
            ]
            assert result.metrics["nll"] == pytest.approx(-np.log(assigned).mean())
    assert joblib.hash(plan) == before
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), original)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("column", ["x", "outcome"])
def test_incompatible_evaluation_schema_is_refused_without_mutating_the_plan(task, column):
    session = _fitted(task)
    plan = session.probabilistic_plan
    before = joblib.hash(plan)
    dataset = Dataset.from_pandas(session.to_pandas().drop(columns=column))
    with pytest.raises(ValidationError, match=column):
        evaluate_probabilistic(dataset, plan, session.split_plan, partition="test")
    assert joblib.hash(plan) == before
