"""Frozen federated scoring preserves populations and reports their provenance."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.federated.evaluate import evaluate_federated


def _fitted(task):
    rng = np.random.default_rng(163)
    x = rng.normal(size=(180, 2))
    y = 1.2 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(0, 0.2, len(x))
    if task == "classification":
        y = np.where(y > 0, "accepted", "rejected")
    frame = pd.DataFrame(
        {"x": x[:, 0], "z": x[:, 1], "y": y, "client": np.arange(len(x)) % 6},
        index=np.arange(7000, 7180),
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target", "client": "group"})
        .split(test_size=0.25, validation_size=0.2, random_state=15)
    )
    session.federated.fit(
        backend="native",
        method="fedavg",
        estimator="sgd_classifier" if task == "classification" else "sgd_regressor",
        n_rounds=2,
        local_epochs=2,
        random_state=9,
    )
    return session


def _assert_metrics(plan, frame, observed):
    predicted = plan.estimator_.predict(frame[list(plan.columns)].to_numpy())
    if plan.task == "classification":
        actual = plan.label_encoder_.transform(frame.y)
        assert observed["accuracy"] == pytest.approx(np.mean(actual == predicted))
    else:
        residual = frame.y.to_numpy() - predicted
        assert observed["mae"] == pytest.approx(np.mean(np.abs(residual)))
        assert observed["rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("population", ["partial_labels", "unlabeled", "empty", "missing_client"])
def test_evaluation_counts_and_metrics_match_the_labeled_population(task, population):
    session = _fitted(task)
    plan = session.federated_plan
    frame = session.partition("test").copy()
    if population in {"partial_labels", "unlabeled"}:
        positions = (
            np.arange(0, len(frame), 3) if population == "partial_labels" else np.arange(len(frame))
        )
        frame.iloc[positions, frame.columns.get_loc("y")] = np.nan
    elif population == "empty":
        frame = frame.iloc[:0].copy()
    else:
        frame = frame.drop(columns="client")
    original = frame.copy()
    dataset = Dataset.from_pandas(frame)
    fingerprint = joblib.hash(plan)
    result = evaluate_federated(dataset, plan, None, partition="all", backend="native")
    kept = frame.loc[frame.y.notna()]
    assert result.n_rows == len(kept)
    if kept.empty:
        assert result.metrics == result.per_client_metrics == {}
        assert result.n_clients_evaluated == 0
        assert result.warnings
    else:
        _assert_metrics(plan, kept, result.metrics)
        if "client" not in frame:
            assert result.n_clients_evaluated == 0 and result.per_client_metrics == {}
            assert any("Client column" in warning for warning in result.warnings)
        else:
            assert result.n_clients_evaluated == kept.client.nunique()
            assert set(result.per_client_metrics) == set(kept.client.astype(str))
            for client, group in kept.groupby("client"):
                _assert_metrics(plan, group, result.per_client_metrics[str(client)])
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), original)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("partition", ["train", "all", "test", "validation"])
def test_disclosures_distinguish_requested_population_from_verified_provenance(task, partition):
    session = _fitted(task)
    result = session.federated.evaluate(partition=partition, per_client=False)
    assert result.partition == partition
    assert result.per_client_metrics == {} and result.n_clients_evaluated == 0
    text = " ".join(result.disclosures).lower()
    assert "partition=" + repr(partition) in text
    # A loaded plan can be evaluated on another dataset or split. Evaluation
    # alone cannot verify that its rows were excluded from the original fit.
    assert "does not verify" in text and "training" in text
    if partition in {"train", "all"}:
        assert any("training" in item.lower() for item in result.warnings)
    assert "rows were never used" not in text


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("failure", ["feature", "target", "split", "backend"])
def test_incompatible_evaluation_requests_fail_without_mutating_plan(task, failure):
    session = _fitted(task)
    plan = session.federated_plan
    frame = session.to_pandas()
    kwargs = {"partition": "all"}
    match = ""
    if failure in {"feature", "target"}:
        column = "x" if failure == "feature" else "y"
        frame = frame.drop(columns=column)
        match = column
    elif failure == "split":
        kwargs["partition"] = "test"
        match = "SplitPlan"
    else:
        kwargs["backend"] = "flower"
        match = "does not match"
    fingerprint = joblib.hash(plan)
    with pytest.raises(ValidationError, match=match):
        evaluate_federated(Dataset.from_pandas(frame), plan, None, **kwargs)
    assert joblib.hash(plan) == fingerprint


def test_distinct_client_ids_cannot_silently_overwrite_reported_metrics():
    session = _fitted("regression")
    frame = session.partition("test").copy()
    frame["client"] = pd.Series(
        [1, "1"] * (len(frame) // 2) + [1] * (len(frame) % 2), index=frame.index, dtype=object
    )
    plan = session.federated_plan
    fingerprint = joblib.hash(plan)
    with pytest.raises(ValidationError, match="client.*same|same.*client"):
        evaluate_federated(Dataset.from_pandas(frame), plan, None, partition="all")
    assert joblib.hash(plan) == fingerprint
