"""Threshold choices match scalar confusion counts and respect tuning boundaries."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def _session(seed, poison=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(160, 2))
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": (x[:, 0] + x[:, 1] > 0.7).astype(int)})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=13)
    )
    if poison:
        rows = list(session.split_plan.test_indices)
        session.dataset.frame.iloc[rows, session.dataset.frame.columns.get_loc("y")] ^= 1
    session.anomaly.fit(
        method="isolation_forest", contamination=0.2, n_estimators=40, random_state=seed
    )
    return session


def _counts(labels, scores, threshold):
    flags = scores >= threshold
    tp = int(np.sum(flags & (labels == 1)))
    fp = int(np.sum(flags & (labels == 0)))
    fn = int(np.sum(~flags & (labels == 1)))
    tn = int(np.sum(~flags & (labels == 0)))
    return tp, fp, fn, tn


@pytest.mark.parametrize("seed", [7, 33])
@pytest.mark.parametrize("metric", ["f1", "fbeta", "youden", "precision_at_contamination"])
def test_real_detector_thresholds_match_scalar_oracles_and_ignore_test_labels(seed, metric):
    session = _session(seed)
    plan = session.anomaly_plan
    frame = session.partition("validation")
    scores = -plan.estimator_.score_samples(frame[["x", "z"]].to_numpy())
    labels = frame.y.to_numpy()
    beta = 1 if metric == "f1" else 2
    if metric == "precision_at_contamination":
        ordered = sorted(scores)
        position = (len(ordered) - 1) * (1 - plan.contamination)
        lower = int(position)
        threshold = ordered[lower] + (position - lower) * (
            ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
        )
        tp, fp, _, _ = _counts(labels, scores, threshold)
        expected_metric = tp / max(tp + fp, 1)
        key = "precision_at_tuned_threshold"
    else:

        def value(threshold):
            tp, fp, fn, tn = _counts(labels, scores, threshold)
            if metric == "youden":
                return tp / (tp + fn) - fp / (fp + tn)
            denominator = (1 + beta**2) * tp + beta**2 * fn + fp
            return (1 + beta**2) * tp / denominator if denominator else 0

        candidates = sorted(set(scores))
        if metric == "youden":
            candidates.append(float("inf"))
        threshold = max(candidates, key=lambda candidate: (value(candidate), candidate))
        expected_metric = value(threshold)
        key = "youden" if metric == "youden" else "best_metric_value"
    old = plan.threshold_
    actual = session.anomaly.tune_threshold(metric=metric, fbeta=2, update_plan=False)
    assert actual.threshold == pytest.approx(threshold)
    assert actual.tune_metrics[key] == pytest.approx(expected_metric)
    assert actual.alert_rate == pytest.approx(np.mean(scores >= threshold))
    assert plan.threshold_ == old
    replay = _session(seed, poison=True).anomaly.tune_threshold(
        metric=metric, fbeta=2, update_plan=False
    )
    assert replay.threshold == actual.threshold
    assert replay.tune_metrics == actual.tune_metrics


@pytest.fixture(scope="module")
def fitted():
    return _session(4)


def test_exploratory_test_tuning_must_disclose_that_test_was_used(fitted):
    result = fitted.anomaly.tune_threshold(
        partition="test", allow_test_tuning=True, update_plan=False
    )
    text = " ".join(result.disclosures).lower()
    assert "test was not used" not in text
    assert "test" in text and "optimistic" in text


@pytest.mark.parametrize("metric", ["typo", "F1", "", None])
def test_unknown_metric_is_not_silently_interpreted_as_fbeta(fitted, metric):
    with pytest.raises(ValidationError, match="metric"):
        fitted.anomaly.tune_threshold(metric=metric, update_plan=False)


@pytest.mark.parametrize(
    "fbeta", [-1, np.nan, np.inf, True, "2", pytest.param(10**1000, id="overflow")]
)
def test_fbeta_requires_a_finite_nonnegative_number(fitted, fbeta):
    with pytest.raises(ValidationError, match="fbeta"):
        fitted.anomaly.tune_threshold(metric="fbeta", fbeta=fbeta, update_plan=False)


@pytest.mark.parametrize("beta", [0, np.float64(0.5)])
def test_zero_and_numpy_fractional_beta_preserve_confusion_count_optimum(fitted, beta):
    frame = fitted.partition("validation")
    scores = -fitted.anomaly_plan.estimator_.score_samples(frame[["x", "z"]].to_numpy())
    labels = frame.y.to_numpy()

    def value(threshold):
        tp, fp, fn, _ = _counts(labels, scores, threshold)
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp
        return (1 + beta**2) * tp / denominator if denominator else 0

    expected = max(set(scores), key=lambda threshold: (value(threshold), threshold))
    tuned = fitted.anomaly.tune_threshold(metric="fbeta", fbeta=beta, update_plan=False)
    assert tuned.threshold == expected
    assert tuned.tune_metrics["best_metric_value"] == pytest.approx(value(expected))


@pytest.mark.parametrize("metric", ["f1", "fbeta", "youden", "precision_at_contamination"])
def test_applied_validation_threshold_controls_real_test_flags(metric):
    session = _session(7)
    tuned = session.anomaly.tune_threshold(metric=metric, update_plan=True)
    plan = session.anomaly_plan
    assert plan.threshold_ == tuned.threshold
    frame = session.partition("test")
    scores = -plan.estimator_.score_samples(frame[["x", "z"]].to_numpy())
    actual = session.anomaly.score(partition="test")
    np.testing.assert_allclose(actual.scores, scores, rtol=1e-12)
    assert actual.flags == tuple((scores >= tuned.threshold).astype(int))


@pytest.mark.parametrize("partition", ["train", "test"])
def test_stored_tuning_policy_names_the_actual_selection_population(partition):
    session = _session(9)
    session.anomaly.tune_threshold(partition=partition, allow_test_tuning=True, update_plan=True)
    assert session.anomaly_plan.threshold_policy == f"{partition}_tuned"
    assert session.anomaly_plan.to_dict()["threshold_policy"] == f"{partition}_tuned"
    if partition == "test":
        assert any("optimistic" in text for text in session.anomaly_plan.disclosures)
    with pytest.raises(ValidationError, match="set by tune_anomaly_threshold"):
        session.anomaly.fit(method="isolation_forest", threshold_policy=f"{partition}_tuned")
