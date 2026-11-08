"""Probabilistic points, sets and proper scores retain real class identities."""

import itertools

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

VOCABULARIES = [
    ["001", "002"], ["1.00", "1.0"], ["-0", "0"], ["True", "False"],
    [False, True], [2, 10], [1.25, 2.5], ["null", "None", "<NA>"],
    [b"cold", b"warm"], [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
    [np.float32(1.1), np.float32(2.2)],
]
BACKENDS = [("native", "gaussian_nb"), ("mapie", "split"),
            ("mapie", "cv_plus"), ("mapie", "jackknife_plus")]


def _label_type(value):
    return type(value.item() if isinstance(value, np.generic) else value)


def _session(frame):
    return (Session.ingest(frame).set_roles({"a": "feature", "b": "feature", "y": "target"})
            .split(test_size=.2, validation_size=.2, random_state=4))


@pytest.fixture(scope="module", params=list(itertools.product(BACKENDS, VOCABULARIES)))
def fitted(request):
    (backend, method), vocabulary = request.param
    if backend == "mapie":
        pytest.importorskip("mapie")
    rng = np.random.default_rng(819)
    codes = np.arange(80) % len(vocabulary)
    x = rng.normal(size=(80, 2)) + codes[:, None]*1.1
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": [vocabulary[i] for i in codes]},
                          index=np.arange(80) % 9)
    if any(isinstance(value, np.generic) for value in vocabulary):
        frame["y"] = pd.array([vocabulary[i] for i in codes], dtype=object)
    session = _session(frame)
    session.probabilistic.fit(backend=backend, estimator=method, task="classification", alpha=.2)
    return session, frame, sorted(vocabulary, key=str)


def _probabilities(session, x, vocabulary):
    plan = session.probabilistic_plan
    if plan.backend == "native":
        fitted = session.dataset.frame.iloc[list(plan.conformal_fit_indices_)]
        train_x = fitted[["a", "b"]].to_numpy()
        epsilon = 1e-9 * train_x.var(axis=0).max()
        logs = []
        for label in vocabulary:
            values = train_x[fitted.y.to_numpy() == label]
            variance = values.var(axis=0) + epsilon
            logs.append(np.log(len(values)/len(train_x)) - .5*np.sum(
                np.log(2*np.pi*variance)+(x-values.mean(axis=0))**2/variance, axis=1))
        logs = np.column_stack(logs)
    else:
        # Actual fitted MAPIE 1.5 consumer, independently of BuildML's adapter.
        base = plan.estimator_.estimator._mapie_classifier.estimator_.single_estimator_
        logits = x @ base.coef_.T + base.intercept_
        if logits.shape[1] == 1:
            positive = 1/(1+np.exp(-logits[:, 0]))
            return np.column_stack([1-positive, positive])
        logs = logits
    probabilities = np.exp(logs-logs.max(axis=1, keepdims=True))
    return probabilities/probabilities.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("replay", [False, True])
def test_classification_consumers_match_density_or_affine_oracles_and_exact_labels(fitted, partition, replay, tmp_path):
    original, frame, vocabulary = fitted
    session = original
    if replay:
        session.probabilistic.save_bundle(tmp_path / "model")
        session = _session(frame).probabilistic.load_bundle(tmp_path / "model", trusted=True)
    plan = session.probabilistic_plan
    rows = frame.iloc[list(session.split_plan.indices_for(partition))]
    x = rows[["a", "b"]].to_numpy()
    probabilities = _probabilities(session, x, vocabulary)
    expected = [vocabulary[i] for i in probabilities.argmax(axis=1)]
    fingerprint = joblib.hash((plan.label_encoder_, plan.conformal_fit_indices_, plan.conformal_calib_indices_))
    prediction = session.probabilistic.predict(partition=partition)
    assert list(prediction.predictions) == expected
    assert all(_label_type(a) is _label_type(b) for a, b in zip(prediction.predictions, expected, strict=True))
    assert prediction.probabilities is not None
    np.testing.assert_allclose(prediction.probabilities, probabilities, rtol=2e-8, atol=2e-10)
    plain = session.probabilistic.predict(partition=partition, return_proba=False)
    assert plain.predictions == prediction.predictions and plain.probabilities is None
    interval = session.probabilistic.predict_interval(partition=partition)
    if plan.backend == "native":
        calibration = frame.iloc[list(plan.conformal_calib_indices_)]
        calibration_proba = _probabilities(session, calibration[["a", "b"]].to_numpy(), vocabulary)
        true_codes = [vocabulary.index(value) for value in calibration.y]
        scores = sorted(1-calibration_proba[i, code] for i, code in enumerate(true_codes))
        quantile = scores[int(np.ceil((len(scores)+1)*.8))-1]
        assert plan.conformal_quantile_ == pytest.approx(quantile)
        expected_sets = [tuple(label for label, p in zip(vocabulary, row, strict=True) if 1-p <= quantile)
                         or (vocabulary[int(np.argmax(row))],) for row in probabilities]
    else:
        _, mask = plan.estimator_.estimator.predict_set(x)
        mask = np.asarray(mask).reshape(len(x), len(vocabulary), -1)[:, :, 0]
        expected_sets = [tuple(label for label, include in zip(vocabulary, row, strict=True) if include)
                         or (expected[i],) for i, row in enumerate(mask)]
    assert list(interval.prediction_sets) == expected_sets
    assert all(_label_type(a) is _label_type(b) for actual, expected_set in zip(interval.prediction_sets, expected_sets, strict=True)
               for a, b in zip(actual, expected_set, strict=True))
    assert interval.point == prediction.predictions
    actual = rows.y.tolist()
    actual_codes = np.array([vocabulary.index(value) for value in actual])
    report = session.probabilistic.evaluate(partition=partition)
    assert report.metrics["accuracy"] == pytest.approx(np.mean(actual_codes == probabilities.argmax(axis=1)))
    assert report.metrics["nll"] == pytest.approx(-np.log(probabilities[np.arange(len(x)), actual_codes]).mean())
    if len(vocabulary) == 2:
        assert report.metrics["brier"] == pytest.approx(np.mean(((actual_codes == 1)-probabilities[:, 1])**2))
    assert report.metrics["set_coverage"] == pytest.approx(np.mean([
        value in members for value, members in zip(actual, expected_sets, strict=True)]))
    assert report.metrics["mean_set_size"] == pytest.approx(np.mean([len(members) for members in expected_sets]))
    assert joblib.hash((plan.label_encoder_, plan.conformal_fit_indices_, plan.conformal_calib_indices_)) == fingerprint
    metadata = plan.to_dict()
    assert metadata["classes"] == [str(label.item() if isinstance(label, np.generic) else label)
                                   for label in vocabulary]
    assert metadata["original_classes"] == vocabulary
    assert all(_label_type(a) is _label_type(b)
               for a, b in zip(metadata["original_classes"], vocabulary, strict=True))


@pytest.mark.parametrize("values", [[1, "1", 2], [True, "True", False], [1.5, "1.5", 2.5]])
def test_ambiguous_label_encodings_refuse_before_replacing_a_working_plan(values):
    frame = pd.DataFrame({"a": np.cos(np.arange(80)), "b": np.sin(np.arange(80)),
                          "y": ["cold", "warm"]*40})
    session = _session(frame)
    session.probabilistic.fit(estimator="gaussian_nb", task="classification", conformal=False)
    plan = session.probabilistic_plan
    before = list(session._history)
    fingerprint = joblib.hash(plan)
    session.dataset.frame["y"] = pd.Series([values[i % len(values)] for i in range(80)], dtype=object)
    with pytest.raises(ValidationError, match="Ambiguous|ambiguous"):
        session.probabilistic.fit(estimator="gaussian_nb", task="classification", conformal=False)
    assert session.probabilistic_plan is plan and session._history == before
    assert joblib.hash(plan) == fingerprint
