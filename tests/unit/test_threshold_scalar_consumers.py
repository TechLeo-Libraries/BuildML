"""Real GaussianNB decision costs agree with independent probability and policy sums."""

import copy
import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline

BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")
COSTS = dict(fp_cost=2., fn_cost=5., tp_benefit=1., tn_benefit=3.)
FIELDS = tuple(COSTS)


def _session(backing, directory, kind="probabilistic"):
    labels = np.asarray(["no", "yes"] * 30)
    x = np.random.default_rng(841).normal(size=(60, 2))
    x += (labels == "yes")[:, None] * [1.5, -.3]
    y = 2*x[:, 0] - x[:, 1] if kind == "regression" else labels
    frame = pd.DataFrame(dict(x=x[:, 0], z=x[:, 1], label=y))
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "decisions.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="memory" if engine == "pandas" else "lazy")
    session.set_roles({"x": "feature", "z": "feature", "label": "target"}).inject_split(
        train_indices=list(range(40)), validation_indices=list(range(40, 50)), test_indices=list(range(50, 60)))
    estimator = Ridge() if kind == "regression" else (LinearSVC() if kind == "no-probabilities" else GaussianNB())
    session.fit(estimator, task="regression" if kind == "regression" else "classification")
    return session, x, labels


def _probabilities(session, x, labels):
    """Derive fitted Gaussian densities directly, including variance smoothing."""
    epsilon = 1e-9 * np.var(x[:40], axis=0, ddof=0).max()
    means, variances, joint = [], [], []
    for label in ("no", "yes"):
        subset = x[:40][labels[:40] == label]
        mean, variance = subset.mean(axis=0), subset.var(axis=0, ddof=0) + epsilon
        means.append(mean)
        variances.append(variance)
        joint.append(np.log(len(subset)/40) - .5 * np.sum(
            np.log(2*np.pi*variance) + (x[40:]-mean)**2/variance, axis=1))
    joint = np.asarray(joint).T
    unnormalized = np.exp(joint - joint.max(axis=1, keepdims=True))
    expected = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    fitted = session.fit_result.estimator
    np.testing.assert_allclose(fitted.theta_, means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted.var_, variances, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(fitted.class_count_, [20, 20])
    actual = fitted.predict_proba(pd.DataFrame(x[40:], columns=["x", "z"]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return actual[:, 1]


def _assert_policy(report, probabilities, truth, costs):
    candidates = []
    for row in report.payload["rows"]:
        # Boolean enumeration is independent of the production prefix-count sweep.
        chosen, positive = probabilities >= row["threshold"], truth == "yes"
        counts = dict(tp=int((chosen & positive).sum()), fp=int((chosen & ~positive).sum()),
            tn=int((~chosen & ~positive).sum()), fn=int((~chosen & positive).sum()))
        assert {name: row[name] for name in counts} == counts
        total = (float(costs["fp_cost"])*counts["fp"] + float(costs["fn_cost"])*counts["fn"]
            - float(costs["tp_benefit"])*counts["tp"] - float(costs["tn_benefit"])*counts["tn"])
        denominator = 2*counts["tp"] + counts["fp"] + counts["fn"]
        f1 = 2*counts["tp"]/denominator if denominator else 0.
        assert row["f1"] == pytest.approx(f1)
        assert row["expected_cost_total"] == pytest.approx(total)
        assert row["expected_cost_mean"] == pytest.approx(total/len(truth))
        candidates.append((total, -f1, row["threshold"]))
    winner = min(candidates)
    assert report.payload["recommended_threshold"]["threshold"] == winner[2]
    assert report.payload["expected_cost_at_recommended"] == pytest.approx(winner[0])
    assert report.payload["recommendation_basis"] == "min_expected_cost"
    assert report.payload["positive_class"] == "yes"
    assert report.payload["n_rows"] == len(truth)
    assert any(row["tp"] + row["fp"] == 0 for row in report.payload["rows"])
    assert any(row["tn"] + row["fn"] == 0 for row in report.payload["rows"])
    for name in costs:
        assert report.payload["cost_model"][name] == float(costs[name])


def _snapshot(session):
    return (session.fit_result, joblib.hash(session.fit_result.estimator),
        session.to_pandas().copy(deep=True), copy.deepcopy(session.split_plan), copy.deepcopy(session.history))


def _unchanged(session, snapshot, *, history):
    fit, weights, frame, split, previous_history = snapshot
    if history:
        assert session.history == previous_history
    assert session.fit_result is fit
    assert joblib.hash(session.fit_result.estimator) == weights
    pd.testing.assert_frame_equal(session.to_pandas(), frame)
    assert session.split_plan == split


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("scalar", (float, np.float32, np.int64, np.uint32), ids=("python", "float32", "int64", "uint32"))
def test_real_scalar_costs_match_gaussian_and_boolean_policy_oracles(backing, field, scalar, tmp_path):
    session, x, labels = _session(backing, tmp_path)
    probabilities = _probabilities(session, x, labels)
    before = _snapshot(session)
    costs = dict(COSTS, **{field: scalar(COSTS[field])})
    report = session.tune_threshold(**costs)
    _assert_policy(report, probabilities[:10], labels[40:50], costs)
    _unchanged(session, before, history=False)
    if scalar is float:
        path = tmp_path / "trusted-model"
        session.save_pipeline(path, evaluate_partition=None)
        bundle = load_pipeline_bundle(path, trusted=True)
        weights = joblib.hash(bundle)
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        source = session.partition("test").copy(deep=True)
        expected = np.where(probabilities[10:] >= .5, "yes", "no")
        for stored in (path, bundle):
            replay = predict_from_pipeline(stored, source, trusted=True)
            np.testing.assert_array_equal(replay.predictions, expected)
            pd.testing.assert_index_equal(replay.predictions.index, source.index)
        assert joblib.hash(bundle) == weights
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()} == files


INVALID = (True, np.bool_(False), "2", complex(2, 0), np.nan, np.inf, -np.inf, 10**1000)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", INVALID, ids=("bool", "numpy-bool", "string", "complex", "nan", "inf", "negative-inf", "huge-int"))
def test_invalid_costs_raise_domain_errors_without_changing_prior_report_or_fit(backing, field, value, tmp_path):
    session, x, labels = _session(backing, tmp_path)
    probabilities = _probabilities(session, x, labels)
    report = session.tune_threshold(**COSTS)
    before = _snapshot(session)
    with pytest.raises(ValidationError, match=field + " must be a finite number"):
        session.tune_threshold(**dict(COSTS, **{field: value}))
    _unchanged(session, before, history=True)
    assert session._last_diagnostic is report
    _assert_policy(report, probabilities[:10], labels[40:50], COSTS)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("field", ("fp_cost", "fn_cost"))
@pytest.mark.parametrize("value", (-.5, np.float32(-.5)), ids=("python", "float32"))
def test_negative_costs_are_refused_atomically(backing, field, value, tmp_path):
    session, _, _ = _session(backing, tmp_path)
    report = session.tune_threshold(**COSTS)
    before = _snapshot(session)
    with pytest.raises(ValidationError, match=field + " must be >= 0"):
        session.tune_threshold(**dict(COSTS, **{field: value}))
    _unchanged(session, before, history=True)
    assert session._last_diagnostic is report


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("field", ("tp_benefit", "tn_benefit"))
@pytest.mark.parametrize("value", (-.5, np.float32(-.5)), ids=("python", "float32"))
def test_negative_benefits_retain_their_signed_policy_meaning(backing, field, value, tmp_path):
    session, x, labels = _session(backing, tmp_path)
    probabilities = _probabilities(session, x, labels)
    costs = dict(COSTS, **{field: value})
    _assert_policy(session.tune_threshold(**costs), probabilities[:10], labels[40:50], costs)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("kind", ("regression", "no-probabilities"))
def test_threshold_tuning_refuses_real_models_without_binary_probabilities(backing, kind, tmp_path):
    session, _, _ = _session(backing, tmp_path, kind)
    before = _snapshot(session)
    with pytest.raises(ValidationError, match="probabilistic binary classifier"):
        session.tune_threshold(**COSTS)
    _unchanged(session, before, history=True)
    assert session._last_diagnostic is None


@pytest.mark.parametrize("backing", BACKINGS)
def test_loaded_probabilistic_model_requires_a_policy_partition(backing, tmp_path):
    session, x, labels = _session(backing, tmp_path)
    probabilities = _probabilities(session, x, labels)
    path = tmp_path / "trusted-model"
    session.save_pipeline(path, evaluate_partition=None)
    receiver = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "z": "feature", "label": "target"}).load_pipeline(path, trusted=True)
    assert receiver.split_plan is None
    before = _snapshot(receiver)
    with pytest.raises(ValidationError, match="Split required"):
        receiver.tune_threshold(**COSTS)
    _unchanged(receiver, before, history=True)
    receiver.inject_split(train_indices=list(range(40)), validation_indices=list(range(40, 50)), test_indices=list(range(50, 60)))
    _assert_policy(receiver.tune_threshold(**COSTS), probabilities[:10], labels[40:50], COSTS)
