"""Fitted synthetic populations and vocabulary checks retain category meaning."""

import json
from collections import Counter
from dataclasses import replace
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.synthetic.types import ColumnSchemaSpec
from buildml.synthetic.validation import validate_synthetic


def _key(value):
    if pd.isna(value):
        return ("missing",)
    if isinstance(value, np.generic):
        value = value.item()
    return (type(value).__name__, value)


def _session(values, method="gaussian_copula", replay=False, tmp_path=None):
    values = pd.Series([values[i % len(values)] for i in range(120)], dtype=object)
    frame = pd.DataFrame({"category": values})
    frame.index = np.arange(900, 1020)
    session = Session.ingest(frame).set_roles({"category": "feature"}).inject_split(
        train_indices=list(range(80)), validation_indices=list(range(80, 100)),
        test_indices=list(range(100, 120)),
    )
    session.synthetic.fit(method=method, random_state=3)
    if replay:
        session.synthetic.save_bundle(tmp_path / "generator")
        # The public JSON summary must remain consumable by ordinary JSON tools,
        # even when fitted categories contain NumPy or missing scalar values.
        metadata = json.loads((tmp_path / "generator/meta.json").read_text(encoding="utf-8"))
        assert metadata["plan"]["has_generator"]
        session.synthetic.load_bundle(tmp_path / "generator", trusted=True)
    return session


@pytest.mark.parametrize("values", [(1, "1"), (False, "False"), (1.5, "1.5"),
                                    (True, 1), (np.int64(1), "1"), ("left", "right")])
@pytest.mark.parametrize("replay", [False, True])
def test_copula_marginals_preserve_distinct_typed_populations(values, replay, tmp_path):
    session = _session(values, replay=replay, tmp_path=tmp_path)
    frozen, before = joblib.hash(session.synthesizer_plan), session.to_pandas().copy()
    sample = session.synthetic.sample(n=2000, random_state=11, validate=True)
    expected = Counter(map(_key, session.partition("train").category))
    observed = Counter(map(_key, sample.frame.category))
    assert set(observed) == set(expected)
    for category, count in expected.items():
        assert observed[category] / 2000 == pytest.approx(count / 80, abs=.04)
    assert "validate_synthetic passed=True" in sample.warnings
    repeated = session.synthetic.sample(n=2000, random_state=11)
    pd.testing.assert_frame_equal(repeated.frame, sample.frame)
    assert joblib.hash(session.synthesizer_plan) == frozen
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("values", [(1, "1"), (False, "False"), (True, 1),
                                    (None, "__NA__"), (None, "None")])
@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("merge_mode", ["none", "extend_train"])
def test_conditions_select_exact_categorical_identity_and_preserve_holdouts(
    values, which, replay, merge_mode, tmp_path,
):
    session = _session(values, replay=replay, tmp_path=tmp_path)
    before, split = session.to_pandas().copy(), session.split_plan
    frozen = joblib.hash(session.synthesizer_plan)
    selected = session.synthetic.sample(n=24, random_state=19,
                                        condition={"category": values[which]},
                                        merge_mode=merge_mode)
    assert {_key(value) for value in selected.frame.category} == {_key(values[which])}
    assert joblib.hash(session.synthesizer_plan) == frozen
    for partition in ("validation", "test"):
        actual = session.partition(partition)
        expected = before.iloc[list(getattr(split, partition + "_indices"))]
        pd.testing.assert_frame_equal(actual[["category"]].reset_index(drop=True),
                                      expected.reset_index(drop=True))
    if merge_mode == "extend_train":
        assert len(session.split_plan.train_indices) == len(split.train_indices) + 24
        assert {_key(value) for value in session.partition("train").category.iloc[-24:]} == {_key(values[which])}
    else:
        assert session.split_plan is split
        pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("values,probe", [
    ((1, 2), ("1", 2)), ((False, True), ("False", True)),
    ((0, 1), (False, True)), ((False, True), (0, 1)),
    ((1, "1"), (1, "1")), ((1.5, "1.5"), (1.5, "1.5")),
    ((None, "ordinary"), (None, "ordinary")),
    ((None, "ordinary"), ("__NA__", "ordinary")),
])
@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("tolerance", [0, .5])
def test_vocabulary_oov_rates_match_typed_nonmissing_counts(
    values, probe, method, replay, tolerance, tmp_path,
):
    session = _session(values, method=method, replay=replay, tmp_path=tmp_path)
    before = joblib.hash(session.synthesizer_plan)
    frame = pd.DataFrame({"category": pd.Series(list(probe) * 10, dtype=object)})
    known = {_key(value) for value in values if not pd.isna(value)}
    expected = sum(not pd.isna(value) and _key(value) not in known for value in frame.category) / len(frame)
    result = validate_synthetic(session.synthesizer_plan, frame, run_great_expectations=False,
                                categorical_oov_tolerance=tolerance, null_rate_tolerance=1)
    assert result.metrics["oov_rate::category"] == pytest.approx(expected)
    assert result.checks["categorical_vocab::category"] == (expected <= tolerance)
    assert result.passed == (expected <= tolerance)
    assert result.n_failed == sum(not value for value in result.checks.values())
    assert not any("legacy" in warning.lower() for warning in result.warnings)
    assert joblib.hash(session.synthesizer_plan) == before


@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
@pytest.mark.parametrize("values", [(None, "ordinary"), (pd.NA, "__NA__", "ordinary")])
@pytest.mark.parametrize("replay", [False, True])
def test_seen_missing_categories_do_not_fail_vocabulary_validation(method, values, replay, tmp_path):
    session = _session(values, method=method, replay=replay, tmp_path=tmp_path)
    sample = session.synthetic.sample(n=400, random_state=11, validate=True)
    assert "validate_synthetic passed=True" in sample.warnings
    result = validate_synthetic(session.synthesizer_plan, sample.frame, run_great_expectations=False)
    assert result.metrics["oov_rate::category"] == 0
    assert result.metrics["null_rate::category"] == pytest.approx(sample.frame.category.isna().mean())
    assert result.passed


@pytest.mark.parametrize("values,wrong", [((0, 1), True), ((False, True), 1),
                                         ((1, 2), "1")])
def test_unseen_typed_condition_refuses_before_extending_training(values, wrong):
    session = _session(values)
    before, split, history = session.to_pandas().copy(), session.split_plan, list(session.history)
    frozen = joblib.hash(session.synthesizer_plan)
    with pytest.raises(ValidationError, match="condition"):
        session.synthetic.sample(n=8, condition={"category": wrong}, random_state=11,
                                 merge_mode="extend_train")
    assert session.split_plan is split and session.history == history
    assert joblib.hash(session.synthesizer_plan) == frozen
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
def test_legacy_string_vocabulary_discloses_its_limit_without_mutating_plan(method):
    session = _session((1, 2), method=method)
    plan = replace(session.synthesizer_plan, column_specs=(
        ColumnSchemaSpec(name="category", kind="categorical", n_unique=2,
                         categories=("1", "2")),
    ))
    before = joblib.hash(plan)
    result = validate_synthetic(plan, pd.DataFrame({"category": ["1", "2"]}),
                                run_great_expectations=False)
    assert result.passed
    assert any("legacy" in warning.lower() and "refit" in warning.lower() for warning in result.warnings)
    assert joblib.hash(plan) == before


def test_copula_numeric_and_typed_latent_correlation_matches_independent_ranks():
    x = np.arange(120, dtype=float)
    labels = [1 if i % 3 else "1" for i in range(120)]
    frame = pd.DataFrame({"x": x, "category": pd.Series(labels, dtype=object)})
    session = Session.ingest(frame).set_roles({"x": "feature", "category": "feature"}).inject_split(
        train_indices=list(range(80)), validation_indices=list(range(80, 100)),
        test_indices=list(range(100, 120)),
    )
    session.synthetic.fit(method="gaussian_copula", random_state=3)
    train = session.partition("train")
    distribution = NormalDist()
    ranks = np.array([distribution.inv_cdf((i + .5) / 80) for i in range(80)])
    known = Counter(map(_key, train.category))
    ordered = sorted(known, key=lambda key: -known[key])
    cumulative, mids = 0., {}
    for key in ordered:
        mass = known[key] / 80
        mids[key] = distribution.inv_cdf(cumulative + mass / 2)
        cumulative += mass
    categorical = np.array([mids[_key(value)] for value in train.category])
    a, b = ranks - ranks.mean(), categorical - categorical.mean()
    correlation = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)) / 1.001
    expected = np.array([[1., correlation], [correlation, 1.]])
    np.testing.assert_allclose(session.synthesizer_plan.generator_.corr, expected, atol=1e-12)
    z = np.random.default_rng(11).multivariate_normal(np.zeros(2), expected, size=200)
    u = np.array([[distribution.cdf(value) for value in row] for row in z])
    expected_x = np.minimum((u[:, 0] * 80).astype(int), 79).astype(float)
    boundary = known[ordered[0]] / 80
    expected_labels = [ordered[int(value > boundary)] for value in u[:, 1]]
    sample = session.synthetic.sample(n=200, random_state=11)
    np.testing.assert_array_equal(sample.frame.x, expected_x)
    assert list(map(_key, sample.frame.category)) == expected_labels
