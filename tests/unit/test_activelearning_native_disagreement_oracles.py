"""Native acquisition scores agree with scalar real-committee calculations."""

import math
from copy import copy

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from buildml.activelearning.adapters.industry_native import score_industry_native_pool
from buildml.core.errors import ValidationError


def _inputs(seed, n_estimators=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(90, 3))
    y = np.argmax(x + rng.normal(scale=1.2, size=x.shape), axis=1)
    pool = rng.normal(size=(15, 3))
    committee = BaggingClassifier(
        estimator=LogisticRegression(max_iter=400), n_estimators=n_estimators, random_state=seed
    ).fit(x, y)
    return x, y, pool, committee


def _score(strategy, x, y, pool, committee):
    return score_industry_native_pool(
        strategy=strategy,
        x_labeled=x,
        y_labeled=y,
        x_pool=pool,
        estimator=getattr(committee, "estimator_", None),
        committee=committee,
    )


@pytest.mark.parametrize("seed", [3, 17, 41])
@pytest.mark.parametrize("strategy", ["core_set", "qbc_kl", "qbc_variation_ratios"])
def test_real_committee_matches_scalar_acquisition_oracle(seed, strategy):
    x, y, pool, committee = _inputs(seed)
    before = pool.copy()
    if strategy == "core_set":
        expected = [min(math.dist(point, row) for row in x) for point in pool]
    elif strategy == "qbc_variation_ratios":
        votes = [member.predict(pool).tolist() for member in committee.estimators_]
        expected = [
            1
            - max(
                sum(member[i] == label for member in votes)
                for label in set(member[i] for member in votes)
            )
            / len(votes)
            for i in range(len(pool))
        ]
    else:
        distributions = [
            np.clip(member.predict_proba(pool), 1e-12, 1) for member in committee.estimators_
        ]
        # Mean over all ordered member pairs; no arbitrary member gets to be
        # the numerator merely because it appeared first in estimators_.
        expected = [
            math.fsum(
                math.fsum(
                    float(p) * math.log(float(p) / float(q))
                    for p, q in zip(left[row], right[row], strict=True)
                )
                for i, left in enumerate(distributions)
                for j, right in enumerate(distributions)
                if i != j
            )
            / (len(distributions) * (len(distributions) - 1))
            for row in range(len(pool))
        ]
    actual = _score(strategy, x, y, pool, committee)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        np.argsort(-actual, kind="stable"), np.argsort(-np.array(expected), kind="stable")
    )
    np.testing.assert_array_equal(pool, before)


@pytest.mark.parametrize("seed", [3, 17, 41])
def test_kl_disagreement_is_invariant_to_committee_member_order(seed):
    x, y, pool, committee = _inputs(seed)
    reversed_committee = copy(committee)
    reversed_committee.estimators_ = list(reversed(committee.estimators_))
    expected = _score("qbc_kl", x, y, pool, committee)
    actual = _score("qbc_kl", x, y, pool, reversed_committee)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.argsort(-actual), np.argsort(-expected))


@pytest.mark.parametrize("strategy", ["qbc_kl", "qbc_variation_ratios"])
def test_single_real_member_has_no_disagreement(strategy):
    x, y, pool, committee = _inputs(3, n_estimators=1)
    np.testing.assert_array_equal(_score(strategy, x, y, pool, committee), np.zeros(len(pool)))


def test_missing_committee_and_unfitted_members_are_refused():
    x, y, pool, committee = _inputs(3)
    with pytest.raises(ValidationError, match="requires a fitted committee"):
        score_industry_native_pool(
            strategy="qbc_kl",
            x_labeled=x,
            y_labeled=y,
            x_pool=pool,
            estimator=committee,
            committee=None,
        )
    with pytest.raises(ValidationError, match="no fitted estimators"):
        _score("qbc_kl", x, y, pool, BaggingClassifier())
    no_probabilities = BaggingClassifier(
        estimator=RidgeClassifier(), n_estimators=2, random_state=3
    ).fit(x, y)
    with pytest.raises(ValidationError, match="predict_proba"):
        _score("qbc_kl", x, y, pool, no_probabilities)


def test_empty_pool_and_empty_labeled_seed_have_explicit_fallback_scores():
    x, y, pool, committee = _inputs(3)
    assert _score("core_set", x, y, pool[:0], committee).shape == (0,)
    norms = [math.sqrt(math.fsum(float(value) ** 2 for value in row)) for row in pool]
    np.testing.assert_allclose(
        _score("core_set", x[:0], y[:0], pool, committee),
        np.array(norms) / (max(norms) + 1e-12),
        atol=1e-12,
    )
    with pytest.raises(ValidationError, match="Unsupported native industry strategy"):
        _score("unknown", x, y, pool, committee)
