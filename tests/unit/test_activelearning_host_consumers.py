"""Require real host execution and independent distance/disagreement scores."""
import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

pytest.importorskip('skactiveml')
from buildml.activelearning.adapters.scikit_activeml import (
    _SKACTIVEML_SUCCESS_DISCLOSURE,
    _score_with_skactiveml,
    score_industry_pool,
)


def inputs(n_labeled=8, n_classes=2):
    rng = np.random.default_rng(64)
    train = rng.normal(size=(64, 3))
    y = np.digitize(train[:, 0]+train[:, 1], [0.] if n_classes == 2 else [-.6, .6])
    committee = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=5,
                                  random_state=7).fit(train, y)
    return dict(x_labeled=train[:n_labeled], y_labeled=y[:n_labeled],
                x_pool=rng.normal(size=(11, 3)), estimator=committee, committee=committee)


def oracle(strategy, args):
    pool, labeled = args['x_pool'], args['x_labeled']
    if strategy == 'core_set':
        if len(labeled):
            return np.linalg.norm(pool[:, None]-labeled[None, :], axis=2).min(axis=1)
        return -np.linalg.norm(pool[:, None]-pool[None, :], axis=2).sum(axis=1)
    members = args['committee'].estimators_
    if strategy == 'qbc_variation_ratios':
        votes = np.array([member.predict(pool) for member in members])
        return np.array([1-np.unique(votes[:, j], return_counts=True)[1].max()/len(members)
                         for j in range(len(pool))])
    probabilities = np.array([member.predict_proba(pool) for member in members])
    probabilities = np.clip(probabilities, 1e-7, 1)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    mean = probabilities.mean(axis=0)
    return (probabilities*np.log(probabilities/mean)).sum(axis=2).mean(axis=0)


@pytest.mark.parametrize('strategy', ['core_set', 'qbc_kl', 'qbc_variation_ratios'])
@pytest.mark.parametrize('n_labeled', [0, 8])
@pytest.mark.parametrize('n_classes', [2, 3])
def test_host_scores_match_numerical_oracle_without_refitting(strategy, n_labeled, n_classes):
    args = inputs(n_labeled, n_classes)
    before = [member.tree_.value.copy() for member in args['committee'].estimators_]
    actual = _score_with_skactiveml(strategy=strategy, **args)
    # Host pairwise-distance kernels can leave ~4e-8 self-distance roundoff.
    tolerance = 1e-7 if strategy == 'core_set' else 1e-10
    np.testing.assert_allclose(actual, oracle(strategy, args), rtol=1e-9, atol=tolerance)
    assert actual.shape == (len(args['x_pool']),)
    for value, member in zip(before, args['committee'].estimators_, strict=True):
        np.testing.assert_array_equal(member.tree_.value, value)


def test_public_adapter_reports_host_success_instead_of_native_fallback():
    args = inputs()
    actual, disclosures = score_industry_pool(strategy='core_set', **args)
    assert _SKACTIVEML_SUCCESS_DISCLOSURE in disclosures
    np.testing.assert_allclose(actual, oracle('core_set', args))


def test_empty_pool_returns_an_empty_score_vector():
    args = inputs()
    args['x_pool'] = np.empty((0, 3))
    assert _score_with_skactiveml(strategy='core_set', **args).shape == (0,)


@pytest.mark.parametrize('strategy', ['qbc_kl', 'unknown'])
def test_invalid_host_query_fails_explicitly(strategy):
    args = inputs()
    args['committee'] = None
    with pytest.raises(ValueError):
        _score_with_skactiveml(strategy=strategy, **args)
