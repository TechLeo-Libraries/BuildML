"""Weighted blending agrees with independent holdout fits and preserves prototypes."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from buildml.ensemble.blending import HoldoutBlendClassifier, HoldoutBlendRegressor


@pytest.mark.parametrize('passthrough', [False, True])
@pytest.mark.parametrize('refit', [False, True])
@pytest.mark.parametrize('pandas_input', [False, True])
def test_weighted_blend_matches_independent_holdout_solution(passthrough, refit, pandas_input):
    rng = np.random.default_rng(36)
    x = rng.normal(size=(100, 3))
    y = 3*x[:, 0] - 2*x[:, 1] + rng.normal(size=100)
    weights = np.linspace(.1, 5, 100)
    prototypes = [('linear', LinearRegression()), ('ridge', Ridge(alpha=3))]
    inputs = (pd.DataFrame(x), pd.Series(y), pd.Series(weights)) if pandas_input else (x, y, weights)
    blend = HoldoutBlendRegressor(estimators=prototypes, holdout_fraction=.25,
        random_state=9, passthrough=passthrough, refit_bases_on_full_train=refit)
    blend.fit(inputs[0], inputs[1], sample_weight=inputs[2])
    base, meta = train_test_split(np.arange(100), test_size=.25, random_state=9)
    reference = [clone(est).fit(x[base], y[base], sample_weight=weights[base])
                 for _, est in prototypes]
    meta_x = np.column_stack([est.predict(x[meta]) for est in reference])
    if passthrough:
        meta_x = np.column_stack([meta_x, x[meta]])
    final = Ridge().fit(meta_x, y[meta], sample_weight=weights[meta])
    np.testing.assert_allclose(blend.final_estimator_.coef_, final.coef_)
    if refit:
        reference = [clone(est).fit(x, y, sample_weight=weights) for _, est in prototypes]
    probe = x[:8] + .2
    probe_meta = np.column_stack([est.predict(probe) for est in reference])
    if passthrough:
        probe_meta = np.column_stack([probe_meta, probe])
    np.testing.assert_allclose(blend.predict(probe), final.predict(probe_meta))
    assert blend.blend_holdout_rows_ == 25
    assert blend.blend_base_rows_ == 75
    for name, prototype in prototypes:
        assert not hasattr(prototype, 'coef_')
        np.testing.assert_allclose(blend.named_estimators_[name].coef_,
                                   reference[[n for n, _ in prototypes].index(name)].coef_)


def test_unweighted_only_base_is_accepted_with_weighted_meta_learner():
    x = np.arange(120).reshape(40, 3)
    y = np.sin(x[:, 0])
    blend = HoldoutBlendRegressor(estimators=[('knn', KNeighborsRegressor(2)),
        ('ridge', Ridge())], random_state=4, refit_bases_on_full_train=False)
    blend.fit(x, y, sample_weight=np.linspace(1, 3, 40))
    assert np.isfinite(blend.predict(x[:4])).all()


@pytest.mark.parametrize('fraction', [0, .049, .5, float('nan')])
def test_invalid_holdout_is_refused_before_fitting(fraction):
    blend = HoldoutBlendRegressor(estimators=[('a', Ridge()), ('b', Ridge())],
                                 holdout_fraction=fraction)
    with pytest.raises(ValueError, match='holdout_fraction'):
        blend.fit(np.ones((20, 2)), np.arange(20))
    assert not hasattr(blend, 'estimators_')


def test_blend_requires_multiple_bases_and_fitting_before_prediction():
    blend = HoldoutBlendRegressor(estimators=[('a', Ridge())])
    with pytest.raises(NotFittedError):
        blend.predict(np.ones((2, 2)))
    with pytest.raises(ValueError, match='at least two'):
        blend.fit(np.ones((20, 2)), np.arange(20))


@pytest.mark.parametrize('method', ['predict', 'predict_proba'])
def test_unfitted_classifier_reports_not_fitted(method):
    with pytest.raises(NotFittedError):
        getattr(HoldoutBlendClassifier(), method)(np.ones((2, 2)))


class _FailingWeightedRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise TypeError('weighted backend failed internally')
        return super().fit(X, y)


@pytest.mark.parametrize('placement', ['base', 'meta'])
def test_internal_weighted_fit_errors_are_not_retried_without_weights(placement):
    base = _FailingWeightedRidge() if placement == 'base' else Ridge()
    final = _FailingWeightedRidge() if placement == 'meta' else Ridge()
    blend = HoldoutBlendRegressor(estimators=[('a', base), ('b', Ridge())],
                                 final_estimator=final, random_state=4)
    with pytest.raises(TypeError, match='weighted backend failed internally'):
        blend.fit(np.arange(120).reshape(40, 3), np.arange(40),
                  sample_weight=np.ones(40))
    assert not hasattr(blend, 'estimators_')
