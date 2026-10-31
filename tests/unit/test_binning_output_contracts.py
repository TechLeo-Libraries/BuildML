"""Frozen binning must preserve output ownership and missingness."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.preprocess.apply import apply_preprocess_plans
from buildml.preprocess.binning import fit_binning, transform_binning


def population(values=None):
    values = np.arange(10, dtype=float) if values is None else np.asarray(values)
    frame = pd.DataFrame({'x': values, 'y': np.arange(len(values)) % 2}, index=[4]*len(values))
    dataset = Session.ingest(frame).set_roles({'x': 'feature', 'y': 'target'}).dataset
    split = SplitPlan('injected', None, None, 1, None, tuple(range(8)), (), tuple(range(8, len(frame))))
    return dataset, split


@pytest.mark.parametrize('encode_as', ['ordinal', 'onehot'])
@pytest.mark.parametrize('role', ['id', 'target', 'weight', 'feature'])
def test_replay_cannot_overwrite_an_existing_generated_column(encode_as, role):
    dataset, split = population()
    plan = fit_binning(dataset, split, columns=['x'], n_bins=3, encode_as=encode_as)
    collision = 'x_bin' if encode_as == 'ordinal' else plan.labels_['x'][0]
    score = pd.DataFrame({'x': [0., 7., 9.], collision: [91, 17, 44]})
    incoming = Session.ingest(score).set_roles({collision: role}).dataset
    before = incoming._ensure_pandas().copy(deep=True)
    with pytest.raises(ValidationError, match='colli|overwrite|already'):
        apply_preprocess_plans(incoming, plans={'binning': plan})
    pd.testing.assert_frame_equal(incoming._ensure_pandas(), before)


@pytest.mark.parametrize('encode_as', ['ordinal', 'onehot'])
@pytest.mark.parametrize('strategy', ['uniform', 'quantile'])
def test_frozen_bins_match_independent_search_and_preserve_missingness(encode_as, strategy):
    dataset, split = population()
    plan = fit_binning(dataset, split, n_bins=3, strategy=strategy, encode_as=encode_as)
    score = pd.DataFrame({'x': [-100., 0., 2., 7., 100., np.nan]}, index=[1, 1, 9, 4, 5, 3])
    before = (list(plan.edges_['x']), list(plan.labels_['x']))
    result = apply_preprocess_plans(score, plans={'binning_plan': plan}).dataset._ensure_pandas()
    codes = np.searchsorted(np.asarray(plan.edges_['x'])[1:-1], score.x.iloc[:-1], side='left')
    if encode_as == 'ordinal':
        np.testing.assert_array_equal(result.x_bin.iloc[:-1], codes)
        assert pd.isna(result.x_bin.iloc[-1])
    else:
        expected = np.eye(len(plan.edges_['x'])-1)[codes]
        np.testing.assert_array_equal(result[plan.labels_['x']].iloc[:-1], expected)
        assert result[plan.labels_['x']].iloc[-1].isna().all()
    assert result.index.equals(score.index)
    assert (plan.edges_['x'], plan.labels_['x']) == before


@pytest.mark.parametrize('strategy', ['uniform', 'quantile'])
def test_learning_edges_uses_finite_train_values_only(strategy):
    dataset, split = population([0., 1., 2., 3., 4., 5., np.inf, -np.inf, -1e6, 1e6])
    plan = fit_binning(dataset, split, n_bins=3, strategy=strategy)
    reference = np.linspace(0., 5., 4) if strategy == 'uniform' else np.quantile(np.arange(6), np.linspace(0, 1, 4))
    np.testing.assert_allclose(plan.edges_['x'][1:-1], reference[1:-1])
    assert np.isfinite(plan.edges_['x'][1:-1]).all()
    result, _ = transform_binning(dataset, plan)
    assert len(result._ensure_pandas()) == 10
    assert result._ensure_pandas().x_bin.iloc[6] == len(plan.edges_['x']) - 2
    assert result._ensure_pandas().x_bin.iloc[7] == 0


@pytest.mark.parametrize('value', [True, False, 2.5, float('nan'), float('inf'), 0])
def test_bin_count_requires_an_integer_at_least_two(value):
    dataset, split = population()
    with pytest.raises(ValidationError):
        fit_binning(dataset, split, n_bins=value)


@pytest.mark.parametrize('strategy', ['quantile', 'uniform'])
def test_constant_train_column_still_has_frozen_extreme_coverage(strategy):
    dataset, split = population([2.]*8 + [-100., 100.])
    plan = fit_binning(dataset, split, n_bins=4, strategy=strategy, encode_as='onehot')
    result, _ = transform_binning(dataset, plan)
    assert plan.edges_['x'] == [-np.inf, np.inf]
    np.testing.assert_array_equal(result._ensure_pandas()[plan.labels_['x']], np.ones((10, 1)))


@pytest.mark.parametrize('strategy', ['quantile', 'uniform'])
def test_no_finite_training_support_is_refused_even_when_holdout_is_finite(strategy):
    dataset, split = population([np.inf, -np.inf, np.nan, np.inf]*2 + [0., 1.])
    with pytest.raises(ValidationError, match='finite train'):
        fit_binning(dataset, split, strategy=strategy)
