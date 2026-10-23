"""Native AutoML nested estimates honor fold populations and evaluation weights."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.preprocess import PreprocessRecipe


def predictions(fit, score, alpha):
    design = np.column_stack([np.ones(len(fit)), fit[['x', 'z']].to_numpy()])
    target_design = np.column_stack([np.ones(len(score)), score[['x', 'z']].to_numpy()])
    weighted = design.T * fit.weight.to_numpy()
    coefficients = np.linalg.solve(weighted @ design + np.diag([0., alpha, alpha]),
                                    weighted @ fit.y.to_numpy())
    return target_design @ coefficients


def weighted_score(data, prediction, metric):
    error = data.y.to_numpy()-prediction
    if metric == 'mae':
        return np.average(abs(error), weights=data.weight)
    mse = np.average(error**2, weights=data.weight)
    if metric == 'mse':
        return mse
    variance = np.average((data.y-np.average(data.y, weights=data.weight))**2, weights=data.weight)
    return 1-mse/variance


@pytest.mark.parametrize('mode', ['kfold', 'group_role', 'group_explicit', 'time', 'custom_inner'])
@pytest.mark.parametrize('metric', ['mae', 'mse', 'r2'])
def test_nested_automl_matches_independent_weighted_selection_and_outer_scoring(mode, metric):
    rng = np.random.default_rng(82)
    n = 120
    clock = np.arange(n)
    x, z = rng.normal(size=(2, n))
    frame = pd.DataFrame({'x': x, 'z': z, 'y': 2*x-.4*z+.03*clock+rng.normal(size=n),
                          'weight': np.where(clock % 7 == 0, 40., 1.),
                          'cluster': clock//10,
                          'clock': pd.Timestamp('2025-01-01')+pd.to_timedelta(clock, unit='D')},
                         index=[f'original-{i}' for i in range(n)]).sample(frac=1, random_state=4)
    roles = {'x': 'feature', 'z': 'feature', 'y': 'target', 'weight': 'weight',
             'cluster': 'group' if mode == 'group_role' else 'ignore',
             'clock': 'time' if mode == 'time' else 'ignore'}
    s = Session.ingest(frame).set_roles(roles).split(test_size=.2, validation_size=.1, random_state=3)
    before = s.to_pandas().copy(deep=True)
    train = s.partition('train').reset_index(drop=True)
    if mode == 'time':
        train = train.sort_values('clock', kind='mergesort').reset_index(drop=True)
        outer_splits = TimeSeriesSplit(3).split(train)
        strategy = 'time'
    elif mode.startswith('group'):
        outer_splits = GroupKFold(3).split(train, groups=train.cluster)
        strategy = 'group'
    else:
        outer_splits = KFold(3, shuffle=True, random_state=42).split(train)
        strategy = 'kfold'
    expected = []
    for fit_pos, eval_pos in outer_splits:
        fit, score = train.iloc[fit_pos], train.iloc[eval_pos]
        candidates = []
        for alpha in [.1, 1., 10.]:
            if mode == 'time':
                splits = TimeSeriesSplit(2).split(fit)
            elif mode.startswith('group'):
                splits = GroupKFold(2).split(fit, groups=fit.cluster)
            else:
                splits = KFold(2, shuffle=mode != 'custom_inner',
                               random_state=None if mode == 'custom_inner' else 7).split(fit)
            scores = []
            for inner_fit, inner_eval in splits:
                fitting, scoring = fit.iloc[inner_fit], fit.iloc[inner_eval]
                if mode.startswith('group'):
                    assert set(fitting.cluster).isdisjoint(scoring.cluster)
                if mode == 'time':
                    assert fitting.clock.max() < scoring.clock.min()
                scores.append(weighted_score(scoring, predictions(fitting, scoring, alpha), metric))
            candidates.append((np.mean(scores), alpha))
        best = (max if metric == 'r2' else min)(candidates, key=lambda value: value[0])
        expected.append(weighted_score(score, predictions(fit, score, best[1]), metric))
    options = {}
    if mode == 'group_explicit':
        options['groups'] = s.partition('train').cluster.reset_index(drop=True)
    result = s.automl.run(method='grid', selection='nested', task='regression',
        families=('ridge',), n_trials=3, cv=KFold(2) if mode == 'custom_inner' else 2,
        outer_cv=3, cv_strategy=strategy, ranking_metric=metric, random_state=7,
        include_recipe_search=False, preprocess=PreprocessRecipe(impute=None), **options)
    assert result.outer_score_mean == pytest.approx(np.mean(expected), rel=1e-9, abs=1e-10)
    assert result.outer_score_std == pytest.approx(np.std(expected, ddof=1), rel=1e-9, abs=1e-10)
    assert s.automl_plan.outer_score_mean == result.outer_score_mean
    pd.testing.assert_frame_equal(s.to_pandas(), before)


@pytest.mark.parametrize('strategy', ['group', 'stratified_group', 'time', 'stratified', 'kfold',
                                      'GroupKFold', 'StratifiedGroupKFold'])
def test_inner_fold_construction_preserves_declared_population_constraints(strategy):
    from buildml.automl.search import _nested_inner_splits
    x = pd.DataFrame({'x': np.arange(60.)})
    y = pd.Series([0, 1]*30)
    grouped = strategy in {'group', 'stratified_group', 'GroupKFold', 'StratifiedGroupKFold'}
    groups = pd.Series(np.repeat(np.arange(10), 6)) if grouped else None
    splits = _nested_inner_splits(x, y, cv=3, strategy=strategy, groups=groups,
                                  task='classification', random_state=7)
    assert len(splits) == 3
    for fit, score in splits:
        assert len(fit) and len(score) and set(fit).isdisjoint(score)
        if grouped:
            assert set(groups.iloc[fit]).isdisjoint(groups.iloc[score])
        if strategy == 'time':
            assert max(fit) < min(score)
        if strategy in {'stratified', 'stratified_group', 'StratifiedGroupKFold'}:
            assert set(y.iloc[fit]) == set(y.iloc[score]) == {0, 1}


class InvalidInnerSplitter:
    def __init__(self, kind):
        self.kind = kind

    def split(self, x, y=None, groups=None):
        if self.kind == 'no_folds':
            return
        cases = {'negative': ([-1, 0], [19]), 'duplicate': ([0, 0], [2, 3]),
                 'out_of_bounds': ([0, 20], [2, 3]), 'fractional': ([.5, 1.5], [2, 3]),
                 'boolean': ([True, False], [2, 3]), 'two_dimensional': ([[0, 1]], [2, 3]),
                 'overlap': ([0, 1], [1, 2])}
        yield cases[self.kind]


@pytest.mark.parametrize('kind', ['negative', 'duplicate', 'out_of_bounds', 'fractional',
                                 'boolean', 'two_dimensional', 'overlap', 'no_folds'])
def test_invalid_inner_fold_membership_is_refused(kind):
    from buildml.automl.search import _nested_inner_splits
    with pytest.raises((ValidationError, LeakageError), match='fold|indices'):
        _nested_inner_splits(pd.DataFrame({'x': np.arange(20.)}), pd.Series(np.arange(20.)),
            cv=InvalidInnerSplitter(kind), strategy='kfold', groups=None,
            task='regression', random_state=7)


def test_grouped_inner_selection_requires_aligned_groups():
    from buildml.automl.search import _nested_inner_splits
    with pytest.raises(ValidationError, match='groups'):
        _nested_inner_splits(pd.DataFrame({'x': np.arange(20.)}), pd.Series(np.arange(20.)),
            cv=2, strategy='group', groups=None, task='regression', random_state=7)
