"""Subsampling preserves row identity and isolates holdout membership."""
import numpy as np
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.tda.subsample import apply_train_subsample


def _inputs():
    plan = SplitPlan(kind='random', test_size=.2, validation_size=.2,
                     random_state=3, stratify_column=None,
                     train_indices=(6, 2, 9, 1, 8, 4), validation_indices=(0, 3),
                     test_indices=(5, 7))
    frame = pd.DataFrame({'row_id':plan.train_indices, 'target':['a','a','a','b','b','b']},
                         index=['custom_'+str(i) for i in plan.train_indices])
    return plan, frame


@pytest.mark.parametrize('strategy', ['random', 'stratified'])
def test_subsampling_uses_train_positions_not_pandas_index_labels(strategy):
    plan, frame = _inputs()
    original = frame.copy(deep=True)
    reduced, data, disclosures, warnings = apply_train_subsample(plan, frame,
        max_points=4, strategy=strategy, target_column='target', random_state=18)
    assert len(data) == len(reduced.train_indices) == 4
    assert tuple(data.row_id) == reduced.train_indices
    assert set(reduced.train_indices) <= set(plan.train_indices)
    assert reduced.validation_indices == plan.validation_indices
    assert reduced.test_indices == plan.test_indices
    assert disclosures and not warnings
    if strategy == 'stratified':
        assert data.target.value_counts().to_dict() == {'a':2, 'b':2}
    repeated, repeated_data, _, _ = apply_train_subsample(plan, frame,
        max_points=4, strategy=strategy, target_column='target', random_state=18)
    assert repeated.train_indices == reduced.train_indices
    pd.testing.assert_frame_equal(data, repeated_data)
    pd.testing.assert_frame_equal(frame, original)


def test_missing_stratification_target_falls_back_to_reproducible_random_rows():
    plan, frame = _inputs()
    fallback, data, _, warnings = apply_train_subsample(plan, frame, max_points=3,
        strategy='stratified', target_column='absent', random_state=8)
    random, reference, _, _ = apply_train_subsample(plan, frame, max_points=3,
        strategy='random', target_column=None, random_state=8)
    assert warnings and 'requires a target' in warnings[0]
    assert fallback.train_indices == random.train_indices
    pd.testing.assert_frame_equal(data, reference)


@pytest.mark.parametrize('cap', [0, -1, True, 2.5, '3', float('nan')])
def test_invalid_caps_are_refused_before_sampling(cap):
    plan, frame = _inputs()
    with pytest.raises(ValidationError, match='max_points'):
        apply_train_subsample(plan, frame, max_points=cap, strategy='random',
                              target_column=None, random_state=0)


def test_unknown_strategy_and_misaligned_frame_are_refused_even_below_cap():
    plan, frame = _inputs()
    with pytest.raises(ValidationError, match='subsample_strategy'):
        apply_train_subsample(plan, frame, max_points=20, strategy='typo',
                              target_column=None, random_state=0)
    with pytest.raises(ValidationError, match='train'):
        apply_train_subsample(plan, frame.iloc[:-1], max_points=20, strategy='random',
                              target_column=None, random_state=0)


def test_unchanged_rows_and_explicit_size_guard():
    plan, frame = _inputs()
    unchanged, data, disclosures, warnings = apply_train_subsample(plan, frame,
        max_points=np.int64(6), strategy='error', target_column=None, random_state=0)
    assert unchanged is plan and data is frame
    assert not disclosures and not warnings
    with pytest.raises(ValidationError, match='Train size'):
        apply_train_subsample(plan, frame, max_points=3, strategy='error',
                              target_column=None, random_state=0)


@pytest.mark.parametrize('cap,labels', [(1, ['a']*3+['b']*3),
                                      (4, ['a']+['b']*5), (5, ['a']*3+['b']*3)])
def test_stratification_fills_capacity_without_duplicate_rows(cap, labels):
    plan, frame = _inputs()
    frame['target'] = labels
    reduced, data, _, _ = apply_train_subsample(plan, frame, max_points=cap,
        strategy='stratified', target_column='target', random_state=18)
    assert len(data) == cap == len(set(reduced.train_indices))
    assert tuple(data.row_id) == reduced.train_indices
    if labels.count('a') == 1:
        assert data.target.value_counts().to_dict() == {'b':3, 'a':1}


def test_real_tda_stratified_fit_uses_only_selected_train_rows():
    pytest.importorskip('ripser')
    pytest.importorskip('persim')
    from buildml import Session

    frame = pd.DataFrame({'row_id':np.arange(20), 'x':np.random.default_rng(9).normal(size=20),
                          'target':[0,1]*10}, index=['custom_'+str(i) for i in range(20)])
    def make_session(data):
        return Session.ingest(data).set_roles({'row_id':'feature', 'x':'feature',
            'target':'target'}).split(test_size=.2, validation_size=.2,
                                     random_state=2, stratify=True)
    session = make_session(frame)
    options = dict(knn=3, maxdim=0, homology_dims=(0,), n_bins=4, n_layers=1,
                   head='none', vectorization='landscape', standardize=False,
                   max_points_guard=6, subsample_strategy='stratified', random_state=8)
    session.tda.fit(**options)
    plan = session.tda_plan
    assert plan.n_train_rows == 6
    selected = set(plan.train_x_[:, 0].astype(int))
    assert selected <= set(session.split_plan.train_indices)
    heldout = list(session.split_plan.validation_indices + session.split_plan.test_indices)
    altered = frame.copy()
    altered.iloc[heldout, 1] = 1e6
    repeated = make_session(altered)
    repeated.tda.fit(**options)
    np.testing.assert_array_equal(plan.train_x_, repeated.tda_plan.train_x_)
    np.testing.assert_array_equal(plan.train_tda_features_, repeated.tda_plan.train_tda_features_)
