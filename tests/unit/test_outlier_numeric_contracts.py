"""Outlier fences preserve numerical meaning and positional populations."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.preprocess.apply import apply_preprocess_plans
from buildml.preprocess.outliers import apply_outlier_plan, fit_outlier_plan


def population():
    frame = pd.DataFrame({'x': [0., 1., 2., 3., 4., 5., -100., 100., 2., 3.],
                          'y': np.arange(10), 'record': list('abcdefghij')}, index=[7]*10)
    dataset = Session.ingest(frame).set_roles({'x': 'feature', 'y': 'target', 'record': 'id'}).dataset
    split = SplitPlan('injected', None, None, 0, None, tuple(range(6)), (6, 8), (7, 9))
    return dataset, split, frame


@pytest.mark.parametrize('method', ['iqr', 'zscore'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_invalid_fence_parameters_are_refused(method, value):
    dataset, split, _ = population()
    name = 'iqr_multiplier' if method == 'iqr' else 'zscore_threshold'
    with pytest.raises(ValidationError):
        fit_outlier_plan(dataset, split, method=method, **{name: value})


@pytest.mark.parametrize('method', ['iqr', 'zscore'])
@pytest.mark.parametrize('value', [float('inf'), -float('inf')])
def test_nonfinite_train_values_cannot_generate_nan_fences(method, value):
    dataset, split, _ = population()
    dataset._ensure_pandas().iloc[0, 0] = value
    with pytest.raises(ValidationError):
        fit_outlier_plan(dataset, split, method=method)


@pytest.mark.parametrize('action', ['cap', 'drop'])
@pytest.mark.parametrize('with_split', [False, True])
@pytest.mark.parametrize('bad', ['not a number', float('inf')])
def test_replay_refuses_lossy_or_nonfinite_numeric_input(action, with_split, bad):
    dataset, split, _ = population()
    plan = fit_outlier_plan(dataset, split, action=action)
    frame = pd.DataFrame({'x': [1., bad, None], 'y': [0, 1, 0]})
    before = frame.copy(deep=True)
    score_split = SplitPlan('injected', None, None, 0, None, (0,), (), (1, 2)) if with_split else None
    with pytest.raises(ValidationError):
        apply_preprocess_plans(frame, split_plan=score_split, plans={'outliers': plan})
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('method', ['iqr', 'zscore'])
@pytest.mark.parametrize('action', ['detect', 'cap', 'drop'])
def test_frozen_fences_and_partition_rewrites_match_numpy_positions(method, action):
    dataset, split, frame = population()
    plan = fit_outlier_plan(dataset, split, method=method, action=action)
    train = frame.x.to_numpy()[:6]
    if method == 'iqr':
        q1, q3 = np.quantile(train, [.25, .75])
        lo, hi = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    else:
        lo, hi = train.mean()-3*train.std(), train.mean()+3*train.std()
    assert plan.lower_['x'] == pytest.approx(lo)
    assert plan.upper_['x'] == pytest.approx(hi)
    result, new_split, updated, _ = apply_outlier_plan(dataset, split, plan)
    if action == 'drop':
        keep = np.flatnonzero((frame.x >= lo) & (frame.x <= hi))
        expected = frame.iloc[keep].reset_index(drop=True)
        mapping = {old: new for new, old in enumerate(keep)}
        for old, new in [(split.train_indices, new_split.train_indices),
                         (split.validation_indices, new_split.validation_indices),
                         (split.test_indices, new_split.test_indices)]:
            assert new == tuple(mapping[i] for i in old if i in mapping)
        assert updated.n_dropped == 2
    else:
        expected = frame.copy()
        if action == 'cap':
            expected['x'] = np.clip(frame.x, lo, hi)
        assert new_split is split
    pd.testing.assert_frame_equal(result._ensure_pandas(), expected)
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), frame)


@pytest.mark.parametrize('action', ['cap', 'drop', 'detect'])
def test_score_only_replay_preserves_rows_missing_values_and_fitted_fences(action):
    dataset, split, _ = population()
    plan = fit_outlier_plan(dataset, split, action=action)
    before = (dict(plan.lower_), dict(plan.upper_), plan.action)
    frame = pd.DataFrame({'x': ['-100', '2', '100', None]}, index=[4, 4, 8, 1])
    result = apply_preprocess_plans(frame, plans={'outlier_plan': plan})
    actual = result.dataset._ensure_pandas()
    assert len(actual) == 4
    if action != 'detect':
        expected = np.clip(np.array([-100., 2., 100., np.nan]), plan.lower_['x'], plan.upper_['x'])
        np.testing.assert_allclose(actual.x.to_numpy(dtype=float), expected, equal_nan=True)
    else:
        pd.testing.assert_frame_equal(actual, frame)
    assert (plan.lower_, plan.upper_, plan.action) == before
    assert actual.index.equals(frame.index)
    assert bool(result.warnings) == (action == 'drop')
