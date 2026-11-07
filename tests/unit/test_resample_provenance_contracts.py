"""Real resampling preserves donor metadata and untouched holdout populations."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.preprocess.imbalance import resample_train

pytest.importorskip('imblearn')
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def population(metadata=True, duplicate_index=False, validation=True, categorical=True):
    rng = np.random.default_rng(77)
    frame = pd.DataFrame(rng.normal(size=(80, 2)), columns=['x', 'z'])
    frame['y'] = [0]*48 + [1]*16 + [0, 1]*8
    roles = {'x': 'feature', 'z': 'feature', 'y': 'target'}
    if metadata:
        frame['record'] = pd.array(range(80), dtype='Int64')
        frame['weight'] = np.arange(80, dtype=float) + .5
        frame['site'] = pd.Categorical(['east', 'west'] * 40) if categorical else ['east', 'west'] * 40
        frame['time'] = pd.date_range('2020-01-01', periods=80, tz='UTC')
        frame['note'] = [f'row {i}' for i in range(80)]
        roles.update(record='id', weight='weight', site='group', time='time', note='ignore')
    if duplicate_index:
        frame.index = [5]*80
    session = Session.ingest(frame).set_roles(roles)
    split = SplitPlan('injected', None, None, 3, None, tuple(range(64)),
                      tuple(range(64, 72)) if validation else (),
                      tuple(range(72 if validation else 64, 80)))
    return session.dataset, split, frame


@pytest.mark.parametrize('name,sampler_cls', [
    ('random_oversample', RandomOverSampler), ('random_undersample', RandomUnderSampler)])
@pytest.mark.parametrize('duplicate_index', [False, True])
@pytest.mark.parametrize('validation', [False, True])
@pytest.mark.parametrize('categorical', [False, True])
def test_random_resampling_preserves_each_donor_row(name, sampler_cls, duplicate_index, validation, categorical):
    dataset, split, frame = population(duplicate_index=duplicate_index, validation=validation, categorical=categorical)
    before = frame.copy(deep=True)
    reference = sampler_cls(random_state=17)
    reference.fit_resample(frame.iloc[:64][['x', 'z']], frame.iloc[:64].y)
    expected = frame.iloc[:64].iloc[reference.sample_indices_].reset_index(drop=True)
    result, plan, report = resample_train(dataset, split, sampler=name, random_state=17)
    actual = result._ensure_pandas()
    pd.testing.assert_frame_equal(actual.iloc[list(plan.train_indices)].reset_index(drop=True), expected)
    for old_indices, new_indices in [(split.validation_indices, plan.validation_indices),
                                     (split.test_indices, plan.test_indices)]:
        pd.testing.assert_frame_equal(actual.iloc[list(new_indices)].reset_index(drop=True),
                                      frame.iloc[list(old_indices)].reset_index(drop=True))
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), before)
    assert result.roles == dataset.roles
    assert report.n_train_after == len(reference.sample_indices_)
    assert report.class_counts_after == expected.y.astype(str).value_counts().to_dict()
    assert not (set(plan.train_indices) & set(plan.test_indices))


@pytest.mark.parametrize('name', ['smote', 'adasyn', 'borderline_smote'])
def test_synthesis_cannot_invent_identifiers_groups_times_or_sample_weights(name):
    dataset, split, frame = population()
    with pytest.raises(ValidationError, match='auxiliary|metadata'):
        resample_train(dataset, split, sampler=name)
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), frame)


@pytest.mark.parametrize('name,sampler_cls', [
    ('smote', SMOTE), ('adasyn', ADASYN), ('borderline_smote', BorderlineSMOTE)])
@pytest.mark.parametrize('strategy', ['auto', .8])
def test_synthetic_features_match_backend_and_preserve_holdouts(name, sampler_cls, strategy):
    dataset, split, frame = population(metadata=False, duplicate_index=True)
    sampler = sampler_cls(random_state=13, sampling_strategy=strategy)
    expected_x, expected_y = sampler.fit_resample(frame.iloc[:64][['x', 'z']], frame.iloc[:64].y)
    result, plan, report = resample_train(dataset, split, sampler=name,
                                         random_state=13, sampling_strategy=strategy)
    actual = result._ensure_pandas()
    train = actual.iloc[list(plan.train_indices)]
    np.testing.assert_allclose(train[['x', 'z']], expected_x, rtol=0, atol=0)
    np.testing.assert_array_equal(train.y, expected_y)
    pd.testing.assert_frame_equal(actual.iloc[list(plan.test_indices)].reset_index(drop=True),
                                  frame.iloc[list(split.test_indices)].reset_index(drop=True))
    assert report.n_train_before == 64
    assert report.n_train_after == len(expected_y)


@pytest.mark.parametrize('name', ['random_oversample', 'random_undersample'])
def test_explicitly_non_feature_columns_cannot_be_used_as_fallback_features(name):
    dataset, split, _ = population(metadata=False)
    dataset.set_roles({'x': 'id', 'z': 'ignore'})
    with pytest.raises(ValidationError, match='feature'):
        resample_train(dataset, split, sampler=name)


@pytest.mark.parametrize('name,sampler_cls', [
    ('random_oversample', RandomOverSampler), ('random_undersample', RandomUnderSampler)])
def test_resampled_session_fits_with_the_actual_donor_weights(name, sampler_cls):
    from sklearn.linear_model import LogisticRegression

    dataset, split, frame = population(duplicate_index=True)
    reference = sampler_cls(random_state=11)
    reference.fit_resample(frame.iloc[:64][['x', 'z']], frame.iloc[:64].y)
    donors = frame.iloc[:64].iloc[reference.sample_indices_]
    expected = LogisticRegression(max_iter=200).fit(
        donors[['x', 'z']], donors.y, sample_weight=donors.weight.to_numpy())
    session = Session.ingest(frame).set_roles(dict(dataset.roles))
    session._split_plan = split
    session.resample(sampler=name, random_state=11)
    session.fit(LogisticRegression(max_iter=200), task='classification')
    assert session.fit_result.feature_columns == ('x', 'z')
    assert session.fit_result.weight_column == 'weight'
    np.testing.assert_allclose(session.fit_result.estimator.coef_, expected.coef_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(session.fit_result.estimator.intercept_, expected.intercept_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(session.fit_result.estimator.predict_proba(frame.iloc[72:][['x', 'z']]),
                               expected.predict_proba(frame.iloc[72:][['x', 'z']]), rtol=1e-10)
