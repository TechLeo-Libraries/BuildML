"""Fold PCA dimensional controls, naming and numerical projections are explicit."""

import numpy as np
import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor


def frame():
    rng = np.random.default_rng(28)
    values = rng.normal(size=(30, 3)) @ np.array([[3., 1., 0.], [0., 1., .2], [.1, 0., .1]])
    return pd.DataFrame(values, columns=['a', 'b', 'c'], index=[f'fit-{i}' for i in range(30)])


@pytest.mark.parametrize('components', [None, 2, np.int64(2), 100, .8, np.float32(.8), 1.0])
@pytest.mark.parametrize('drop', [False, True])
def test_fold_pca_matches_independent_svd_subspace_and_frozen_center(components, drop):
    train = frame()
    original = train.copy(deep=True)
    prep = build_fold_preprocessor(train, PreprocessRecipe(impute=None, reduce='pca',
        reduce_n_components=components, reduce_drop_input=drop))
    centered = train.to_numpy() - train.to_numpy().mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if components is None or isinstance(components, (int, np.integer)):
        width = 3 if components is None else min(int(components), 3)
    else:
        width = 3 if components == 1 else int(np.searchsorted(
            np.cumsum(singular**2) / np.sum(singular**2), float(components), side='right')) + 1
    query = train.iloc[[2, 7, 12]].copy() + [900., -600., 100.]
    query_before = query.copy(deep=True)
    actual = prep.transform(query)
    columns = [f'pc_{i+1}' for i in range(width)]
    projected = (query.to_numpy() - train.to_numpy().mean(axis=0)) @ vectors[:width].T
    # A score Gram matrix is invariant to SVD's arbitrary component signs.
    scores = actual[columns].to_numpy()
    np.testing.assert_allclose(scores @ scores.T, projected @ projected.T, rtol=1e-10, atol=1e-8)
    assert prep._reducer.n_samples_ == len(train)
    assert list(actual.columns) == (columns if drop else [*train.columns, *columns])
    assert actual.index.equals(query.index) and actual.columns.is_unique
    if not drop:
        pd.testing.assert_frame_equal(actual[list(train.columns)], query)
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(query, query_before)


@pytest.mark.parametrize('components', [True, np.bool_(False), 0, -1, np.nan, np.inf, '2', 1+0j, [2]])
def test_invalid_component_controls_are_refused(components):
    with pytest.raises(ValidationError, match='reduce_n_components'):
        build_fold_preprocessor(frame(), PreprocessRecipe(impute=None, reduce='pca',
                                                          reduce_n_components=components))


@pytest.mark.parametrize('phase', ['fit', 'transform'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 2+3j])
def test_nonfinite_or_complex_pca_values_are_refused_without_lossy_projection(phase, value):
    train = frame()
    dirty = train.astype(complex) if isinstance(value, complex) else train.copy()
    dirty.iloc[0, 0] = value
    recipe = PreprocessRecipe(impute=None, reduce='pca', reduce_n_components=2)
    with pytest.raises(ValidationError, match='PCA'):
        if phase == 'fit':
            build_fold_preprocessor(dirty, recipe)
        else:
            prep = build_fold_preprocessor(train, recipe)
            prep.transform(dirty)


@pytest.mark.parametrize('phase', ['fit', 'transform'])
@pytest.mark.parametrize('drop', [False, True])
def test_component_names_cannot_collide_with_retained_features(phase, drop):
    train = frame()
    collision = train.assign(pc_1=123.)
    recipe = PreprocessRecipe(impute=None, reduce='pca', reduce_n_components=2,
                              reduce_columns=('a', 'b'), reduce_drop_input=drop)
    with pytest.raises(ValidationError, match='collid|collision'):
        if phase == 'fit':
            build_fold_preprocessor(collision, recipe)
        else:
            build_fold_preprocessor(train, recipe).transform(collision)


def test_a_removed_input_name_can_be_reused_for_its_component():
    train = frame().rename(columns={'a': 'pc_1'})
    prep = build_fold_preprocessor(train, PreprocessRecipe(impute=None, reduce='pca',
        reduce_n_components=2, reduce_drop_input=True))
    actual = prep.transform(train)
    assert list(actual.columns) == ['pc_1', 'pc_2'] and actual.columns.is_unique
