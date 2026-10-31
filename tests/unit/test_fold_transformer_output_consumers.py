"""Real sklearn outputs preserve sparse design semantics and row identity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from buildml.core.errors import ValidationError
from buildml.preprocess.fold import transform_fold_features


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('composite', [False, True])
@pytest.mark.parametrize('duplicate_index', [False, True])
def test_real_encoder_output_fits_ridge_against_independent_design(sparse, composite, duplicate_index):
    train = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                          'number': np.arange(9, dtype=float)})
    query = pd.DataFrame({'category': ['c', 'unknown', 'a'], 'number': [4., 100., -2.]},
                         index=['q', 'q', 'r'] if duplicate_index else ['q', 'r', 's'])
    before = query.copy(deep=True)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)
    if composite:
        transformer = ColumnTransformer([('category', encoder, ['category']),
                                         ('number', StandardScaler(), ['number'])],
                                        sparse_threshold=1. if sparse else 0.)
        names = ['category__category_a', 'category__category_b', 'category__category_c', 'number__number']
        fit_input, query_input = train, query
    else:
        transformer = encoder
        names = ['category_a', 'category_b', 'category_c']
        fit_input, query_input = train[['category']], query[['category']]
    transformer.fit(fit_input)
    transformed_train = transform_fold_features(transformer, fit_input)
    transformed_query = transform_fold_features(transformer, query_input)
    expected_train = (train.category.to_numpy()[:, None] == np.array(['a', 'b', 'c'])).astype(float)
    expected_query = (query.category.to_numpy()[:, None] == np.array(['a', 'b', 'c'])).astype(float)
    if composite:
        expected_train = np.column_stack([expected_train, (train.number-train.number.mean())/train.number.std(ddof=0)])
        expected_query = np.column_stack([expected_query, (query.number-train.number.mean())/train.number.std(ddof=0)])
    assert list(transformed_query.columns) == names
    assert transformed_query.index.equals(query.index)
    if sparse:
        assert all(isinstance(dtype, pd.SparseDtype) for dtype in transformed_query.dtypes)
        np.testing.assert_allclose(transformed_query.sparse.to_coo().toarray(), expected_query)
    else:
        np.testing.assert_allclose(transformed_query.to_numpy(), expected_query)
    target = np.array([1., 4., -2., 2., 3., -1., 1., 2., -3.])
    fitted = Ridge(alpha=.6, fit_intercept=False, solver='lsqr', tol=1e-12).fit(transformed_train, target)
    coefficients = np.linalg.solve(expected_train.T @ expected_train + .6*np.eye(expected_train.shape[1]),
                                    expected_train.T @ target)
    np.testing.assert_allclose(fitted.predict(transformed_query), expected_query @ coefficients, atol=1e-10)
    pd.testing.assert_frame_equal(query, before)


@pytest.mark.parametrize('pandas_output', [False, True])
def test_dense_and_pandas_scaler_outputs_keep_names_and_index(pandas_output):
    train = pd.DataFrame({'a': [1., 3., 8.], 'b': [2., 4., 9.]})
    query = pd.DataFrame({'a': [-2., 10.], 'b': [1., 7.]}, index=['first', 'second'])
    transformer = StandardScaler().set_output(transform='pandas' if pandas_output else 'default').fit(train)
    actual = transform_fold_features(transformer, query)
    expected = (query-train.mean())/train.std(ddof=0)
    pd.testing.assert_frame_equal(actual, expected)


def test_unnamed_array_transformer_uses_positional_feature_names():
    train = pd.DataFrame({'a': [1., 3., 8.], 'b': [2., 4., 9.]}, index=['a', 'b', 'c'])
    transformer = FunctionTransformer(lambda values: np.asarray(values)**2).fit(train)
    actual = transform_fold_features(transformer, train)
    pd.testing.assert_frame_equal(actual, pd.DataFrame(train.to_numpy()**2, index=train.index))


@pytest.mark.parametrize('change', ['reverse', 'drop_row', 'reset_index'])
def test_dataframe_transformers_cannot_change_row_identity_silently(change):
    frame = pd.DataFrame({'a': [1., 2., 3.]}, index=['a', 'b', 'c'])
    functions = {'reverse': lambda x: x.iloc[::-1], 'drop_row': lambda x: x.iloc[:-1],
                 'reset_index': lambda x: x.reset_index(drop=True)}
    transformer = FunctionTransformer(functions[change]).fit(frame)
    with pytest.raises(ValidationError, match='row|index'):
        transform_fold_features(transformer, frame)


def test_real_encoder_duplicate_feature_names_are_refused():
    frame = pd.DataFrame({'a_b': ['c', 'c'], 'a': ['b_c', 'b_c']})
    transformer = OneHotEncoder(sparse_output=False).fit(frame)
    assert list(transformer.get_feature_names_out()) == ['a_b_c', 'a_b_c']
    with pytest.raises(ValidationError, match='duplicate|unique'):
        transform_fold_features(transformer, frame)


@pytest.mark.parametrize('failure', ['rows', 'names'])
def test_array_shape_and_feature_name_mismatches_are_reported_as_validation_errors(failure):
    frame = pd.DataFrame({'a': [1., 2., 3.], 'b': [4., 5., 6.]})
    transformer = (FunctionTransformer(lambda x: np.asarray(x)[:-1]) if failure == 'rows'
                   else FunctionTransformer(np.asarray, feature_names_out=lambda self, names: ['a']))
    transformer.fit(frame)
    with pytest.raises(ValidationError, match='shape|names|rows'):
        transform_fold_features(transformer, frame)
