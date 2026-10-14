"""Invalid search budgets and unsafe folds fail before fitting an estimator."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.model.selection import evolutionary_search
from buildml.preprocess import PreprocessRecipe
from buildml.preprocess.fold import FoldLocalPreprocessor


@pytest.mark.parametrize('options', [
    {'population_size': 1}, {'n_generations': 0}, {'elite_size': 0},
    {'elite_size': 12}, {'crossover_rate': -0.1}, {'crossover_rate': 1.1},
    {'mutation_rate': -0.1}, {'mutation_rate': 1.1}, {'tournament_size': 1},
    {'max_evaluations': 0},
])
def test_invalid_evolutionary_budget_does_not_fit(options, monkeypatch):
    session = Session.ingest(pd.DataFrame({'x': np.arange(40), 'y': [0, 1] * 20})).set_roles(
        {'x': 'feature', 'y': 'target'}
    ).split(test_size=.2, random_state=3)
    fitting = Mock(side_effect=AssertionError('Invalid search started fitting'))
    monkeypatch.setattr(LogisticRegression, 'fit', fitting)
    with pytest.raises(ValidationError):
        evolutionary_search(session.dataset, session.split_plan, LogisticRegression(),
                            param_space={'C': [.1, 1]}, **options)
    fitting.assert_not_called()


@pytest.mark.parametrize('recipe', [
    {'binning': 'quantile', 'n_bins': 1},
    {'text': 'tfidf', 'text_max_features': 0},
    {'text': 'tfidf', 'text_ngram_range': (0, 1)},
    {'text': 'tfidf', 'text_ngram_range': (3, 2)},
    {'reduce': 'unsupported'}, {'reduce': 'pca', 'reduce_prefix': ''},
    {'reduce': 'pca', 'reduce_prefix': 'bad prefix'},
    {'reduce': 'pca', 'reduce_n_components': 0},
    {'reduce': 'pca', 'reduce_n_components': 1.5},
    {'outliers': 'iqr', 'outlier_action': 'remove'},
    {'outliers': 'iqr', 'iqr_multiplier': 0},
    {'select': 'univariate', 'select_columns': ('absent',)},
    {'impute_columns': ('absent',)},
    {'scale': 'standard', 'scale_columns': ('absent',)},
    {'encode': 'onehot', 'encode_columns': ('absent',)},
    {'text': 'count', 'text_columns': ('absent',)},
    {'dates': True, 'date_columns': ('absent',)},
])
def test_invalid_fold_recipe_preserves_input(recipe):
    frame = pd.DataFrame({'x': [1., 2., 3., 4.], 'category': ['a', 'b', 'a', 'b']})
    before = frame.copy(deep=True)
    processor = FoldLocalPreprocessor(PreprocessRecipe(**recipe))
    with pytest.raises(ValidationError):
        processor.fit(frame, pd.Series([0, 1, 0, 1]))
    pd.testing.assert_frame_equal(frame, before)


class _OverlappingSplitter:
    def split(self, x, y=None, groups=None):
        yield np.arange(10), np.arange(5, 15)


@pytest.mark.parametrize('method', ['cv_score', 'nested_cv_score'])
def test_overlapping_cv_indices_rejected_before_fit(method, monkeypatch):
    session = Session.ingest(pd.DataFrame({'x': np.arange(40), 'y': [0, 1] * 20})).set_roles(
        {'x': 'feature', 'y': 'target'}
    ).split(test_size=.2, random_state=3)
    fitting = Mock(side_effect=AssertionError('Leaking fold started fitting'))
    monkeypatch.setattr(LogisticRegression, 'fit', fitting)
    kwargs = {'cv': _OverlappingSplitter()} if method == 'cv_score' else {
        'outer_cv': _OverlappingSplitter(), 'param_grid': {'C': [1]}, 'inner_cv': 2,
    }
    with pytest.raises(LeakageError, match='overlap'):
        getattr(session, method)(LogisticRegression(), **kwargs)
    fitting.assert_not_called()
