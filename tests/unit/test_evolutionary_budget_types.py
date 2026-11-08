"""Budget configuration must fail before fitting, without truncating values."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.model.selection import evolutionary_search


@pytest.mark.parametrize('name', [
    'population_size', 'n_generations', 'elite_size', 'tournament_size', 'max_evaluations',
])
@pytest.mark.parametrize('value', [True, np.bool_(True), 2.5, float('nan'), float('inf'), '3'])
def test_invalid_budget_type_refused_before_fit(name, value, monkeypatch):
    session = (Session.ingest(pd.DataFrame({'x': np.arange(40), 'y': [0, 1] * 20}))
               .set_roles({'y': 'target'}).split(test_size=.2, random_state=1))
    fitting = Mock(side_effect=AssertionError('Invalid budget started fitting'))
    monkeypatch.setattr(DecisionTreeClassifier, 'fit', fitting)
    options = dict(population_size=3, n_generations=2, elite_size=1,
                   tournament_size=2, max_evaluations=4)
    options[name] = value
    with pytest.raises(ValidationError, match=f'{name} must be an integer'):
        evolutionary_search(session.dataset, session.split_plan, DecisionTreeClassifier(),
                            param_space={'max_depth': [1, 2]}, cv=2, refit=False, **options)
    fitting.assert_not_called()


def test_numpy_integer_budgets_remain_supported():
    session = (Session.ingest(pd.DataFrame({'x': np.arange(40), 'y': [0, 1] * 20}))
               .set_roles({'y': 'target'}).split(test_size=.2, random_state=1))
    result = evolutionary_search(
        session.dataset, session.split_plan, DecisionTreeClassifier(random_state=1),
        param_space={'max_depth': [1, 2]}, population_size=np.int64(3),
        n_generations=np.int64(2), elite_size=np.int64(1), tournament_size=np.int64(2),
        max_evaluations=np.int64(4), cv=2, refit=False,
    )
    assert 1 <= len(result.trials) <= 4
    assert np.isfinite(result.best_score)
