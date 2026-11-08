"""Evolutionary search respects budgets, holdouts and metric direction."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.automl.types import AutoMLBudget


@pytest.mark.parametrize('task,metric,family', [('classification', 'accuracy', 'decision_tree'), ('regression', 'rmse', 'ridge')])
@pytest.mark.parametrize('budget', [1, 2, 8])
def test_evolutionary_search_caps_trials_and_ranks_correctly(task, metric, family, budget):
    rng = np.random.default_rng(532)
    x = rng.normal(size=120)
    z = rng.normal(size=120)
    y = (x + z > 0).astype(int) if task == 'classification' else 2 * x + .3 * z
    session = Session.ingest(pd.DataFrame({'x': x, 'z': z, 'y': y})).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'}).split(test_size=.2, validation_size=.2, random_state=5)
    holdout = session.partition('test').copy()
    result = session.automl.run(backend='native', method='evolutionary', task=task,
        ranking_metric=metric, families=(family,), n_trials=budget, cv=2,
        include_recipe_search=False, budget=AutoMLBudget(max_trials=budget), random_state=6)
    assert 0 < len(result.trials) <= budget
    scores = [trial.mean_score for trial in result.trials if np.isfinite(trial.mean_score)]
    optimum = min(scores) if metric == 'rmse' else max(scores)
    assert result.best_score == pytest.approx(optimum)
    board = result.leaderboard()
    assert board.iloc[0]['mean_score'] == pytest.approx(optimum)
    assert (board['gap_to_best'] >= -1e-12).all()
    pd.testing.assert_frame_equal(session.partition('test'), holdout)
    incomplete = replace(result.trials[0], trial=999, mean_score=float('nan'))
    with_missing = replace(result, trials=(*result.trials, incomplete))
    board = with_missing.leaderboard()
    assert board.iloc[-1]['trial'] == 999
    assert np.isfinite(board.iloc[0]['mean_score'])
