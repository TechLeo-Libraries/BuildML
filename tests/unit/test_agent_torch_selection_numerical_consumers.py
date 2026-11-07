"""Real agent Torch searches retain train-only selection and numerical scores."""

import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def folds(population, seed, classification):
    if classification:
        from sklearn.model_selection import StratifiedKFold
        return list(StratifiedKFold(2, shuffle=True, random_state=seed).split(
            np.zeros(len(population)), population.y))
    parts = np.array_split(np.random.default_rng(seed).permutation(len(population)), 2)
    return [(parts[1], parts[0]), (parts[0], parts[1])]


def key(params):
    return json.dumps(params, sort_keys=True)


def metric_value(model, fit, evaluation):
    torch = pytest.importorskip('torch')
    linear, = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    x = fit[['x', 'z']].to_numpy()
    values = (evaluation[['x', 'z']].to_numpy()-x.mean(axis=0))/x.std(axis=0)
    prediction = values @ linear.weight.detach().cpu().numpy().T
    prediction += linear.bias.detach().cpu().numpy()
    if prediction.shape[1] > 1:
        return float(np.mean(prediction.argmax(axis=1) == evaluation.y.to_numpy()))
    return float(np.mean((prediction.ravel()-evaluation.y.to_numpy())**2))


@pytest.mark.parametrize('operation', ['search_torch', 'nested_cv_torch'])
@pytest.mark.parametrize('space', ['grid', 'randomized'])
@pytest.mark.parametrize('classification', [False, True])
def test_real_agent_selection_scores_and_outer_refits_match_numpy(operation, space, classification, monkeypatch):
    pytest.importorskip('torch')
    from buildml.data.dataset import Dataset
    from buildml.dl import search

    rng = np.random.default_rng(82)
    x = rng.normal(size=(64, 2))
    y = (x[:, 0] > .3*x[:, 1]).astype(int) if classification else x @ [2., -.7]
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'y': y})
    s = (Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
         .split(test_size=.25, validation_size=.25, random_state=9))
    poisoned = s.to_pandas().copy()
    held = list(s.split_plan.test_indices+s.split_plan.validation_indices)
    poisoned.iloc[held, poisoned.columns.get_loc('x')] = np.nan
    poisoned.iloc[held, poisoned.columns.get_loc('y')] = 9 if classification else 1e12
    s._dataset = Dataset.from_transformed(s.dataset, poisoned)
    snapshot = s.to_pandas().copy(deep=True)
    training = s.partition('train')
    actual_factory = search._default_factory
    models = []
    def trace_actual_factory(width, task, classes, params):
        model = actual_factory(width, task, classes, params)
        models.append((dict(params), model))
        return model
    monkeypatch.setattr(search, '_default_factory', trace_actual_factory)
    options = {'epochs': 1, 'n_iter': 2,
               'param_grid' if space == 'grid' else 'param_distributions': {
                   'hidden': [[]], 'dropout': [0.], 'batch_size': [64],
                   'learning_rate': [.001, .01]}}
    options.update({'n_folds': 2} if operation == 'search_torch' else
                   {'outer_cv': 2, 'inner_cv': 2})
    registry = build_default_registry()
    proposal = propose_tool_execution(operation, options, registry)
    executed = execute_tool(s, proposal, confirmed=True, registry=registry)
    assert executed.executed, executed.error
    result = executed.result
    metric = 'accuracy' if classification else 'mse'
    best = max if classification else min
    assert set(result.held_out_partitions) == {'validation', 'test'}
    def inner_oracles(population, captured, seed):
        expected = {}
        for offset in (0, 2):
            values = []
            params = captured[offset][0]
            for j, (fit, score) in enumerate(folds(population, seed, classification)):
                assert captured[offset+j][0] == params
                values.append(metric_value(captured[offset+j][1], population.iloc[fit], population.iloc[score]))
            expected[key(params)] = (np.mean(values), np.std(values, ddof=0))
        return expected
    if operation == 'search_torch':
        assert len(models) == 4 and len(result.trials) == 2
        expected = inner_oracles(training, models, 0)
        for trial in result.trials:
            mean, std = expected[key(trial.params)]
            assert trial.mean_metrics[metric] == pytest.approx(mean, rel=2e-6, abs=1e-6)
            assert trial.std_metrics[metric] == pytest.approx(std, rel=2e-6, abs=1e-6)
        assert result.best_metrics[metric] == pytest.approx(best(v[0] for v in expected.values()), rel=2e-6)
    else:
        assert len(models) == 10 and len(result.outer_folds) == 2
        outer = []
        for i, ((fit, score), reported) in enumerate(zip(folds(training, 0, classification), result.outer_folds, strict=True)):
            assert reported.train_size == len(fit) and reported.eval_size == len(score)
            captured = models[5*i:5*i+5]
            inner = inner_oracles(training.iloc[fit], captured, 1000*(i+1))
            assert inner[key(reported.best_params)][0] == pytest.approx(best(v[0] for v in inner.values()), rel=2e-6)
            assert reported.inner_best.mean_metrics[metric] == pytest.approx(
                inner[key(reported.best_params)][0], rel=2e-6, abs=1e-6)
            assert captured[-1][0] == reported.best_params
            expected = metric_value(captured[-1][1], training.iloc[fit], training.iloc[score])
            assert reported.outer_metrics[metric] == pytest.approx(expected, rel=2e-6, abs=1e-6)
            outer.append(expected)
        assert result.mean_metrics[metric] == pytest.approx(np.mean(outer), rel=2e-6)
        assert result.std_metrics[metric] == pytest.approx(np.std(outer), rel=2e-6)
    pd.testing.assert_frame_equal(s.to_pandas(), snapshot)
