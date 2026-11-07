"""Session Torch CV must reserve outer holdouts and score its real training folds."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def session(*, split=True, poison=False, classification=False):
    rng = np.random.default_rng(63)
    x = rng.normal(size=(64, 2))
    y = (x[:, 0] > 0).astype(int) if classification else x @ [2., -.4]
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'y': y},
                          index=[f'row-{i}' for i in range(64)])
    s = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
    if split:
        s.split(test_size=.25, validation_size=.25, random_state=7)
    if poison:
        frame = s.to_pandas().copy()
        held = list(s.split_plan.test_indices+s.split_plan.validation_indices)
        frame.iloc[held, frame.columns.get_loc('x')] = np.nan
        frame.iloc[held, frame.columns.get_loc('y')] = 9 if classification else 1e12
        from buildml.data.dataset import Dataset
        s._dataset = Dataset.from_transformed(s.dataset, frame)
    return s


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_real_torch_cv_fold_population_and_mse_match_independent_linear_arithmetic(split, normalize):
    torch = pytest.importorskip('torch')
    s = session(split=split)
    snapshot = s.to_pandas().copy(deep=True)
    training = s.partition('train') if split else snapshot
    models = []
    def factory(width, task, classes):
        model = torch.nn.Linear(width, 1)
        models.append(model)
        return model
    result = s.dl.cross_validate(n_folds=2, epochs=1, batch_size=64, device='cpu',
        seed=17, normalize=normalize, stratify=False, task='regression', module_factory=factory)
    assert all(f.train_size+f.val_size == len(training) for f in result.fold_scores)
    assert sum(f.val_size for f in result.fold_scores) == len(training)
    order = np.random.default_rng(17).permutation(len(training))
    folds = np.array_split(order, 2)
    for i, (model, scored) in enumerate(zip(models, result.fold_scores, strict=True)):
        fit = training.iloc[folds[1-i]][['x', 'z']].to_numpy()
        evaluation = training.iloc[folds[i]]
        values = evaluation[['x', 'z']].to_numpy()
        if normalize:
            values = (values-fit.mean(axis=0))/fit.std(axis=0)
        prediction = values @ model.weight.detach().numpy().T + model.bias.detach().numpy()
        mse = np.mean((prediction.ravel()-evaluation.y.to_numpy())**2)
        assert scored.metrics['mse'] == pytest.approx(mse, rel=2e-6, abs=1e-6)
    assert result.config['population'] == ('train' if split else 'all')
    assert result.config['n_rows'] == len(training)
    assert result.config['held_out_partitions'] == (['validation', 'test'] if split else [])
    pd.testing.assert_frame_equal(s.to_pandas(), snapshot)


@pytest.mark.parametrize('classification', [False, True])
def test_reserved_poisoned_rows_do_not_enter_session_or_agent_torch_cv(classification):
    pytest.importorskip('torch')
    s = session(poison=True, classification=classification)
    before = s.to_pandas().copy(deep=True)
    registry = build_default_registry()
    proposal = propose_tool_execution('cross_validate_torch', {'n_folds': 2, 'epochs': 1}, registry)
    outcome = execute_tool(s, proposal, confirmed=True, registry=registry)
    assert outcome.executed, outcome.error
    result = outcome.result
    assert result.task == ('classification' if classification else 'regression')
    assert sum(f.val_size for f in result.fold_scores) == len(s.split_plan.train_indices)
    assert np.isfinite(list(result.mean_metrics.values())).all()
    pd.testing.assert_frame_equal(s.to_pandas(), before)


@pytest.mark.parametrize('invalid', ['empty', 'test_overlap', 'validation_overlap'])
def test_malformed_outer_split_is_refused_before_any_fold_model_is_created(invalid):
    from dataclasses import replace

    from buildml.core.errors import LeakageError, ValidationError

    s = session()
    plan = s.split_plan
    train = (() if invalid == 'empty' else plan.train_indices +
             ((plan.test_indices if invalid == 'test_overlap' else plan.validation_indices)[0],))
    s._split_plan = replace(plan, train_indices=train)
    def never_create_model(*args):
        raise AssertionError('Malformed outer partitions must be refused before training')
    with pytest.raises((LeakageError, ValidationError)):
        s.dl.cross_validate(n_folds=2, epochs=1, device='cpu', module_factory=never_create_model)
    assert s.dl_cv_result is None
