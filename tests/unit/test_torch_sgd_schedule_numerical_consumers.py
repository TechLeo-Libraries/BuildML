"""Real Torch SGD, schedules and resume agree with independent scalar recurrences."""

from dataclasses import replace
from math import cos, pi

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.loaders import make_loaders
from buildml.dl.train import train_supervised_module
from buildml.dl.types import LoaderConfig, TrainConfig


def _bundle(constant):
    pytest.importorskip("torch")
    x = np.zeros(6) if constant else np.array([-2., -1., -.5, .5, 1., 2.])
    y = np.ones(6) if constant else 2*x
    frame = pd.DataFrame({'x': np.tile(x, 3), 'y': np.tile(y, 3)})
    s = Session.ingest(frame).set_roles({'x': 'feature', 'y': 'target'})
    s.inject_split(train_indices=list(range(6)), validation_indices=list(range(6, 12)),
                   test_indices=list(range(12, 18)))
    bundle = make_loaders(s.dataset, s.split_plan, task='regression',
                          config=LoaderConfig(batch_size=6, shuffle_train=False, normalize=False))
    return bundle, x, y


def _module(weight=.3):
    import torch
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(weight)
    return module


def _optimizer(params):
    import torch
    return torch.optim.SGD(params, lr=.1, momentum=.4)


def _config(scheduler, clip, epochs):
    return TrainConfig(epochs=epochs, device='cpu', scheduler=scheduler,
                       scheduler_step_size=2, scheduler_gamma=.5, scheduler_t_max=8,
                       scheduler_patience=0, scheduler_factor=.5, scheduler_threshold=0,
                       grad_clip_norm=clip, restore_best_weights=False)


def _oracle(x, y, scheduler, clip):
    weight, velocity, lr = .3, 0., .1
    best = float('inf')
    rows = []
    for epoch in range(1, 9):
        train_loss = np.mean((weight*x-y)**2)
        gradient = 2*np.mean((weight*x-y)*x)
        if clip is not None:
            gradient *= min(1., clip/(abs(gradient)+1e-6))
        velocity = .4*velocity + gradient
        weight -= lr*velocity
        val_loss = np.mean((weight*x-y)**2)
        rows.append((epoch, train_loss, val_loss, lr))
        if scheduler == 'step':
            lr = .1*.5**(epoch//2)
        elif scheduler == 'cosine':
            lr = .1*(1+cos(pi*epoch/8))/2
        elif val_loss < best:
            best = val_loss
        else:
            lr *= .5
    return weight, velocity, rows, lr


@pytest.mark.parametrize('scheduler', ['step', 'cosine', 'plateau'])
@pytest.mark.parametrize('clip', [None, .2])
@pytest.mark.parametrize('constant', [False, True])
def test_sgd_schedule_and_resume_match_scalar_oracle(scheduler, clip, constant):
    bundle, x, y = _bundle(constant)
    expected_weight, velocity, rows, next_lr = _oracle(x, y, scheduler, clip)
    uninterrupted = train_supervised_module(_module(), bundle, config=_config(scheduler, clip, 8),
                                             optimizer_factory=_optimizer)
    prior = train_supervised_module(_module(), bundle, config=_config(scheduler, clip, 3),
                                    optimizer_factory=_optimizer)
    resumed = train_supervised_module(_module(999), bundle, config=_config(scheduler, clip, 5),
                                      optimizer_factory=_optimizer, resume_from=prior)
    for result in (uninterrupted, resumed):
        assert result.n_epochs_ran == 8
        assert float(result.module.weight.detach()[0, 0]) == pytest.approx(expected_weight, rel=2e-6, abs=2e-7)
        actual = [[row['epoch'], row['train_loss'], row['val_loss'], row['lr']] for row in result.history]
        np.testing.assert_allclose(actual, rows, rtol=3e-6, atol=2e-7)
        group = result.optimizer_state['param_groups'][0]
        assert group['lr'] == pytest.approx(next_lr)
        state = next(iter(result.optimizer_state['state'].values()))
        assert float(state['momentum_buffer'][0, 0]) == pytest.approx(velocity, rel=3e-6, abs=2e-7)
    assert prior.n_epochs_ran == 3 and len(prior.history) == 3
    assert resumed.resumed_from_epochs == 3
    assert resumed.scheduler_state == uninterrupted.scheduler_state


@pytest.mark.parametrize('field,value,message', [('feature_columns', ('different',), 'feature columns'),
                                               ('target_column', 'other', 'target column'),
                                               ('task', 'classification', 'task mismatch')])
def test_resume_contract_refusal_precedes_weight_restore(field, value, message):
    bundle, _, _ = _bundle(False)
    prior = train_supervised_module(_module(), bundle, config=_config('step', None, 1),
                                    optimizer_factory=_optimizer)
    changed = replace(bundle, contract=replace(bundle.contract, **{field: value}))
    destination = _module(999)
    with pytest.raises(ValidationError, match=message):
        train_supervised_module(destination, changed, config=_config('step', None, 1),
                                 optimizer_factory=_optimizer, resume_from=prior)
    assert float(destination.weight.detach()[0, 0]) == 999

@pytest.mark.parametrize('mode', ['min', 'max'])
def test_resume_preserves_elapsed_early_stopping_patience_with_sparse_history(mode):
    bundle, _, _ = _bundle(True)
    config = replace(_config('step', None, 8), early_stopping_patience=2,
                     early_stopping_mode=mode, restore_best_weights=True, log_every=3)
    uninterrupted = train_supervised_module(_module(), bundle, config=config,
                                             optimizer_factory=_optimizer)
    prior = train_supervised_module(_module(), bundle, config=replace(config, epochs=2),
                                    optimizer_factory=_optimizer)
    resumed = train_supervised_module(_module(999), bundle, config=replace(config, epochs=6),
                                      optimizer_factory=_optimizer, resume_from=prior)
    assert uninterrupted.n_epochs_ran == 3
    assert resumed.n_epochs_ran == 3
    assert resumed.early_stop.triggered
    assert resumed.early_stop.best_epoch == 1
    assert resumed.early_stop.best_value == 1
    assert float(resumed.module.weight.detach()[0, 0]) == pytest.approx(.3)


@pytest.mark.parametrize('change', [{'early_stopping_mode': 'max'},
                                   {'early_stopping_monitor': 'train_loss'},
                                   {'early_stopping_min_delta': .2}])
def test_changed_early_stopping_monitor_starts_new_window(change):
    bundle, _, _ = _bundle(True)
    config = replace(_config('step', None, 2), early_stopping_patience=2,
                     restore_best_weights=True)
    prior = train_supervised_module(_module(), bundle, config=config, optimizer_factory=_optimizer)
    resumed = train_supervised_module(_module(999), bundle,
                                      config=replace(config, epochs=6, **change),
                                      optimizer_factory=_optimizer, resume_from=prior)
    assert resumed.n_epochs_ran == 5
    assert resumed.early_stop.best_epoch == 3
    assert any('fresh patience window' in warning for warning in resumed.warnings)
