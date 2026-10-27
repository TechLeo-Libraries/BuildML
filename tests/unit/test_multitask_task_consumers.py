"""Explicit task meaning survives real learners, metrics and saved consumers."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)

from buildml import Session
from buildml.data.splits import SplitPlan

SEED = 511
POPULATIONS = ['regression-low', 'regression-mixed', 'regression-ordinary',
               'classification-high', 'classification-mixed', 'classification-ordinary']


def population(name):
    rng = np.random.default_rng(329)
    x = rng.normal(size=(80, 3))
    x[48:] += [.8, -.4, .3]
    frame = pd.DataFrame(x, columns=['a', 'b', 'c'], index=[f'row-{i // 2}' for i in range(80)])
    task, shape = name.split('-')
    for i in range(3):
        if task == 'regression':
            values = x[:, i] * (i + 1) + rng.normal(scale=.05, size=80)
            if shape == 'low' or (shape == 'mixed' and i == 0):
                values = np.where(x[:, i] > 0, 5. + 3 * i, -2. - i)
            frame[f't{i}'] = values
        else:
            if shape == 'high' or (shape == 'mixed' and i == 0):
                values = (np.arange(80) + i * 3) % 12 + 10
                frame[f't{i}'] = pd.Series(values, dtype='Int64').to_numpy()
                frame[f't{i}'] = frame[f't{i}'].astype('Int64')
            elif i == 1:
                values = x[:, i] > 0
            else:
                values = np.where(x[:, i] > 0, '001', '010')
            if not (shape == 'high' or (shape == 'mixed' and i == 0)):
                frame[f't{i}'] = pd.Series(values, dtype=object).to_numpy()
    # Missing labels are excluded from scoring, while prediction retains all rows.
    frame.iloc[51, frame.columns.get_loc('t0')] = None
    frame.iloc[69, frame.columns.get_loc('t2')] = None
    s = Session.ingest(frame).set_roles({f't{i}': 'target' for i in range(3)})
    s._split_plan = SplitPlan('injected', .2, .2, None, None,
                              tuple(range(48)), tuple(range(48, 64)), tuple(range(64, 80)))
    return s, frame, task


def options(learner, task, targets):
    common = dict(task=task, targets=targets, random_state=SEED,
                  prefer_reduce_components=False)
    if learner == 'torch':
        return common | dict(backend='torch', method='shared_trunk_multihead',
                             epochs=1, batch_size=48, learning_rate=.001, device='cpu')
    if learner == 'lightgbm':
        return common | dict(backend='industry', method='multi_output_lgbm')
    return common | dict(backend='sklearn', method=(
        'multi_output' if learner == 'multi_output' else
        'regressor_chain' if task == 'regression' else 'classifier_chain'),
        order=None if learner == 'multi_output' else list(reversed(targets)),
        base_estimator='ridge' if task == 'regression' else 'logistic_regression')


def encode_reference(frame, targets, task):
    vocabularies = {}
    if task == 'regression':
        return frame[targets].to_numpy(dtype=float), vocabularies
    encoded = []
    for target in targets:
        vocabulary = [value.item() if isinstance(value, np.generic) else value for value in sorted(set(frame[target]))]
        vocabularies[target] = vocabulary
        encoded.append([vocabulary.index(value) for value in frame[target]])
    return np.column_stack(encoded), vocabularies


def torch_reference(x, y, task, vocabularies, targets):
    """Independent network and one AdamW step, without BuildML model helpers."""
    import torch
    from torch import nn

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.heads = nn.ModuleList([nn.Linear(64, len(vocabularies[t]) if task == 'classification' else 1)
                                        for t in targets])

        def forward(self, features):
            hidden = self.trunk(features)
            return [head(hidden) for head in self.heads]

    torch.manual_seed(SEED)
    model = Reference()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    permutation = np.random.default_rng(SEED).permutation(len(x))
    features = torch.tensor(np.asarray(x, dtype=np.float32)[permutation])
    outputs = model(features)
    losses = []
    for i, output in enumerate(outputs):
        if task == 'classification':
            losses.append(nn.functional.cross_entropy(output, torch.tensor(y[permutation, i], dtype=torch.long)))
        else:
            losses.append(nn.functional.mse_loss(output.squeeze(-1), torch.tensor(y[permutation, i], dtype=torch.float32)))
    sum(losses).backward()
    optimizer.step()
    model.eval()
    return model


def reference_fit(learner, task, frame, targets):
    x = frame[['a', 'b', 'c']].to_numpy(dtype=float)
    y, vocabularies = encode_reference(frame, targets, task)
    if learner == 'torch':
        return torch_reference(x, y, task, vocabularies, targets), vocabularies
    if learner == 'lightgbm':
        from lightgbm import LGBMClassifier, LGBMRegressor
        factory = LGBMClassifier if task == 'classification' else LGBMRegressor
        # Fit separate scalar consumers; no BuildML multi-target wrapper is reused.
        models = [factory(n_estimators=160, learning_rate=.08, num_leaves=31,
                          random_state=SEED, verbose=-1).fit(x, y[:, i]) for i in range(len(targets))]
        return models, vocabularies
    base = Ridge(random_state=SEED) if task == 'regression' else LogisticRegression(max_iter=500, random_state=SEED)
    if learner == 'chain':
        wrapper = RegressorChain if task == 'regression' else ClassifierChain
        model = wrapper(base, order=list(reversed(range(len(targets)))), random_state=SEED)
    else:
        model = MultiOutputRegressor(base) if task == 'regression' else MultiOutputClassifier(base)
    return model.fit(x, y), vocabularies


def reference_predict(reference, learner, task, frame, targets, vocabularies):
    x = frame[['a', 'b', 'c']].to_numpy(dtype=float)
    if learner == 'torch':
        import torch
        with torch.no_grad():
            outputs = reference(torch.tensor(x, dtype=torch.float32))
        raw = np.column_stack([o.argmax(1).numpy() if task == 'classification' else o[:, 0].numpy() for o in outputs])
    elif learner == 'lightgbm':
        raw = np.column_stack([model.predict(x) for model in reference])
    else:
        raw = reference.predict(x)
    return {target: ([vocabularies[target][int(code)] for code in raw[:, i]] if task == 'classification'
                     else raw[:, i]) for i, target in enumerate(targets)}


def macro_f1(truth, predicted):
    truth, predicted = list(map(str, truth)), list(map(str, predicted))
    scores = []
    for label in set(truth) | set(predicted):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return np.mean(scores)


@pytest.mark.parametrize('name', POPULATIONS)
@pytest.mark.parametrize('learner', ['multi_output', 'chain', 'lightgbm', 'torch'])
@pytest.mark.parametrize('reverse_targets', [False, True])
@pytest.mark.parametrize('replay', [False, True])
def test_explicit_tasks_match_independent_learners_and_frozen_consumers(name, learner, reverse_targets, replay, tmp_path):
    if learner == 'lightgbm':
        pytest.importorskip('lightgbm')
    if learner == 'torch':
        torch = pytest.importorskip('torch')
        torch.manual_seed(SEED)
    source, frame, task = population(name)
    targets = ['t2', 't1', 't0'] if reverse_targets else ['t0', 't1', 't2']
    source.multitask.fit(**options(learner, task, targets))
    plan = source.multitask_plan
    assert plan.task_kinds_ == dict.fromkeys(targets, task)
    reference, vocabularies = reference_fit(learner, task, frame.iloc[:48], targets)
    consumer = source
    if replay:
        consumer = Session(dataset=source.dataset, split_plan=source.split_plan)
        consumer.multitask.load_bundle(source.multitask.save_bundle(tmp_path / 'multi'), trusted=True)
    before = joblib.hash(consumer.multitask_plan)
    for partition, positions in [('validation', range(48, 64)), ('test', range(64, 80)), ('all', range(80))]:
        block = frame.iloc[list(positions)]
        expected = reference_predict(reference, learner, task, block, targets, vocabularies)
        predictions = consumer.multitask.predict(partition=partition).predictions
        for target in targets:
            if task == 'classification':
                assert list(predictions[target]) == expected[target]
                assert [type(v) for v in predictions[target]] == [type(v) for v in expected[target]]
            else:
                np.testing.assert_allclose(predictions[target], expected[target], atol=2e-6, rtol=2e-6)
        scored = block.dropna(subset=targets)
        scored_expected = reference_predict(reference, learner, task, scored, targets, vocabularies)
        result = consumer.multitask.evaluate(partition=partition)
        assert result.n_rows == len(scored)
        errors = []
        for target in targets:
            metrics = result.per_task_metrics[target]
            if task == 'classification':
                accuracy = np.mean(scored[target].astype(str).to_numpy() == np.asarray(list(map(str, scored_expected[target]))))
                assert metrics['accuracy'] == pytest.approx(accuracy)
                assert metrics['f1_macro'] == pytest.approx(macro_f1(scored[target], scored_expected[target]))
                errors.append(accuracy)
            else:
                truth = scored[target].to_numpy(dtype=float)
                residuals = truth - scored_expected[target]
                assert metrics['mae'] == pytest.approx(np.mean(abs(residuals)), abs=2e-6)
                assert metrics['rmse'] == pytest.approx(np.sqrt(np.mean(residuals ** 2)), abs=2e-6)
                assert metrics['r2'] == pytest.approx(1 - np.sum(residuals ** 2) / np.sum((truth - truth.mean()) ** 2), abs=2e-6)
                errors.append(np.mean(abs(residuals)))
        assert result.metrics['mean_accuracy' if task == 'classification' else 'mean_mae'] == pytest.approx(np.mean(errors), abs=2e-6)
        assert not ({'mean_mae', 'mean_accuracy'} - {('mean_accuracy' if task == 'classification' else 'mean_mae')}) & result.metrics.keys()
    attached = consumer.multitask.predict(partition='all', attach=True, prediction_prefix='held')
    assert attached.attached
    output = consumer.to_pandas()
    pd.testing.assert_frame_equal(output[frame.columns], frame)
    for target in targets:
        assert len(output['held_' + target]) == len(frame)
    assert joblib.hash(consumer.multitask_plan) == before
