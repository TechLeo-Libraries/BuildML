"""Real Torch consistency training: seeded state, numerical consumers and refits."""
from copy import deepcopy

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.semisupervised.adapters.torch_consistency import TabularConsistencyClassifier

torch = pytest.importorskip('torch')


def data():
    x = np.random.default_rng(19).normal(size=(16, 3)).astype('float32')
    y = np.array([2, 7, -1, -1] * 4)
    return x, y


@pytest.mark.parametrize('method', ['fixmatch_tabular', 'mixmatch_tabular'])
def test_seed_covers_initialization_dropout_and_preserves_ambient_rng(method):
    x, y = data()
    options = dict(method=method, epochs=2, batch_size=8, random_state=41, threshold=0.)
    torch.manual_seed(100)
    before = torch.get_rng_state().clone()
    first = TabularConsistencyClassifier(**options).fit(x, y)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(900)
    second = TabularConsistencyClassifier(**options).fit(x, y)
    for key, value in first.module_.state_dict().items():
        torch.testing.assert_close(value, second.module_.state_dict()[key], rtol=0, atol=0)
    probs = first.predict_proba(x)
    with torch.no_grad():
        logits = first.module_(torch.as_tensor(x)).numpy()
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(probs, exp / exp.sum(axis=1, keepdims=True), rtol=1e-6)
    np.testing.assert_array_equal(first.predict(x), np.array([2, 7])[probs.argmax(axis=1)])
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    if method == 'fixmatch_tabular':
        assert first.n_pseudo_labels_ == 16


@pytest.mark.parametrize('field,value', [
    ('epochs', 0), ('epochs', 1.5), ('epochs', True),
    ('batch_size', 0), ('batch_size', 1), ('batch_size', 2.5),
    ('learning_rate', float('nan')), ('learning_rate', 0),
    ('consistency_weight', -1), ('mixup_alpha', 0),
    ('threshold', -0.1), ('threshold', float('nan')), ('method', 'unknown'),
])
def test_invalid_configuration_refuses_before_fitted_state_changes(field, value):
    x, y = data()
    model = TabularConsistencyClassifier(epochs=1, batch_size=8).fit(x, y)
    original = model.module_
    classes = model.classes_.copy()
    setattr(model, field, value)
    with pytest.raises(ValidationError):
        model.fit(x[:, :2], y)
    assert model.module_ is original
    assert model.n_features_ == 3
    np.testing.assert_array_equal(model.classes_, classes)


@pytest.mark.parametrize('case', ['nan', 'infinite', 'width', 'length', 'fractional_labels'])
def test_invalid_demonstrations_preserve_a_fitted_model(case):
    x, y = data()
    model = TabularConsistencyClassifier(epochs=1, batch_size=8).fit(x, y)
    original = model.module_
    weights = deepcopy(original.state_dict())
    if case == 'nan':
        x[0, 0] = np.nan
    elif case == 'infinite':
        x[0, 0] = np.inf
    elif case == 'width':
        x = x[:, :0]
    elif case == 'length':
        y = y[:-1]
    else:
        y = y.astype(float)
        y[0] = 2.5
    with pytest.raises(ValidationError):
        model.fit(x, y)
    assert model.module_ is original
    for key, value in weights.items():
        torch.testing.assert_close(original.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('method', ['fixmatch_tabular', 'mixmatch_tabular'])
def test_train_partition_matches_independent_fit_despite_poisoned_holdout(method):
    import pandas as pd

    from buildml import Session
    from buildml.data.dataset import Dataset
    from buildml.ingest.detect import schema_from_dataframe
    from buildml.semisupervised.fit import fit_semisupervised

    rng = np.random.default_rng(22)
    frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=['a', 'b', 'c'])
    frame['y'] = np.arange(40) % 2
    session = Session.ingest(frame).set_roles({'y': 'target'}).split(
        test_size=0.25, stratify=True, random_state=5)
    train = list(session.split_plan.train_indices)
    holdout = list(session.split_plan.test_indices)
    frame.loc[train[2::3], 'y'] = np.nan
    frame.index = [4] * len(frame)
    options = dict(backend='torch', method=method, columns=['a', 'b', 'c'],
                   epochs=2, batch_size=8, threshold=0., random_state=13,
                   prefer_reduce_components=False)
    plans = []
    for poison in [False, True]:
        current = frame.copy()
        if poison:
            current.iloc[holdout, :3] = np.nan
            current.iloc[holdout, 3] = 999
        dataset = Dataset.from_transformed(session.dataset, current,
            schema=schema_from_dataframe(current), roles=dict(session.dataset.roles))
        plan, result = fit_semisupervised(dataset, session.split_plan, **options)
        assert result.n_train_rows == len(train)
        assert result.n_unlabeled_train == len(train[2::3])
        assert set(result.classes) == {0.0, 1.0}
        plans.append(plan)
    direct_y = frame.iloc[train]['y'].to_numpy()
    direct_y = np.where(np.isnan(direct_y), -1, direct_y).astype(int)
    direct = TabularConsistencyClassifier(method=method, epochs=2, batch_size=8,
                                          threshold=0., random_state=13).fit(
        frame.iloc[train][['a', 'b', 'c']].to_numpy(), direct_y)
    for plan in plans:
        for key, value in direct.module_.state_dict().items():
            torch.testing.assert_close(value, plan.estimator_.module_.state_dict()[key],
                                       rtol=0, atol=0)
