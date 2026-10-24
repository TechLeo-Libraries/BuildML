"""Model-driven resource allocations match exhaustive and fractional oracles."""
import itertools

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session


@pytest.fixture(scope='module', params=['strings', 'integers', 'booleans'])
def fitted(request):
    rng = np.random.default_rng(910)
    x = rng.normal(size=(64, 2))
    positive = x[:, 0] - .7*x[:, 1] + rng.normal(0, .8, 64) > 0
    labels = {'strings': np.where(positive, 'yes', 'no'),
              'integers': np.where(positive, 7, 2), 'booleans': positive}[request.param]
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'target': labels,
                          'expense': 1 + np.arange(64) % 5,
                          'id': [f'row-{i}' for i in range(64)]})
    s = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'target': 'target',
                                        'expense': 'ignore', 'id': 'ignore'})
    s.inject_split(train_indices=list(range(40)), validation_indices=list(range(40, 52)),
                   test_indices=list(range(52, 64)))
    return s.fit(LogisticRegression(C=.4), task='classification')


def oracle(values, costs, method, budget, floor):
    eligible = np.arange(len(values)) if floor is None else np.flatnonzero(values >= floor)
    allocation = np.zeros(len(values))
    if method == 'topk':
        allocation[eligible[np.argsort(-values[eligible], kind='stable')[:3]]] = 1
    elif method == 'knapsack':
        choices = []
        for bits in itertools.product([0, 1], repeat=len(eligible)):
            chosen = eligible[np.asarray(bits, dtype=bool)]
            if costs[chosen].sum() <= budget:
                choices.append((float(values[chosen].sum()), chosen))
        _, chosen = max(choices, key=lambda item: item[0])
        allocation[chosen] = 1
    else:
        remaining = budget
        for index in eligible[np.argsort(-(values[eligible]/costs[eligible]))]:
            amount = min(1., remaining/costs[index])
            allocation[index] = max(0., amount)
            remaining -= amount*costs[index]
    return allocation


@pytest.mark.parametrize('method', ['topk', 'knapsack', 'lp_allocate'])
@pytest.mark.parametrize('partition', ['validation', 'test'])
@pytest.mark.parametrize('budget', [3., 7.])
@pytest.mark.parametrize('floor', [None, .99])
def test_allocation_and_frozen_bundle_metrics_match_independent_oracles(
        fitted, method, partition, budget, floor, tmp_path):
    s = fitted
    data_before = s.to_pandas().copy(deep=True)
    fit_fingerprint = joblib.hash(s.fit_result)
    arguments = dict(method=method, backend='native', partition='validation',
                     score_source='model_proba', cost_column='expense', id_column='id', min_score=floor)
    arguments.update(capacity=3) if method == 'topk' else arguments.update(budget=budget)
    s.decision.fit(**arguments)
    plan_fingerprint = joblib.hash(s.decision_plan)
    frame = s.partition(partition)
    estimator = s.fit_result.estimator
    logits = (frame[['x', 'z']].to_numpy() @ estimator.coef_.T + estimator.intercept_)[:, 0]
    values = 1 / (1 + np.exp(-logits))
    costs = frame.expense.to_numpy()
    allocation = oracle(values, costs, method, budget, floor)
    selected = np.flatnonzero(allocation > 1e-9)
    applied = s.decision.apply(partition=partition)
    evaluated = s.decision.evaluate(partition=partition)
    assert set(applied.selected_indices) == set(selected)
    assert set(applied.selected_ids) == set(frame.id.iloc[selected])
    assert applied.selected_value == pytest.approx(float(allocation @ values))
    assert applied.selected_cost == pytest.approx(float(allocation @ costs))
    assert evaluated.metrics['selected_value'] == pytest.approx(float(allocation @ values))
    assert evaluated.metrics['selected_cost'] == pytest.approx(float(allocation @ costs))
    assert evaluated.n_selected == len(selected)
    if len(selected):
        expected_rate = np.mean(frame.target.iloc[selected].to_numpy() == estimator.classes_[1])
        assert evaluated.metrics['selected_positive_rate'] == pytest.approx(expected_rate)
    else:
        assert 'selected_positive_rate' not in evaluated.metrics
    if method == 'topk':
        assert evaluated.metrics['capacity_utilization'] == pytest.approx(len(selected)/3)
    else:
        assert evaluated.metrics['budget_utilization'] == pytest.approx(float(allocation @ costs)/budget)
    assert any('in-sample' in w for w in evaluated.warnings) is (partition == 'validation')
    assert joblib.hash(s.decision_plan) == plan_fingerprint
    root = s.decision.save_bundle(tmp_path/'allocation')
    s.decision.load_bundle(root, trusted=True)
    assert s.decision.evaluate(partition=partition).metrics == evaluated.metrics
    assert joblib.hash(s.fit_result) == fit_fingerprint
    pd.testing.assert_frame_equal(s.to_pandas(), data_before)
