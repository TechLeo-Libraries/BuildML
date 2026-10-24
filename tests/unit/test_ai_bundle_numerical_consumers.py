"""Actual restored agent policies must retain their numerical meaning."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def run(session, name, **arguments):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    outcome = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert outcome.executed, outcome.error
    return outcome.result


@pytest.mark.parametrize('alpha', [0., .4])
def test_restored_agent_linucb_matches_independent_logged_normal_equations(alpha, tmp_path):
    rng = np.random.default_rng(112)
    x = rng.normal(size=(96, 2))
    arms = rng.choice(['left', 'right'], size=len(x))
    reward = np.where(arms == 'left', x @ [2., -.5], x @ [-1., 1.5])
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'arm': arms, 'reward': reward})
    def session():
        return (Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature',
                                                'arm': 'target', 'reward': 'feature'})
                .split(test_size=.25, random_state=7))
    original = session()
    snapshot = original.to_pandas().copy(deep=True)
    train = snapshot.iloc[list(original.split_plan.train_indices)]
    test = snapshot.iloc[list(original.split_plan.test_indices)]
    expected_scores, expected_rewards = [], []
    context = test[['x', 'z']].to_numpy()
    for arm in ('left', 'right'):
        logged = train.loc[train.arm == arm]
        design = logged[['x', 'z']].to_numpy()
        matrix = np.eye(2) + design.T @ design
        theta = np.linalg.solve(matrix, design.T @ logged.reward.to_numpy())
        mean = context @ theta
        bonus = alpha*np.sqrt(np.einsum('ij,ji->i', context,
                                        np.linalg.solve(matrix, context.T)))
        expected_rewards.append(mean)
        expected_scores.append(mean+bonus)
    expected_scores = np.column_stack(expected_scores)
    expected_rewards = np.column_stack(expected_rewards)
    chosen = np.argmax(expected_scores, axis=1)
    fit = run(original, 'fit_rl', backend='sklearn', mode='contextual_bandit',
              algorithm='linucb', action_column='arm', reward_column='reward', alpha=alpha)
    assert 'n_arms=2' in fit
    result = run(original, 'act_rl', partition='test', deterministic=True)
    np.testing.assert_allclose(result.scores, expected_scores, atol=1e-10)
    assert list(result.actions) == np.asarray(['left', 'right'])[chosen].tolist()
    evaluation = run(original, 'evaluate_rl', partition='test', deterministic=True)
    assert evaluation.offline
    assert evaluation.metrics['direct_method'] == pytest.approx(
        np.mean(expected_rewards[np.arange(len(test)), chosen]))
    assert evaluation.metrics['action_match_rate'] == pytest.approx(
        np.mean(np.asarray(result.actions) == test.arm.to_numpy()))
    path = str(tmp_path / 'bandit')
    run(original, 'save_rl_bundle', path=path)
    restored = session()
    run(restored, 'load_rl_bundle', path=path, trusted=True)
    again = run(restored, 'act_rl', partition='test', deterministic=True)
    np.testing.assert_allclose(again.scores, expected_scores, atol=1e-10)
    assert again.actions == result.actions
    pd.testing.assert_frame_equal(original.to_pandas(), snapshot)
    pd.testing.assert_frame_equal(restored.to_pandas(), snapshot)


def test_agent_automl_bundle_preserves_selected_model_predictions_and_metrics(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(72, 2))
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'y': x @ [2., -.3]})
    def session():
        return (Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
                .split(test_size=.25, random_state=7))
    original = session()
    result = run(original, 'run_automl', method='grid', task='regression',
                 families=['ridge'], include_recipe_search=False, n_trials=3)
    assert result.best_family == 'ridge'
    before = run(original, 'evaluate_automl', partition='test')
    path = str(tmp_path / 'automl')
    run(original, 'save_automl_bundle', path=path)
    restored = session()
    run(restored, 'load_automl_bundle', path=path, trusted=True)
    after = run(restored, 'evaluate_automl', partition='test')
    assert after.metrics == pytest.approx(before.metrics)
    np.testing.assert_allclose(restored.fit_result.estimator.predict(frame[['x', 'z']]),
                               original.fit_result.estimator.predict(frame[['x', 'z']]))
    assert restored.automl_plan.best_params == original.automl_plan.best_params
