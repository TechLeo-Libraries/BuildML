"""Restored agent models retain their frozen numerical contracts."""

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


def restore(source, empty, domain, path):
    before = empty.to_pandas().copy(deep=True)
    run(source, f'save_{domain}_bundle', path=str(path))
    run(empty, f'load_{domain}_bundle', path=str(path), trusted=True)
    pd.testing.assert_frame_equal(empty.to_pandas(), before)
    return empty


@pytest.mark.parametrize('method,norm', [('transe', 'l1'), ('transe', 'l2'), ('distmult', 'l1')])
def test_agent_kg_restored_scores_match_embedding_algebra(method, norm, tmp_path):
    frame = pd.DataFrame([(f'e{a}', f'r{r}', f'e{b}')
                          for a in range(8) for b in range(8) for r in range(2)
                          if a != b], columns=['head', 'relation', 'tail'])
    def session():
        return (Session.ingest(frame).set_roles(dict.fromkeys(frame.columns, 'id'))
                .split(test_size=.2, validation_size=.2, random_state=23))
    original = session()
    snapshot = original.to_pandas().copy(deep=True)
    run(original, 'fit_kg', backend='native', method=method, norm=norm,
        head_column='head', relation_column='relation', tail_column='tail',
        embedding_dim=5, epochs=3, batch_size=16, random_state=3)
    plan = original.kg_plan
    entities = dict(zip(plan.entity_ids, plan.entity_embeddings_, strict=True))
    relations = dict(zip(plan.relation_ids, plan.relation_embeddings_, strict=True))
    expected = []
    test = frame.iloc[list(original.split_plan.test_indices)]
    for row in test.itertuples():
        h, r, t = entities[row.head], relations[row.relation], entities[row.tail]
        expected.append(np.sum(h*r*t) if method == 'distmult' else
                        -np.linalg.norm(h+r-t, ord=1 if norm == 'l1' else 2))
    scored = run(original, 'score_triples', partition='test')
    assert scored.unknown_entities == scored.unknown_relations == 0
    np.testing.assert_allclose(scored.scores, expected, rtol=1e-10, atol=1e-10)
    restored = restore(original, session(), 'kg', tmp_path / method / norm)
    again = run(restored, 'score_triples', partition='test')
    np.testing.assert_allclose(again.scores, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(restored.kg_plan.entity_embeddings_, plan.entity_embeddings_)
    pd.testing.assert_frame_equal(original.to_pandas(), snapshot)


@pytest.mark.parametrize('method', ['item_knn', 'user_knn'])
@pytest.mark.parametrize('exclude', [False, True])
def test_agent_recommender_restored_ranks_match_train_only_cosine_oracle(method, exclude, tmp_path):
    rng = np.random.default_rng(81)
    rows = [(f'u{u}', f'i{i}', float(rng.integers(1, 6)))
            for u in range(20) for i in rng.choice(12, size=8, replace=False)]
    frame = pd.DataFrame(rows, columns=['user', 'item', 'rating'])
    def session():
        return (Session.ingest(frame).set_roles({'user': 'id', 'item': 'id', 'rating': 'target'})
                .split(test_size=.2, validation_size=.2, random_state=13))
    original = session()
    run(original, 'fit_recommender', backend='sklearn', method=method,
        user_column='user', item_column='item', rating_column='rating',
        feedback='explicit', n_neighbors=100)
    plan = original.recommender_plan
    train = frame.iloc[list(original.split_plan.train_indices)]
    matrix = (train.pivot(index='user', columns='item', values='rating')
              .reindex(index=plan.user_ids, columns=plan.item_ids).fillna(0).to_numpy())
    np.testing.assert_allclose(plan.matrix_, matrix)
    vectors = matrix.T if method == 'item_knn' else matrix
    lengths = np.linalg.norm(vectors, axis=1)
    similarity = np.divide(vectors@vectors.T, lengths[:, None]*lengths[None, :],
                           out=np.zeros((len(vectors), len(vectors))),
                           where=lengths[:, None]*lengths[None, :] > 0)
    users = {user: i for i, user in enumerate(plan.user_ids)}
    items = {item: i for i, item in enumerate(plan.item_ids)}
    def check(session):
        result = run(session, 'recommend', partition='test', k=5, exclude_train_items=exclude)
        assert not result.cold_start_users
        for user, recommended, scores in zip(result.user_ids, result.recommendations,
                                             result.scores, strict=True):
            index = users[user]
            if method == 'item_knn':
                rated = matrix[index] != 0
                weights = similarity[:, rated]
                denominator = np.abs(weights).sum(axis=1)
                expected = np.divide(weights@matrix[index, rated], denominator,
                                     out=np.zeros(len(items)), where=denominator > 1e-12)
            else:
                weights = similarity[index].copy()
                weights[index] = 0
                denominator = np.abs(weights)@(matrix != 0)
                expected = np.divide(weights@matrix, denominator,
                                     out=np.zeros(len(items)), where=denominator > 1e-12)
            eligible = np.ones(len(items), dtype=bool)
            if exclude:
                eligible = matrix[index] == 0
            positions = [items[item] for item in recommended]
            assert all(eligible[positions])
            np.testing.assert_allclose(scores, expected[positions], atol=1e-10)
            assert len(positions) == min(5, int(eligible.sum()))
            assert min(scores) >= np.sort(expected[eligible])[-len(scores)]-1e-10
        return result
    before = check(original)
    restored = restore(original, session(), 'recommender', tmp_path / method)
    after = check(restored)
    assert after.recommendations == before.recommendations
    assert after.scores == before.scores


@pytest.mark.parametrize('latent_dim', [2, 4])
def test_agent_ssl_restores_latent_activations_and_supervised_head(latent_dim, tmp_path):
    rng = np.random.default_rng(52)
    x = rng.normal(size=(96, 2))
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'y': (x[:, 0]+x[:, 1] > 0).astype(int)})
    def session():
        return (Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
                .split(test_size=.2, validation_size=.2, random_state=7))
    original = session()
    with pytest.warns(DeprecationWarning, match='legacy sklearn MLP'):
        run(original, 'fit_ssl_pretext', method='masked_tabular', latent_dim=latent_dim,
            max_iter=200, random_state=3)
    run(original, 'finetune_ssl_head', estimator='logistic_regression')
    encoder = original.ssl_plan.encoder_
    test = frame.iloc[list(original.split_plan.test_indices)]
    activation = test[['x', 'z']].to_numpy()
    for layer in range(len(encoder.hidden)+1):
        activation = np.maximum(activation@encoder.mlp_.coefs_[layer]
                                 +encoder.mlp_.intercepts_[layer], 0.)
    np.testing.assert_allclose(encoder.transform(test[['x', 'z']]), activation, atol=1e-12)
    prediction = original._ssl_head_plan.estimator_.predict(activation)
    expected_accuracy = np.mean(prediction == test.y)
    before = run(original, 'evaluate_ssl', partition='test')
    assert before.metrics['accuracy'] == pytest.approx(expected_accuracy)
    restored = restore(original, session(), 'ssl', tmp_path / 'ssl')
    np.testing.assert_allclose(restored.ssl_plan.encoder_.transform(test[['x', 'z']]),
                               activation, atol=1e-12)
    after = run(restored, 'evaluate_ssl', partition='test')
    assert after.metrics == pytest.approx(before.metrics)
