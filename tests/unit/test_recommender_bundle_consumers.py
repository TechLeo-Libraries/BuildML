"""Real fitted recommender restoration preserves catalog identity and scores."""
import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.recommenders.checkpoint import load_recommender_bundle, save_recommender_bundle
from buildml.recommenders.recommend import recommend_for_users


@pytest.fixture(scope='module', params=['item_knn', 'user_knn', 'svd', 'nmf', 'content'])
def fitted(request):
    frame = pd.DataFrame([dict(user=f'u{u}', item=f'i{i}', rating=float(1+(u*3+i*5)%7), f1=float(i%3), f2=float(i)) for u in range(6) for i in range(8)])
    s = Session.ingest(frame).set_roles({'user':'id','item':'id','rating':'target','f1':'feature','f2':'feature'}).split(test_size=.3, validation_size=.15, random_state=7)
    s.recommender.fit(backend='sklearn', method=request.param, user_column='user', item_column='item', rating_column='rating', feedback='explicit', n_neighbors=99, n_factors=3, n_iterations=30, item_feature_columns=['f1','f2'], random_state=5)
    return s.recommender_plan


@pytest.mark.parametrize('payload_kind', ['direct', 'separate'])
def test_restored_catalog_and_scores(fitted, tmp_path, payload_kind):
    root = save_recommender_bundle(tmp_path/'bundle', fitted)
    path = root/'recommender_plan.joblib'
    data = joblib.load(path)
    plan = data['plan']
    # Older artifacts can omit redundant indices; rebuild from stored catalog order.
    plan.user_index_ = {}
    plan.item_index_ = {}
    if payload_kind == 'direct':
        joblib.dump(plan, path)
    else:
        for name in ['matrix_', 'similarity_', 'user_factors_', 'item_factors_', 'item_popularity_', 'item_features_', 'item_feature_mean_', 'item_feature_scale_']:
            setattr(plan, name, None)
        plan.user_ids = ()
        plan.item_ids = ()
        joblib.dump(data, path)
    restored = load_recommender_bundle(root, trusted=True)
    assert restored.user_index_ == {u:i for i,u in enumerate(fitted.user_ids)}
    assert restored.item_index_ == {v:i for i,v in enumerate(fitted.item_ids)}
    for exclude in [True, False]:
        actual = recommend_for_users(restored, [*fitted.user_ids, 'unknown-user'], k=99, exclude_train_items=exclude)
        expected = recommend_for_users(fitted, [*fitted.user_ids, 'unknown-user'], k=99, exclude_train_items=exclude)
        assert actual.recommendations == expected.recommendations
        for a,b in zip(actual.scores, expected.scores, strict=True):
            np.testing.assert_allclose(a,b,atol=1e-12)
        # Independent SVD/NMF dot-product oracle after reload.
        if restored.method in {'svd','nmf'}:
            for u,items,scores in zip(actual.user_ids,actual.recommendations,actual.scores,strict=True):
                if u == 'unknown-user':
                    continue
                values = fitted.user_factors_[fitted.user_index_[u]] @ fitted.item_factors_.T
                if fitted.method == 'svd':
                    values += fitted.global_mean_
                np.testing.assert_allclose(scores,[values[fitted.item_index_[i]] for i in items],atol=1e-12)


@pytest.mark.parametrize('meta', ['{broken', '[]', 'null'])
def test_invalid_metadata_is_a_validation_error(tmp_path, meta):
    (tmp_path/'meta.json').write_text(meta)
    (tmp_path/'recommender_plan.joblib').write_bytes(b'not-a-pickle')
    with pytest.raises(ValidationError, match='metadata'):
        load_recommender_bundle(tmp_path, trusted=True)


@pytest.mark.parametrize('boundary', ['save', 'load'])
@pytest.mark.parametrize('damage', [
    'user_count', 'item_count', 'user_index', 'item_index',
    'duplicate_users', 'duplicate_items', 'matrix_shape', 'unhashable_users', 'unhashable_items',
])
def test_catalog_contract_refuses_misattributed_scores(fitted, tmp_path, boundary, damage):
    # A stale nonempty mapping used to select another user's ratings silently.
    damaged = copy.deepcopy(fitted)
    if damage == 'user_count':
        damaged.n_users += 1
    elif damage == 'item_count':
        damaged.n_items += 1
    elif damage == 'user_index':
        damaged.user_index_ = dict(zip(damaged.user_ids, reversed(range(damaged.n_users)), strict=True))
    elif damage == 'item_index':
        damaged.item_index_ = dict(zip(damaged.item_ids, reversed(range(damaged.n_items)), strict=True))
    elif damage == 'duplicate_users':
        damaged.user_ids = (damaged.user_ids[1], *damaged.user_ids[1:])
    elif damage == 'duplicate_items':
        damaged.item_ids = (damaged.item_ids[1], *damaged.item_ids[1:])
    elif damage.startswith('unhashable_'):
        name = 'user_ids' if damage.endswith('users') else 'item_ids'
        ids = getattr(damaged, name)
        setattr(damaged, name, ([ids[0]], *ids[1:]))
    else:
        damaged.matrix_ = damaged.matrix_[:, :-1]
    root = tmp_path / 'bundle'
    if boundary == 'save':
        with pytest.raises(ValidationError, match='catalog|matrix'):
            save_recommender_bundle(root, damaged)
        assert not root.exists()
    else:
        save_recommender_bundle(root, fitted)
        payload = joblib.load(root / 'recommender_plan.joblib')
        payload['plan'] = damaged
        joblib.dump(payload, root / 'recommender_plan.joblib')
        with pytest.raises(ValidationError, match='catalog|matrix'):
            load_recommender_bundle(root, trusted=True)


@pytest.mark.parametrize('exclude', [False, True])
def test_restored_scores_match_independent_catalog_arithmetic(fitted, tmp_path, exclude):
    restored = load_recommender_bundle(save_recommender_bundle(tmp_path, fitted), trusted=True)
    actual = recommend_for_users(restored, fitted.user_ids, k=99, exclude_train_items=exclude)
    matrix = fitted.matrix_
    for u, (items, scores) in enumerate(zip(actual.recommendations, actual.scores, strict=True)):
        rated = matrix[u] != 0
        if fitted.method in {'svd', 'nmf'}:
            expected = fitted.user_factors_[u] @ fitted.item_factors_.T
            if fitted.method == 'svd':
                expected = expected + fitted.global_mean_
        elif fitted.method == 'content':
            features = fitted.item_features_
            profile = (np.abs(matrix[u, rated]) @ features[rated]) / np.abs(matrix[u, rated]).sum()
            expected = (features @ profile) / (np.linalg.norm(features, axis=1) * np.linalg.norm(profile))
        else:
            vectors = matrix.T if fitted.method == 'item_knn' else matrix
            norms = np.linalg.norm(vectors, axis=1)
            cosine = (vectors @ vectors.T) / np.outer(norms, norms)
            if fitted.method == 'item_knn':
                weights = cosine[:, rated]
                expected = (weights @ matrix[u, rated]) / np.abs(weights).sum(axis=1)
            else:
                weights = cosine[u].copy()
                weights[u] = 0
                expected = (weights @ matrix) / (np.abs(weights) @ (matrix != 0))
        positions = [fitted.item_ids.index(item) for item in items]
        np.testing.assert_allclose(scores, expected[positions], rtol=1e-10, atol=1e-12)
        assert len(items) == (int((~rated).sum()) if exclude else fitted.n_items)
        if exclude:
            assert all(not rated[position] for position in positions)
