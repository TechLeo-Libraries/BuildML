"""Real fitted recommender restoration preserves catalog identity and scores."""
import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.recommenders.checkpoint import load_recommender_bundle, save_recommender_bundle
from buildml.recommenders.recommend import recommend_for_users


@pytest.fixture(params=['item_knn', 'user_knn', 'svd', 'nmf', 'content'])
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
