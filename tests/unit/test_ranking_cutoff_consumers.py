"""Real retrieval/recommendation consumers refuse lossy ranking cutoffs."""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.rag.corpus import corpus_from_documents
from buildml.rag.evaluate import evaluate_retrieval
from buildml.rag.index import build_index
from buildml.rag.results import Document
from buildml.rag.retrieve import retrieve
from buildml.rag.types import RetrieveConfig

BAD_K = [True, np.bool_(False), 1.5, 1., '2', np.nan, np.inf, 0, -1]


@pytest.fixture(scope='module', params=['item_knn', 'user_knn', 'svd', 'nmf', 'content'])
def trained_session(request):
    frame = pd.DataFrame([dict(user=f'u{u}', item=f'i{i}', rating=float(1+(u*3+i*5)%7),
                              f1=float(i%3), f2=float(i)) for u in range(6) for i in range(8)])
    session = (Session.ingest(frame).set_roles({'user': 'id', 'item': 'id', 'rating': 'target',
                                               'f1': 'feature', 'f2': 'feature'})
               .split(test_size=.3, validation_size=.15, random_state=7))
    session.recommender.fit(backend='sklearn', method=request.param, user_column='user',
                            item_column='item', rating_column='rating', n_neighbors=99,
                            n_factors=3, item_feature_columns=['f1', 'f2'], random_state=5)
    return session


@pytest.mark.parametrize('operation', ['recommend', 'evaluate'])
@pytest.mark.parametrize('k', [*BAD_K, None])
def test_recommender_cutoff_refusals_preserve_previous_results(trained_session, operation, k):
    options = dict(user_ids=trained_session.recommender_plan.user_ids) if operation == 'recommend' else {}
    previous = getattr(trained_session.recommender, operation)(k=1, **options)
    attribute = '_recommender_recommend_result' if operation == 'recommend' else '_recommender_eval_result'
    history = copy.deepcopy(trained_session.history)
    matrix = trained_session.recommender_plan.matrix_.copy()
    with pytest.raises(ValidationError, match='k'):
        getattr(trained_session.recommender, operation)(k=k, **options)
    assert getattr(trained_session, attribute) is previous
    assert trained_session.history == history
    np.testing.assert_array_equal(trained_session.recommender_plan.matrix_, matrix)


@pytest.mark.parametrize('flag', ['false', 0, 1, None, np.nan])
def test_recommender_filter_refusals_preserve_previous_results(trained_session, flag):
    users = trained_session.recommender_plan.user_ids
    previous = trained_session.recommender.recommend(user_ids=users, k=1)
    history = copy.deepcopy(trained_session.history)
    with pytest.raises(ValidationError, match='exclude_train_items'):
        trained_session.recommender.recommend(user_ids=users, k=1, exclude_train_items=flag)
    assert trained_session._recommender_recommend_result is previous
    assert trained_session.history == history


@pytest.mark.parametrize('k', [1, np.int64(3), 99])
def test_holdout_metrics_are_macro_user_averages_with_integer_cutoffs(trained_session, k):
    plan = trained_session.recommender_plan
    frame = trained_session.partition('test')
    relevant = {}
    for row in frame.itertuples(index=False):
        if row.user in plan.user_ids and row.item in plan.item_ids:
            relevant.setdefault(row.user, set()).add(row.item)
    assert len({len(items) for items in relevant.values()}) > 1
    rec = trained_session.recommender.recommend(user_ids=list(relevant), k=k)
    values = []
    for user, items in zip(rec.user_ids, rec.recommendations, strict=True):
        bits = [item in relevant[user] for item in items]
        hits = sum(bits)
        dcg = sum(hit/math.log2(rank+1) for rank, hit in enumerate(bits, 1))
        ideal = sum(1/math.log2(rank+1) for rank in range(1, min(int(k), len(relevant[user]))+1))
        ap = sum(sum(bits[:rank])/rank for rank, hit in enumerate(bits, 1) if hit)/min(int(k), len(relevant[user]))
        values.append([hits/int(k), hits/len(relevant[user]), dcg/ideal, ap])
    result = trained_session.recommender.evaluate(k=k)
    for name, expected in zip(['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k'], np.mean(values, axis=0), strict=True):
        assert result.metrics[name] == pytest.approx(expected)
    assert any('macro' in line.lower() for line in result.disclosures)
    assert result.k == int(k)
    json.dumps(result.to_dict(), allow_nan=False)
    flag_result = trained_session.recommender.recommend(user_ids=list(relevant), k=k, exclude_train_items=np.bool_(True))
    assert json.loads(json.dumps(flag_result.to_dict(), allow_nan=False))['excluded_train_items'] is True


@pytest.fixture(scope='module')
def index():
    texts = ['apple apple orchard red harvest', 'apple orchard apple apple green fruit',
             'ocean blue sailing boat waves', 'apple garden pear pear pear pear']
    corpus = corpus_from_documents([Document(doc_id=f'd{i}', text=text) for i, text in enumerate(texts)])
    return build_index(corpus, embedder='hashing', chunk_size=128, chunk_overlap=0)


@pytest.mark.parametrize('mode', ['dense', 'bm25', 'hybrid'])
@pytest.mark.parametrize('path', ['override', 'config', 'evaluate'])
@pytest.mark.parametrize('k', BAD_K)
def test_retrieval_cutoffs_refuse_coercion_without_mutating_vectors(index, mode, path, k):
    vectors = index.store.embeddings.copy()
    with pytest.raises(ValidationError, match='k'):
        if path == 'evaluate':
            evaluate_retrieval(index, {'apple orchard': ['d0']}, k=k, mode=mode)
        else:
            options = dict(k=k, mode=mode) if path == 'override' else dict(config=RetrieveConfig(k=k, mode=mode))
            retrieve(index, 'apple orchard', **options)
    np.testing.assert_array_equal(index.store.embeddings, vectors)


@pytest.mark.parametrize('mode', ['dense', 'bm25', 'hybrid'])
@pytest.mark.parametrize('k', [1, np.int64(2), 99])
def test_retrieval_cutoffs_preserve_scores_ranks_and_evaluation(index, mode, k):
    query = 'apple orchard'
    result = retrieve(index, query, k=k, mode=mode)
    configured = retrieve(index, query, config=RetrieveConfig(k=k, mode=mode))
    assert [h.chunk_id for h in result.hits] == [h.chunk_id for h in configured.hits]
    assert len(result.hits) == min(int(k), len(index.chunks))
    assert result.k == int(k) and [h.rank for h in result.hits] == list(range(1, len(result.hits)+1))
    if mode == 'dense':
        vector = index.embedder.encode([query])[0]
        scores = index.store.embeddings @ (vector/np.linalg.norm(vector))
        expected = sorted(scores, reverse=True)[:int(k)]
        np.testing.assert_allclose([h.score for h in result.hits], expected, atol=1e-6)
    relevant = {'d0', 'd1'}
    ids = [h.doc_id for h in result.hits]
    bits = [doc in relevant for doc in ids]
    evaluation = evaluate_retrieval(index, {query: list(relevant)}, k=k, mode=mode)
    assert evaluation.recall_at_k == pytest.approx(sum(bits)/len(relevant))
    assert evaluation.mrr == pytest.approx(next((1/rank for rank, bit in enumerate(bits, 1) if bit), 0.))
    dcg = sum(bit/math.log2(rank+1) for rank, bit in enumerate(bits, 1))
    ideal = sum(1/math.log2(rank+1) for rank in range(1, min(int(k), len(relevant))+1))
    assert evaluation.ndcg_at_k == pytest.approx(dcg/ideal)
    assert retrieve(index, query, k=None, config=RetrieveConfig(k=1, mode=mode)).k == 1


@pytest.mark.parametrize('mode', ['dense', 'bm25', 'hybrid'])
@pytest.mark.parametrize('path', ['config', 'evaluate'])
def test_none_is_not_a_configured_or_evaluation_cutoff(index, mode, path):
    with pytest.raises(ValidationError, match='k'):
        if path == 'config':
            retrieve(index, 'apple orchard', config=RetrieveConfig(k=None, mode=mode))
        else:
            evaluate_retrieval(index, {'apple orchard': ['d0']}, k=None, mode=mode)
